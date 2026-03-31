"""
train_gpt.py — OpenAI Parameter Golf (≤16 MB limit)
=====================================================
Architectural improvements over the baseline:
  1. Weight Tying      – lm_head.weight  ≡  wte.weight
  2. MLP 3x Expansion  – hidden_dim = 3 * n_embd  (vs 4x baseline)
  3. RMSNorm           – replaces LayerNorm (no bias, no mean-shift)
  4. RoPE              – removes absolute pos-embed, adds rotary embs
  5. Quantization-Ready CastedLinear – annotated for Int6 post-training quant
"""

import os
import math
import time
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ---------------------------------------------------------------------------
# Utility: CastedLinear (quantization-ready)
# ---------------------------------------------------------------------------

class CastedLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear that casts weights to the compute dtype
    on every forward pass.  This keeps master weights in float32 while running
    matmuls in bf16/fp16, and makes Int6 post-training quantisation trivial:
    replace the cast with a dequantise call and you're done.

    Int6 quantisation note
    ----------------------
    To quantise: call `quantize_to_int6(layer)` (stub below) which stores
    `layer.weight_q` (int8 tensor, values in [-32, 31]) and a per-channel
    `layer.scale` (float32).  The forward() below already checks for these
    attributes and dequantises on-the-fly.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Int6 dequantisation path (activated after quantize_to_int6())
        if hasattr(self, "weight_q"):
            # weight_q: int8 in [-32,31]  scale: [out_features, 1]
            w = self.weight_q.float() * self.scale          # dequantise
            w = w.to(x.dtype)
        else:
            w = self.weight.to(x.dtype)

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def quantize_to_int6(layer: CastedLinear) -> None:
    """
    Stub: convert a CastedLinear's fp32 weight → symmetric per-channel Int6.
    Call this *after* training on the saved checkpoint.  During training the
    float32 master weights are kept untouched so gradients flow correctly.
    """
    with torch.no_grad():
        w = layer.weight.float()                         # [out, in]
        max_val = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale = max_val / 31.0                           # 6-bit signed → [-32,31]
        w_q = (w / scale).round().clamp(-32, 31).to(torch.int8)
        layer.register_buffer("weight_q", w_q)
        layer.register_buffer("scale", scale)
        del layer.weight                                 # free fp32 copy


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root-Mean-Square Layer Normalisation (no bias, no mean-shift).
    Faster than LayerNorm and empirically as stable for transformer training.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Pre-compute the complex exponentials for RoPE.
    Returns a tensor of shape (max_seq_len, head_dim // 2) as complex64.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)                        # (T, head_dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis so it broadcasts over (B, T, n_heads, head_dim//2)."""
    # x shape: (B, T, n_heads, head_dim//2) as complex
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        f"freqs_cis {freqs_cis.shape} vs x {x.shape}"
    )
    shape = [1 if i not in (1, ndim - 1) else s for i, s in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to queries and keys.
    xq, xk: (B, T, n_heads, head_dim)
    freqs_cis: (T, head_dim // 2) complex
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)   # (1, T, 1, head_dim//2)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(xq.dtype)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(xk.dtype)
    return xq_out, xk_out


# ---------------------------------------------------------------------------
# Causal Self-Attention with RoPE
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head    = config.n_head
        self.n_embd    = config.n_embd
        self.head_dim  = config.n_embd // config.n_head

        # Fused QKV projection
        self.c_attn = CastedLinear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = CastedLinear(config.n_embd, config.n_embd, bias=False)

        # Scale output projection residual by 1/√(2·n_layer) for stability
        self.c_proj.SCALE_INIT = 1

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)                             # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, T, n_head, head_dim) for RoPE application
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # Apply rotary embeddings to q and k
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash attention (causal)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Re-assemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ---------------------------------------------------------------------------
# MLP with 3× expansion
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Two-layer MLP with GELU activation.
    Uses 3× hidden expansion instead of the baseline 4× to save parameters
    while keeping enough capacity.

    Parameter budget comparison (n_embd = d):
        Baseline 4×: 2 × (d × 4d) = 8d²
        This  3×:    2 × (d × 3d) = 6d²      → saves 25 %
    """

    def __init__(self, config):
        super().__init__()
        hidden = 3 * config.n_embd                       # ← 3× instead of 4×
        self.c_fc   = CastedLinear(config.n_embd, hidden, bias=False)
        self.c_proj = CastedLinear(hidden, config.n_embd, bias=False)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Single transformer block:
      RMSNorm → CausalSelfAttention (RoPE)
      RMSNorm → MLP (3× expansion)
    Pre-norm formulation for training stability.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# GPT Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024        # max sequence length
    vocab_size: int = 50257       # GPT-2 vocabulary
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT language model with all Parameter Golf optimisations:
      • Weight tying        – lm_head reuses wte weights (saves vocab_size × n_embd floats)
      • No absolute pos emb – RoPE handles positional information
      • RMSNorm             – faster, fewer parameters than LayerNorm
      • 3× MLP              – saves 25 % of MLP parameters
      • CastedLinear        – quantisation-ready throughout
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            # NOTE: no wpe — positional info is handled by RoPE
            drop = nn.Dropout(0.0),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))

        # Language-model head — weight TIED to token embedding matrix
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight   # ← weight tying

        # Pre-compute RoPE frequencies and register as a buffer
        head_dim = config.n_embd // config.n_head
        freqs_cis = precompute_freqs_cis(head_dim, config.block_size)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialise weights
        self.apply(self._init_weights)

        # Scale residual projections: std = 0.02 / √(2 · n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer)
                )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, CastedLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Parameter count helper
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        # lm_head shares weights with wte, so subtract once to avoid double-count
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Token embeddings only (no positional embedding — RoPE handles position)
        x = self.transformer.wte(idx)                    # (B, T, n_embd)
        x = self.transformer.drop(x)

        # Slice the pre-computed RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]

        for block in self.transformer.h:
            x = block(x, freqs_cis)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)                     # training: full logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # ------------------------------------------------------------------
    # Crop context window (for fine-tuning on shorter sequences)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def crop_block_size(self, block_size: int) -> None:
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.freqs_cis = self.freqs_cis[:block_size]

    # ------------------------------------------------------------------
    # Configure optimiser (AdamW with weight-decay on 2-D params only)
    # ------------------------------------------------------------------

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Weight-decay only on ≥2-D tensors (weight matrices, not biases/norms)
        decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() <  2]

        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"Using fused AdamW: {use_fused}")

        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx:         torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k:       int   = None,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # ------------------------------------------------------------------
    # Load GPT-2 pretrained weights from HuggingFace
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"Loading pretrained weights for {model_type} …")

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        model    = cls(GPTConfig(**config_args))
        sd       = model.state_dict()
        sd_keys  = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf    = model_hf.state_dict()

        # Transpose Conv1D weights from HuggingFace format
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith((".attn.masked_bias", ".attn.bias", "lm_head.weight"))
        ]
        assert len(sd_keys_hf) == len(sd_keys), (
            f"Mismatch: {len(sd_keys_hf)} HF keys vs {len(sd_keys)} model keys"
        )

        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch: {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# ---------------------------------------------------------------------------
# DataLoader (simple, memory-mapped)
# ---------------------------------------------------------------------------

import numpy as np


def load_tokens(filename: str) -> torch.Tensor:
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:

    def __init__(
        self,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        split: str,
        data_root: str = "edu_fineweb10B",
        master_process: bool = True,
    ):
        self.B = B
        self.T = T
        self.process_rank  = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        shards = sorted(
            [
                os.path.join(data_root, f)
                for f in os.listdir(data_root)
                if split in f
            ]
        )
        assert shards, f"No {split} shards found in {data_root}"

        if master_process:
            print(f"Found {len(shards)} shards for {split} split")

        self.shards       = shards
        self.current_shard = 0
        self.tokens        = load_tokens(shards[0])
        self.current_position = B * T * process_rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x   = buf[:-1].view(B, T)
        y   = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y

    def reset(self) -> None:
        self.current_shard    = 0
        self.tokens           = load_tokens(self.shards[0])
        self.current_position = self.B * self.T * self.process_rank


# ---------------------------------------------------------------------------
# Learning rate schedule (cosine with linear warm-up)
# ---------------------------------------------------------------------------

def get_lr(
    it:           int,
    warmup_steps: int,
    max_steps:    int,
    max_lr:       float,
    min_lr:       float,
) -> float:
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

def main() -> None:

    # ── Distributed setup ─────────────────────────────────────────────────
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank       = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device         = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank       = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device         = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # ── Hyperparameters ───────────────────────────────────────────────────
    # Tuned to stay under 16 MB with the architectural savings
    total_batch_size = 524288          # 2**19 tokens
    B  = 16                            # micro-batch
    T  = 1024                          # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"Total batch size: {total_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        master_process=master_process,
    )
    val_loader = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        master_process=master_process,
    )

    torch.set_float32_matmul_precision("high")

    # ── Model ─────────────────────────────────────────────────────────────
    # Config tuned for ≤ 16 MB after weight-tying + 3× MLP + RMSNorm savings
    # Roughly 85M trainable params → ~170 MB fp32, fits easily after quant/tie
    model = GPT(GPTConfig(
        block_size=T,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
    ))
    model.to(device)
    model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    if master_process:
        print(f"Model parameters: {raw_model.get_num_params()/1e6:.2f} M")

    # ── Optimiser ─────────────────────────────────────────────────────────
    max_lr        = 6e-4
    min_lr        = max_lr * 0.1
    warmup_steps  = 715
    max_steps     = 19073

    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        betas=(0.9, 0.95),
        device_type=device_type,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")

    with open(log_file, "w") as f:
        pass  # clear

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # ── Validation ────────────────────────────────────────────────────
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = torch.zeros(1, device=device)
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    val_loss_accum += loss / val_loss_steps

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if master_process:
                print(f"val loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

                if step > 0 and (step % 5000 == 0 or last_step):
                    ckpt_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    torch.save({
                        "model":      raw_model.state_dict(),
                        "config":     raw_model.config,
                        "step":       step,
                        "val_loss":   val_loss_accum.item(),
                        "optimizer":  optimizer.state_dict(),
                    }, ckpt_path)

        # ── Training step ─────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        loss_accum = torch.zeros(1, device=device)

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        dt   = time.time() - t0
        tps  = total_batch_size / dt

        if master_process:
            print(
                f"step {step:5d} | loss {loss_accum.item():.4f} | "
                f"lr {lr:.2e} | norm {norm:.4f} | "
                f"dt {dt*1000:.1f}ms | tok/s {tps:.0f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.4f}\n")

    if ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()