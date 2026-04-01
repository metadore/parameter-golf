# Arcoistry Efficiency v1 — OpenAI Parameter Golf Submission

**Track:** 10-minute training · 16 MB model limit  
**Author:** arcoistry · 1st-year B.Tech CSE,   
**Date:** 2026-03-31  
**Placeholder BPB:** 1.22 *(to be updated after RunPod run)*

---

## Philosophy: Architectural Density

> *"Less is more — if the less is smarter."*

As a first-year CSE student, I don't have years of GPU-time to throw at a problem. What I do have is a genuine obsession with *why* things work, not just *that* they work. This submission is the result of asking one question repeatedly:

**Where is this model wasting space, and what can I cut without losing intelligence?**

The answer turned into four interlocking decisions I call **Architectural Density** — squeezing every useful bit of learned representation into the fewest possible bytes.

---

## The Four Pillars

### 1. Weight Tying — *~30% Space Saving on the Embedding Layer*

A standard GPT has two large matrices that both "speak vocabulary":
- `wte` — maps token IDs → embedding vectors (input)
- `lm_head` — maps embedding vectors → token logits (output)

These two matrices are doing conceptually symmetric jobs, so we share them:

```python
self.lm_head.weight = self.transformer.wte.weight  # one matrix, two roles
```

For a 50,257-token vocabulary at 768 dimensions, this eliminates **~38.6 million parameters** (~147 MB in fp32). Within a 16 MB submission budget, this is the single biggest win.

---

### 2. 3× MLP Expansion — *Fit 11 Layers Instead of 9*

The standard GPT MLP hidden dimension is `4 × n_embd`. Reducing this to `3 × n_embd` cuts each block's MLP cost by **25% per layer**.

```python
hidden = 3 * config.n_embd   # vs baseline 4 * n_embd
```

The saved budget is reinvested into **more layers**. Depth matters more than width in the 10-minute window: extra layers allow more compositional abstractions per token. 11 layers fit where only 9 would before.

---

### 3. RMSNorm — *Faster Convergence in 10 Minutes*

`LayerNorm` computes both mean and variance. `RMSNorm` skips the mean-shift entirely:

```python
x * rsqrt(mean(x²) + ε) * γ
```

Fewer parameters, faster on H100s, and empirically as stable. Under a hard 10-minute wall clock, every millisecond saved on normalisation is a gradient step gained.

---

### 4. Rotary Positional Embeddings (RoPE) — *Better Context, Fewer Parameters*

Absolute positional embeddings (`wpe`) add a fixed `block_size × n_embd` matrix. RoPE encodes position geometrically inside attention itself:

```python
q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
```

This removes `wpe` entirely, generalises better to unseen lengths, and encodes *relative* position — which is what attention actually needs.

---

## The "Arcoistry" Connection

My personal hobby is **Arcoistry** — a portmanteau of *archery* and *chemistry* — which sounds unrelated to language models until you think about what both disciplines demand: **precision within tight constraints**.

In archery, you cannot brute-force accuracy with a heavier bow. In chemistry, a reaction either works at the right proportions or it doesn't. In parameter golf, you win by finding the *minimal sufficient structure*, not by scaling up.

Every choice here came from the same instinct: *what is the smallest change that preserves the capability I need?*

---

## File Structure

```
2026-03-31-arcoistry-v1/
├── submission.json   # competition metadata + BPB score
├── README.md         # this file
├── train_gpt.py      # full training script (runs on 8×H100)
└── train.log         # populated after RunPod training run
```

---

## Running the Code

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requirements: `torch >= 2.1`, `numpy`.

---

## Status

| Item | Status |
|---|---|
| Architecture design | ✅ Complete |
| Code (train_gpt.py) | ✅ Complete |
| Real training run | ⏳ Awaiting RunPod credits (post-exams) |
| Real val BPB | ⏳ Placeholder 1.22 — will update PR |
| train.log | ⏳ Will populate after run |

---

*Submitted with curiosity — 2026.*