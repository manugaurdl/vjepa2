# Project Memory

Read this at the start of every new chat.

---

## What This Project Is

Training a recurrent encoder (RNN with predictive coding) on Something-Something v2 (SSv2) using frozen DINO (ViT-S/14, 384-dim) features. Goal: compress short video clips into a compact spatio-temporal state good enough for long-horizon forecasting (stage 2).

Proxy task: next-frame prediction in DINO feature space. No action classification in the final design — CE loss biases the state toward SSv2 action prototypes.

Codebase: built on top of vjepa. My code is in `root/`. Entry point: `train.py`.

---

## Workflow: How We Track Everything

### The Two IDs

Every result is identified by two IDs:

- **wandbID** — the 8-char wandb run ID (e.g. `e6esmgmu`). Uniquely identifies a trained model. Look up training config and checkpoint in `docs/exp_progress.md`.
- **evalID** — the name of the evaluation type (e.g. `ucf101_probe`, `temporal_shuffle`, `static_dynamic`). Defined in `docs/repo_context.md` with exact run commands.

A result = wandbID + checkpoint + evalID + number. All three must be recorded.

### The Three Files

| File | Purpose |
|------|---------|
| `docs/repo_context.md` | Static reference. Eval type definitions, training templates, data paths, conventions. Rarely changes. |
| `docs/exp_progress.md` | Append-only knowledge base. Per-model entries: training config, checkpoint, results per evalID, takeaways. Source of truth for all numbers. |
| `docs/memory.md` | This file. Workflow reference for new chats. |

### Meeting / Results Docs (e.g. `docs/april_3_meet.md`)

Presentation layer only — tables for meetings/slides. Every number in these docs must cite wandbID + evalID. The authoritative number lives in `exp_progress.md`. If you see a table without IDs, treat it as unverified.

---

## Before Reporting Any Number

Checklist:
1. Which wandbID (model)?
2. Which checkpoint — `best.pt` or `last.pt`? (This caused Table 2 to be wrong — numbers were from `last.pt` at ~100k steps, not fully trained)
3. Which evalID?
4. Run the eval fresh — don't copy from memory or old docs
5. Log result to `exp_progress.md` under the model's entry

---

## Conventions

```bash
# Always activate venv first
source /home/manu/vjepa2/.venv/bin/activate

# Always set PYTHONPATH for root/ imports
PYTHONPATH=/home/manu/vjepa2 python root/evals/<script>.py ...

# wandb API — always use timeout=60
api = wandb.Api(timeout=60)
```

- Checkpoints: `/nas/manu/vjepa2/outputs/<run_name>_<wandbID>/{best,last}.pt`
- Cached DINO features: `/nas/manu/ssv2/dino_feats/vits14/`
  - CLS: `{train,validation}.pt` — shape `(N, 8, 384)`
  - Patches: `{train,validation}_patches.pt` — shape `(N, 8, 256, 384)`

---

## Eval Types (evalIDs)

Full commands in `docs/repo_context.md`. Brief summary:

| evalID | Script | What it measures |
|--------|--------|-----------------|
| `ucf101_probe` | `eval_transfer.py --mode probe` | Transfer: freeze encoder, linear head on UCF101 |
| `temporal_shuffle` | `root/evals/temporal_shuffle_test.py` | Does model rely on temporal order? (ratio = shuffled/normal pred_loss) |
| `static_dynamic` | `root/evals/static_dynamic_decomposition.py` | Is improvement over copy concentrated on dynamic patches? (patches only) |
| `ood_decay` | `eval_transfer.py --mode decay` | Does pred_error_l2 curve generalize to OOD data? (tbd) |

---

## Current Models (as of April 2026)

| wandbID | Description | Checkpoint |
|---------|-------------|------------|
| `zyvsy8gk` | RNN, CE+pred, CLS, learned space | `update=w(error)_L2weight1e-1_zyvsy8gk/best.pt` |
| `2ldiw9xk` | RNN, pred only, CLS, DINO space | `pred_in_dino_space_2ldiw9xk/last.pt` |
| `e6esmgmu` | RNN, pred only, patches, DINO space | `patch_pred_dino_space_e6esmgmu/last.pt` |
| `tj9x820q` | RNN, CE+pred, patches, learned space | `patch_ce_pred_tj9x820q/best.pt` |
| `ud2ncxlq` | Causal Transformer, pred only, CLS, DINO space | `causal_pred_cls_ud2ncxlq/last.pt` |
| `r55x2lcn` | Causal Transformer, pred only, patches, DINO space | `causal_pred_patches_r55x2lcn/last.pt` (check) |

Full training configs and results in `docs/exp_progress.md`.

---

## Key Results So Far

See `docs/april_3_meet.md` for full tables. TL;DR:

- DINO-space pred > CE+pred learned space on UCF101 transfer (85.4% vs 84.0% for CLS)
- CLS consistently better than patches on UCF101 (~4-7 pts gap)
- Causal Transformer CLS (`ud2ncxlq`) pred_loss=100.1 — better than RNN (`2ldiw9xk`) 176.9 → state IS a bottleneck for CLS (contradicts earlier stale numbers)
- Patch model (`e6esmgmu`): all improvement over copy comes from temporal order (shuffled ≈ copy baseline)
- CLS model: multi-frame aggregation survives shuffling — model learns both aggregation and some dynamics

---

## Active To Dos

See `docs/april_3_meet.md` → To Do section. Key items:
1. Make pred-only work without CE loss (SigREG or EMA)
2. More expressive W_pred for DINO-space patches (DINO-WM style)
3. Mean-pooled patch experiment (see `docs/meanpool_patches_experiment.md`)
4. OOD decay curve on `2ldiw9xk` and `e6esmgmu`
