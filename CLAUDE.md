# Project Workflow

Project uses a two-ID system for reproducibility:
- **wandbID** (8-char, e.g. `e6esmgmu`) = trained model. Training config + checkpoint in `docs/exp_progress.md`.
- **evalID** (e.g. `temporal_shuffle`, `ucf101_probe`) = eval type. Commands in `docs/repo_context.md`.

Result = wandbID + checkpoint file + evalID + number. All must be recorded.

**Key files:**
- `docs/memory.md` — read at start of every new chat. Full workflow, current models, active todos.
- `docs/exp_progress.md` — source of truth for all experiment results and takeaways.
- `docs/repo_context.md` — eval type definitions and run commands.

**Before reporting any number:**
1. Specify wandbID + checkpoint (best.pt vs last.pt). Many pred-only runs only have last.pt (best.pt save in `train.py:349` is gated on `acc > best_acc`, never triggered when action_classification=False).
2. Run the eval fresh, don't copy from memory.
3. If the eval reads `/nas/manu/ssv2/dino_feats/vits14/validation.pt` directly, **truncate to the row count of `ssv2/data/validation.csv` (currently 24777) before averaging** — the file is zero-padded to 168913. See `docs/repo_context.md` "Cached features" for the full story.
4. Log to exp_progress.md.

**Why (Table 2 history):** Table 2 went through two wrong values. The original `wandb`-logged number for Causal Transformer was 517 (correct — train.py's val loop iterates only real CSV rows). It was then "recomputed" to 100.1 by running `temporal_shuffle_test.py` over the raw `validation.pt` file, which silently averaged in ~144k zero-padded rows and deflated the mean ~7×. The 100.1 figure was wrong; the 517 was right. The truncation fix is now in place in `temporal_shuffle_test.py` and at the cache save site in `train.py`.
