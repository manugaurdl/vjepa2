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
1. Specify wandbID + checkpoint (best.pt vs last.pt — this caused Table 2 to be wrong)
2. Run the eval fresh, don't copy from memory
3. Log to exp_progress.md

**Why:** Table 2 had stale numbers from ~100k steps (last.pt), not the fully trained model. Causal Transformer was logged as 517 but recomputed to 100.1.
