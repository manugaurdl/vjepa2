# Experiment Progress

## What We Care About
The primary goal is building a good **next-frame predictor** in DINO feature space, then analyzing **memory dynamics** to understand how the RNN accumulates temporal information. Classification accuracy is secondary — useful as a sanity check but not the main objective.

### Key metrics (priority order):
1. **eval/pred_loss & eval/pred_error_l2**: Is the model learning to predict next frames? Per-timestep plot reveals if prediction improves or collapses.
2. **eval/h_t_norm**: Should increase over time as novel info accumulates (e.g., 15→17 over 8 frames). Flat = state not accumulating = likely collapse.
3. **eval/update_norm**: Healthy: large at t=0, drops at t=1, then slightly increases as novel info appears. Drops to ~0 after t=0 = trivial solution.
4. **eval/cos_sim**: Direction similarity between consecutive hidden states. Should show meaningful variation. Flat = state direction frozen.
5. **eval/r_novelty**: Ratio of novel info in updates. Should be non-trivial across timesteps.
6. **eval/memory_l2**: L2 shift between consecutive states. Non-zero = meaningful temporal dynamics.
7. **eval/acc**: Only relevant when action_classification=True.

### Collapse detection checklist:
- pred_loss drops to near-zero suspiciously fast
- h_t_norm flat across timesteps
- update_norm → 0 after t=0
- cos_sim flat
- All together = representation collapse. Root cause: if prediction target is in a learned space (W_enc @ dino_feat), the encoder can collapse to make prediction trivial. Fix: predict in frozen DINO space (predict_in_dino_space=True).

## Eval Metrics Reference
- **eval/acc**: top-1 classification accuracy on SSv2 validation set (only meaningful when action_classification=True)
- **eval/ce_loss**: cross-entropy loss on validation set
- **eval/pred_loss**: L2 next-frame prediction error (mean over timesteps 1+, skipping t=0)
- **eval/total_loss**: ce_loss + pred_loss_weight * pred_loss (or just pred_loss when action_classification=False)
- **eval/pred_error_l2**: plotly line plot — per-timestep L2 prediction error across the clip (how well the RNN predicts the next DINO frame feature)
- **eval/update_gate**: plotly line plot — per-timestep gating values (how much the RNN state updates at each frame)
- **eval/update_norm**: plotly line plot — per-timestep L2 norm of the state update vector
- **eval/r_novelty**: plotly line plot — ratio of novel information in the update (u_novelty / u_total)
- **eval/memory_l2**: plotly line plot — L2 shift between consecutive hidden states ||h_t - h_{t-1}||
- **eval/cos_sim**: plotly line plot — cosine similarity between consecutive hidden states (direction stability)
- **eval/h_t_norm**: plotly line plot — norm of hidden state over time

## Train Metrics
- **trainer/loss, trainer/total_loss**: running average total loss
- **trainer/ce_loss**: running average CE loss
- **trainer/pred_loss**: running average weighted pred loss
- **trainer/lr**: current learning rate
- **trainer/iter_ms_avg**: average iteration time in ms

## Experiments

### 1. update=w(error)_L2weight1e-1
- **Config**: action_classification=True, next_frame_pred=True, pred_loss_weight=0.1, update_type=surprise, epochs=100
- **Goal**: train RNN with both CE and next-frame prediction loss
- **Result**: (pending / fill in)

### 2. next_frame_pred_only
- **Config**: action_classification=False, next_frame_pred=True, no pred_loss_weight scaling, epochs=100
- **Goal**: train RNN purely on next-frame prediction (no classification head)
- **Result**: (pending)

## Tools
- **analyze_wandb_run.py**: `python analyze_wandb_run.py <user>/<project>/<run_id>` — fetches and prints final eval/train metrics and loss trajectory from a wandb run

## Open Questions
-
