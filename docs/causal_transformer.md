# Causal Transformer Predictor

## Purpose
Upper-bound baseline for next-frame prediction. Unlike the RNN which compresses history into a fixed-size state, the causal transformer attends to all past frames directly. The gap between RNN and this baseline measures how much information the RNN state loses.

## Architecture
Standard transformer encoder with causal masking:
- Input: DINO features `(B, T, D)` for CLS or `(B, T, S, D)` for patches
- For patches, tokens are flattened to `(B, T*S, D)` = `(B, 2048, 384)`
- Learned frame positional embeddings (shared across patches within a frame)
- Learned spatial positional embeddings for patches (when S > 1)
- Causal mask: position `i` can attend to position `j` only if `frame(j) <= frame(i)`. Patches within the same frame can see each other.
- Prediction: output at position `t` predicts the input at position `t+1` via a linear head
- Loss: `||pred[t] - target[t+1]||²`, same L2 as the RNN, skipping t=0

Returns the same 6-tuple as `VideoRNNTransformerEncoder` (with dummy zeros for update_gates, update_norms, r_novelty) so it plugs into the existing training loop.

Config: `encoder.causal_transformer.depth` (default 4), `encoder.causal_transformer.n_heads` (default 8). ~7.3M params with depth=4, dim=384.

## Files changed
- `root/models/causal_transformer.py` — new file, `CausalTransformerPredictor` class
- `root/models/encoder.py` — added `"causal_transformer"` branch in `build_encoder`, imports `CausalTransformerPredictor`
- `root/models/model.py` — `encoder_type in ("rnn", "causal_transformer")` for forward path and head_in_dim; passes `_n_frames` and `_n_patches` to encoder config; skips per-sample val diagnostics for non-RNN encoders
- `root/config/base.yaml` — added `causal_transformer` section under `encoder`

## Usage
```bash
# CLS
python train.py --config base --wandb.run_name causal_pred_cls --data_dir /nas/manu --output_dir /nas/manu/vjepa2/outputs --load_cache_feats --no-action_classification --encoder.type causal_transformer --epochs 100

# Patches
python train.py --config base --wandb.run_name causal_pred_patches --data_dir /nas/manu --output_dir /nas/manu/vjepa2/outputs --load_cache_feats --use_patch_tokens --no-action_classification --encoder.type causal_transformer --batch_size 8 --val_batch_size 8 --epochs 100
```
