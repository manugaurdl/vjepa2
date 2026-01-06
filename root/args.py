from __future__ import annotations

import argparse
import os
import yaml

def prepare_config():
    #load config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, required=True, help='Name of config file in root/config')
    
    args, unknown_args = pre_parser.parse_known_args()

    #load .yaml
    config_path = os.path.join('root', 'config', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # create and populate main parser
    parser = argparse.ArgumentParser(parents=[pre_parser])
    
    ### args that are not in the config file
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    
    for key, value in config_data.items():
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', type=bool, default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    final_args = parser.parse_args()    
    return final_args


def _resolve_sampling_kwargs(args: argparse.Namespace) -> dict:
    # The dataset enforces exactly one of (fps, duration, frame_step) is not None.
    if args.fps is None and args.duration is None and args.frame_step is None:
        args.frame_step = 4

    specified = [args.frame_step is not None, args.fps is not None, args.duration is not None]
    if sum(specified) != 1:
        raise ValueError("Must specify exactly one of: --frame-step, --fps, --duration (set the others to None).")
    return dict(frame_step=args.frame_step, fps=args.fps, duration=args.duration)



# def _parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Minimal video training script (hackable).")

#     parser.add_argument("--config", type=str, required=True, help="Path to the config file")
#     parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load (.pth)")
#     parser.add_argument("--eval_only", action='store_true', help="Run in evaluation mode only")
    
#     # --- I/O
#     p.add_argument("--data-path", action="append", required=True, help="CSV or NPY path. Repeatable.")
#     p.add_argument("--output-dir", default="./outputs/min_train", help="Where to write checkpoints/logs (rank 0).")
#     p.add_argument("--val-data-path", required=True, help="CSV or NPY path")
#     p.add_argument("--debug", action="store_true") 

#     # --- dataset / sampling
#     p.add_argument("--batch-size", type=int, default=4)
#     p.add_argument("--val-batch-size", type=int, default=16)
#     p.add_argument("--num-workers", type=int, default=8)
#     p.add_argument("--pin-mem", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--deterministic-loader", action=argparse.BooleanOptionalAction, default=True)

#     p.add_argument("--frames-per-clip", type=int, default=16)
#     p.add_argument("--num-clips", type=int, default=1)
#     p.add_argument("--random-clip-sampling", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--allow-clip-overlap", action=argparse.BooleanOptionalAction, default=False)

#     # Exactly one of (frame_step, fps, duration) must be specified by the dataset.
#     p.add_argument(
#         "--frame-step",
#         type=int,
#         default=None,
#         help="Sample every k-th frame (mutually exclusive with --fps/--duration). Default: 4 if neither --fps nor --duration is set.",
#     )
#     p.add_argument("--fps", type=int, default=None, help="Target sampling fps (mutually exclusive with --frame-step/--duration).")
#     p.add_argument("--duration", type=float, default=None, help="Clip duration in seconds (mutually exclusive with --frame-step/--fps).")

#     p.add_argument("--crop-size", type=int, default=224, help="Spatial crop size used by repo transforms.")

#     # --- model
#     p.add_argument("--dino-repo", type=str, default="facebookresearch/dinov2", help="Torch Hub repo for Dinov2.")
#     p.add_argument(
#         "--dino-model",
#         type=str,
#         default="dinov2_vits14",
#         help="Torch Hub entry (e.g. dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14).",
#     )
#     p.add_argument("--dino-pretrained", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--freeze-dino", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--mlp-hidden-dim", type=int, default=1024)
#     p.add_argument("--mlp-out-dim", type=int, default=512)

#     p.add_argument("--num-classes", type=int, required=True, help="Number of classes (labels in CSV must be integer in [0, num_classes)).")

#     # --- optimization
#     p.add_argument("--epochs", type=int, default=1)
#     p.add_argument("--lr", type=float, default=1e-4)
#     p.add_argument("--weight-decay", type=float, default=0.05)
#     p.add_argument("--grad-accum", type=int, default=1)

#     # --- logging / checkpointing
#     p.add_argument("--val-freq", type=int, default=8)
#     p.add_argument("--log-freq", type=int, default=20)
#     p.add_argument("--save-every-epochs", type=int, default=1)

#     # --- distributed
#     p.add_argument("--dist-port", type=int, default=37129)

#     return p.parse_args()