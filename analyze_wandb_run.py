"""Fetch and summarize metrics from a wandb run."""

import argparse
import wandb


def fetch_run(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)
    history = run.history()
    config = run.config
    summary = run.summary._json_dict
    return run, history, config, summary


def summarize(run_path: str):
    run, history, config, summary = fetch_run(run_path)

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"Config: {config}")
    print()

    # Final eval metrics
    eval_cols = [c for c in history.columns if c.startswith("eval/")]
    train_cols = [c for c in history.columns if c.startswith("trainer/")]

    if eval_cols:
        last_eval = history[eval_cols].dropna(how="all").iloc[-1]
        print("=== Last Eval ===")
        for col in sorted(eval_cols):
            val = last_eval.get(col)
            if val is not None and not isinstance(val, dict):
                print(f"  {col}: {val:.6f}")

    if train_cols:
        last_train = history[train_cols].dropna(how="all").iloc[-1]
        print("\n=== Last Train ===")
        for col in sorted(train_cols):
            val = last_train.get(col)
            if val is not None and not isinstance(val, dict):
                print(f"  {col}: {val:.6f}")

    # Eval loss trajectory
    if "eval/total_loss" in history.columns:
        eval_loss = history[["eval/total_loss"]].dropna()
        print(f"\n=== Eval Loss Trajectory ({len(eval_loss)} evals) ===")
        for i, row in eval_loss.iterrows():
            print(f"  step {i}: {row['eval/total_loss']:.6f}")

    if "eval/acc" in history.columns:
        eval_acc = history[["eval/acc"]].dropna()
        if eval_acc["eval/acc"].max() > 0:
            print(f"\n=== Eval Acc Trajectory ({len(eval_acc)} evals) ===")
            for i, row in eval_acc.iterrows():
                print(f"  step {i}: {row['eval/acc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path, e.g. manugaur/fair_project/qqm17c1k")
    args = parser.parse_args()
    summarize(args.run_path)
