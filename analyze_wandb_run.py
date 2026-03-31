"""Fetch and summarize metrics from a wandb run."""

import argparse
import pandas as pd
import wandb


def fetch_run(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)
    config = run.config
    summary = run.summary._json_dict
    history = run.history(pandas=True)
    return run, history, config, summary


def _is_scalar(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def summarize(run_path: str):
    run, history, config, summary = fetch_run(run_path)

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"Config: {config}")
    print()

    # Identify scalar-only columns (skip plotly figures / dicts)
    eval_cols = sorted([c for c in history.columns if c.startswith("eval/")])
    train_cols = sorted([c for c in history.columns if c.startswith("trainer/")])

    # Filter to scalar columns only
    scalar_eval_cols = [c for c in eval_cols if history[c].dropna().apply(_is_scalar).all()]
    scalar_train_cols = [c for c in train_cols if history[c].dropna().apply(_is_scalar).all()]

    # Last eval snapshot
    if scalar_eval_cols:
        eval_rows = history[scalar_eval_cols].dropna(how="all")
        if len(eval_rows) > 0:
            last_eval = eval_rows.iloc[-1]
            print("=== Last Eval ===")
            for col in scalar_eval_cols:
                val = last_eval.get(col)
                if pd.notna(val):
                    print(f"  {col}: {val:.6f}")

    # Last train snapshot
    if scalar_train_cols:
        train_rows = history[scalar_train_cols].dropna(how="all")
        if len(train_rows) > 0:
            last_train = train_rows.iloc[-1]
            print("\n=== Last Train ===")
            for col in scalar_train_cols:
                val = last_train.get(col)
                if pd.notna(val):
                    print(f"  {col}: {val:.6f}")

    # Eval loss trajectory
    if "eval/total_loss" in scalar_eval_cols:
        eval_loss = history[["_step", "eval/total_loss"]].dropna()
        print(f"\n=== Eval Loss Trajectory ({len(eval_loss)} evals) ===")
        for _, row in eval_loss.iterrows():
            print(f"  step {int(row['_step'])}: {row['eval/total_loss']:.6f}")

    # Eval pred_loss trajectory
    if "eval/pred_loss" in scalar_eval_cols:
        eval_pred = history[["_step", "eval/pred_loss"]].dropna()
        print(f"\n=== Eval Pred Loss Trajectory ({len(eval_pred)} evals) ===")
        for _, row in eval_pred.iterrows():
            print(f"  step {int(row['_step'])}: {row['eval/pred_loss']:.6f}")

    # Eval accuracy trajectory
    if "eval/acc" in scalar_eval_cols:
        eval_acc = history[["_step", "eval/acc"]].dropna()
        if eval_acc["eval/acc"].max() > 0:
            print(f"\n=== Eval Acc Trajectory ({len(eval_acc)} evals) ===")
            for _, row in eval_acc.iterrows():
                print(f"  step {int(row['_step'])}: {row['eval/acc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path, e.g. manugaur/fair_project/qqm17c1k")
    args = parser.parse_args()
    summarize(args.run_path)
