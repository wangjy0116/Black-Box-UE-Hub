import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def _valid(x):
    if x is None:
        return False
    if isinstance(x, list):
        return x[-1] is not None
    if isinstance(x, str):
        if x.strip() == "":
            return False
        if "none" in x.lower() or "abstain" in x.lower() or "none of the above" in x.lower():
            return False
    return True


def load_valid_ids(filter_path: str, keys):
    with open(filter_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [rec["id"] for rec in data if all(_valid(rec.get(key)) for key in keys)]


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if mask.any():
            acc, conf = y_true[mask].mean(), y_prob[mask].mean()
            ece += abs(acc - conf) * mask.sum() / len(y_prob)
    return ece


def compute_brier(y_true, y_prob):
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return np.mean((y_prob - y_true) ** 2)

def main():
    parser = argparse.ArgumentParser(description="Batch-compute AUROC / ECE / Brier for every JSON in a folder")
    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to run")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names to run")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods to run")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--key", nargs="+", default=["pred_score", "pred_answer", "label"], help="Keys to check for validity when filtering samples")
    parser.add_argument("--yprob_key", default="pred_score")
    parser.add_argument("--ytrue_key", default="label")
    args = parser.parse_args()

    summary_dir = os.path.join(f"{args.input_dir}-eval")
    os.makedirs(summary_dir, exist_ok=True)
    summary_rows = []

    print(f"{'FILE':<50} {'ACC':>8} {'AUROC':>8} {f'ECE({args.bins})':>10} {'BRIER':>8}")
    print("-" * 120)

    for method in args.methods:
        summary_rows = []
        out_csv = os.path.join(summary_dir, f"{method}.csv")
        for ds in args.datasets:
            in_dir = os.path.join(args.input_dir, method, ds)
            if not os.path.isdir(in_dir):
                continue

            json_files = sorted([f for f in os.listdir(in_dir) if f.endswith(".json")])
            if not json_files:
                continue

            for fname in json_files:
                md = fname.split("_")[0]
                fpath = os.path.join(in_dir, fname)
                try:
                    valid_ids = load_valid_ids(fpath, args.key)
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                filtered = [rec for rec in data if rec.get("id") in valid_ids]
                if not filtered:
                    print(f"{fpath:<50} (skip, no valid samples)")
                    continue

                y_true_list = []
                y_prob_list = []
                for rec in filtered:
                    yt = rec.get(args.ytrue_key)
                    yp = rec.get(args.yprob_key)
                    if yt is None or yp is None:
                        continue
                    y_true_list.append(float(yt))
                    y_prob_list.append(float(yp))

                if len(y_true_list) == 0 or len(y_prob_list) == 0:
                    print(f"{fpath:<50} (skip, unable to parse y_true or y_prob)")
                    continue

                y_true = np.asarray(y_true_list, dtype=float)
                y_prob = np.asarray(y_prob_list, dtype=float)

                try:
                    auroc = roc_auc_score(y_true, y_prob)
                except ValueError:
                    auroc = float("nan")

                acc = float(np.mean(y_true))
                ece = compute_ece(y_true, y_prob, args.bins)
                brier = compute_brier(y_true, y_prob)

                print(
                    f"{fpath:<50} "
                    f"{acc:8.4f} {auroc:8.4f} {ece:10.4f} {brier:8.4f}"
                )

                summary_rows.append(
                    {
                        "dataset": ds,
                        "model": md,
                        "acc": acc * 100,
                        "auroc": auroc * 100,
                        f"ece_{args.bins}": ece * 100,
                        "brier": brier * 100
                    }
                )

        if not summary_rows:
            raise RuntimeError("No usable JSON results found (check the directory structure or arguments)")

        dataset_order = {ds: i for i, ds in enumerate(args.datasets)}
        model_order = {md: i for i, md in enumerate(args.models)}
        df = pd.DataFrame(summary_rows)
        df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
        df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
        df = df.sort_values(by=["dataset", "model"]).reset_index(drop=True)

        for col in ["acc", "auroc", f"ece_{args.bins}", "brier"]:
            df[col] = df[col].round(2)

        df.to_csv(out_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
