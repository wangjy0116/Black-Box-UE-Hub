import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _valid(x: Any) -> bool:
    if x is None:
        return False

    if isinstance(x, list):
        if len(x) == 0:
            return False
        return x[-1] is not None

    if isinstance(x, str):
        if x.strip() == "":
            return False

    return True


def _to_float(x: Any):
    if x is None:
        return None

    if isinstance(x, list):
        if len(x) == 0 or x[-1] is None:
            return None
        x = x[-1]

    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return None

    try:
        return float(x)
    except Exception:
        return None


def load_valid_ids(filter_path: str, keys):
    with open(filter_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_ids = []
    for rec in data:
        if all(_valid(rec.get(key)) for key in keys):
            valid_ids.append(rec["id"])

    return valid_ids


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.any():
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += abs(acc - conf) * mask.sum() / len(y_prob)

    return float(ece)


def compute_brier(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def main():
    parser = argparse.ArgumentParser(
        description="Batch-compute AUROC / ECE / Brier for JSON result files."
    )

    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to evaluate")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names to evaluate")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods to evaluate")
    parser.add_argument("--bins", type=int, default=10)

    parser.add_argument(
        "--key",
        nargs="+",
        default=["pred_score", "pred_answer", "label"],
        help="Keys to check for validity when filtering samples",
    )
    parser.add_argument("--yprob_key", default="pred_score")
    parser.add_argument("--ytrue_key", default="label")

    args = parser.parse_args()

    summary_dir = f"{args.input_dir}-eval"
    os.makedirs(summary_dir, exist_ok=True)

    print(
        f"{'METHOD':<15} {'DATASET':<15} {'MODEL':<30} "
        f"{'ACC':>8} {'AUROC':>8} {f'ECE({args.bins})':>10} {'BRIER':>8}"
    )
    print("-" * 120)

    all_has_result = False

    for method in args.methods:
        summary_rows = []
        out_csv = os.path.join(summary_dir, f"{method}.csv")

        for ds in args.datasets:
            in_dir = os.path.join(args.input_dir, method, ds)

            if not os.path.isdir(in_dir):
                print(f"[skip] directory not found: {in_dir}")
                continue

            json_files = sorted(
                [
                    f for f in os.listdir(in_dir)
                    if f.endswith(".json") and os.path.splitext(f)[0] in args.models
                ]
            )

            if not json_files:
                print(f"[skip] no matched model json in: {in_dir}")
                continue

            for fname in json_files:
                model_name = os.path.splitext(fname)[0]
                fpath = os.path.join(in_dir, fname)

                try:
                    valid_ids = set(load_valid_ids(fpath, args.key))

                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                except Exception as e:
                    print(f"[skip] failed to load {fpath}: {e}")
                    continue

                filtered = [rec for rec in data if rec.get("id") in valid_ids]

                if not filtered:
                    print(f"[skip] no valid samples: {fpath}")
                    continue

                y_true_list = []
                y_prob_list = []

                for rec in filtered:
                    yt = _to_float(rec.get(args.ytrue_key))
                    yp = _to_float(rec.get(args.yprob_key))

                    if yt is None or yp is None:
                        continue

                    y_true_list.append(yt)
                    y_prob_list.append(yp)

                if len(y_true_list) == 0:
                    print(f"[skip] unable to parse y_true or y_prob: {fpath}")
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
                    f"{method:<15} {ds:<15} {model_name:<30} "
                    f"{acc:8.4f} {auroc:8.4f} {ece:10.4f} {brier:8.4f}"
                )

                summary_rows.append(
                    {
                        "dataset": ds,
                        "model": model_name,
                        "acc": acc * 100,
                        "auroc": auroc * 100,
                        f"ece_{args.bins}": ece * 100,
                        "brier": brier * 100,
                        "num_samples": len(y_true),
                    }
                )

        if not summary_rows:
            print(f"[skip] no usable results for method: {method}")
            continue

        all_has_result = True

        df = pd.DataFrame(summary_rows)

        df["dataset"] = pd.Categorical(
            df["dataset"],
            categories=args.datasets,
            ordered=True,
        )

        df["model"] = pd.Categorical(
            df["model"],
            categories=args.models,
            ordered=True,
        )

        df = df.sort_values(by=["dataset", "model"]).reset_index(drop=True)

        for col in ["acc", "auroc", f"ece_{args.bins}", "brier"]:
            df[col] = df[col].round(2)

        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[saved] {out_csv}")

    if not all_has_result:
        raise RuntimeError(
            "No usable JSON results found. Please check --input_dir, --methods, --datasets, and --models."
        )


if __name__ == "__main__":
    main()