import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


def calc(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = (2 * prec * sens) / max(1e-12, prec + sens)
    j = sens + spec - 1
    return tp, tn, fp, fn, sens, spec, prec, f1, j


def find_best_threshold(y_true, y_prob, mode="youden", sens_min=0.85):
    best = None
    best_thr = None
    best_stats = None

    for thr in np.linspace(0.001, 0.999, 999):
        tp, tn, fp, fn, sens, spec, prec, f1, j = calc(y_true, y_prob, thr)

        if mode == "youden":
            key = (j, sens, spec, -thr)
        elif mode == "f1":
            key = (f1, sens, spec, -thr)
        elif mode == "sens_atleast":
            ok = sens >= sens_min
            key = (ok, spec, sens, -thr)
        else:
            raise ValueError("mode must be youden/f1/sens_atleast")

        if best is None or key > best:
            best = key
            best_thr = float(thr)
            best_stats = {
                "thr": float(thr),
                "sens": float(sens),
                "spec": float(spec),
                "precision": float(prec),
                "f1": float(f1),
                "youden_j": float(j),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }

    return best_stats


def resolve_pred_csv(exp_dir: Path, epoch: int, split: str) -> Path:
    p = exp_dir / f"{split}_preds_epoch{epoch:02d}.csv"
    if p.exists():
        return p

    # fallback: if val_preds not found for that epoch, pick the latest available
    cand = sorted(exp_dir.glob(f"{split}_preds_epoch*.csv"))
    if len(cand) == 0:
        raise FileNotFoundError(f"No {split}_preds_epoch*.csv found in {exp_dir}")
    return cand[-1]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--exp_dir", type=str, help="e.g. outputs/cls/resnetrs50_POST (auto uses best.pt epoch)")
    src.add_argument("--pred_csv", type=str, help="explicit csv path like outputs/cls/.../val_preds_epochXX.csv")

    ap.add_argument("--split", default="val", choices=["val", "test"], help="which preds file prefix to use")
    ap.add_argument("--mode", default="youden", choices=["youden", "f1", "sens_atleast"])
    ap.add_argument("--sens_min", type=float, default=0.85, help="only for sens_atleast")
    ap.add_argument("--save", action="store_true", help="save selected_threshold.json into exp_dir (if provided)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir) if args.exp_dir else None

    if args.pred_csv:
        pred_csv = Path(args.pred_csv)
        epoch = None
    else:
        assert exp_dir is not None
        best_pt = exp_dir / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Missing {best_pt}")
        ckpt = torch.load(best_pt, map_location="cpu")
        epoch = int(ckpt.get("epoch", 0))
        pred_csv = resolve_pred_csv(exp_dir, epoch, args.split)

    df = pd.read_csv(pred_csv)
    y_true = df["y_true"].astype(int).to_numpy()
    y_prob = df["y_prob"].to_numpy()

    best_stats = find_best_threshold(y_true, y_prob, mode=args.mode, sens_min=args.sens_min)

    out = {
        "mode": args.mode,
        "sens_min": float(args.sens_min),
        "split": args.split,
        "pred_csv": str(pred_csv),
        "best_epoch": epoch,
        **best_stats,
    }

    print("[BEST]", json.dumps(out, indent=2, ensure_ascii=False))

    if args.save and exp_dir is not None:
        out_path = exp_dir / "selected_threshold.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
