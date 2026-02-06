import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.letterbox import letterbox_pil  # 只用 letterbox，normalize 交给 transforms.Normalize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

import timm  # for ResNet-RS / ConvNeXtV2 and hf_hub weights

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support


class CSVDataset(Dataset):
    """
    Reads outputs/splits/*.csv (created from your bs80k_index).
    Filters to wholeBodyANT/wholeBodyPOST. Then optionally filters view.
    Returns (image_tensor, label_float, image_path_str).
    Uses LETTERBOX to keep aspect ratio (strategy B).
    """
    def __init__(self, csv_path: str, view_filter: str, img_h: int, img_w: int, after_letterbox_tf=None):
        df = pd.read_csv(csv_path)

        # Keep only wholeBody
        df = df[df["folder"].isin(["wholeBodyANT", "wholeBodyPOST"])].copy()

        # Filter view
        view_filter = view_filter.upper()
        if view_filter in ("ANT", "POST"):
            df = df[df["folder"] == f"wholeBody{view_filter}"].copy()
        elif view_filter == "BOTH":
            pass
        else:
            raise ValueError("view must be ANT/POST/BOTH")

        # Drop missing files
        df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
        missing = int((~df["exists"]).sum())
        if missing > 0:
            print(f"[WARN] {missing} files missing in {csv_path}, dropping them.")
        df = df[df["exists"]].copy()

        self.df = df.reset_index(drop=True)
        self.img_h = img_h
        self.img_w = img_w
        self.after_letterbox_tf = after_letterbox_tf  # applied on PIL after letterbox

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        y = int(row["label"])

        img = Image.open(img_path).convert("RGB")

        # ---- STRATEGY B: letterbox (keep aspect ratio) ----
        # NOTE: letterbox_pil expects out_w, out_h
        img_pad, _, _, _ = letterbox_pil(img, out_w=self.img_w, out_h=self.img_h)

        # augment + to tensor + normalize
        if self.after_letterbox_tf is not None:
            x = self.after_letterbox_tf(img_pad)
        else:
            # default: just tensor + normalize
            x = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])(img_pad)

        return x, torch.tensor(y, dtype=torch.float32), str(img_path)


def build_model(name: str) -> nn.Module:
    """
    Supports:
    - torchvision: resnet50, convnext_tiny
    - timm arch name: resnetrs50, convnextv2_tiny, ...
    - timm hf_hub weight id: hf_hub:timm/resnetrs50.tf_in1k
                            hf_hub:timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384
    """
    name = name.strip()

    # ---- Torchvision baselines ----
    if name.lower() == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m

    if name.lower() == "convnext_tiny":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        return m

    # ---- timm models ----
    try:
        m = timm.create_model(name, pretrained=True, num_classes=1)
        return m
    except Exception as e:
        raise ValueError(
            f"Unknown/failed model name: {name}\n"
            f"Try one of:\n"
            f"  resnet50\n"
            f"  convnext_tiny\n"
            f"  hf_hub:timm/resnetrs50.tf_in1k\n"
            f"  hf_hub:timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384\n"
            f"Original error: {repr(e)}"
        )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    paths = []

    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logit = model(x).squeeze(1)
        prob = torch.sigmoid(logit)

        ys.append(y.cpu().numpy())
        ps.append(prob.cpu().numpy())
        paths.extend(list(p))

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    # AUC
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    # Metrics at threshold 0.5 (just for logging; final threshold chosen later)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true.astype(int), y_pred, average="binary", zero_division=0
    )
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))

    metrics = {
        "auc": float(auc),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "sens": float(sens),
        "spec": float(spec),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n": int(tp + tn + fp + fn),
    }

    pred_df = pd.DataFrame({"image_path": paths, "y_true": y_true, "y_prob": y_prob})
    return metrics, pred_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="outputs/splits/train.csv")
    ap.add_argument("--val_csv", default="outputs/splits/val.csv")
    ap.add_argument("--model", default="resnet50",
                    help="resnet50 | convnext_tiny | hf_hub:timm/resnetrs50.tf_in1k | hf_hub:timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384")
    ap.add_argument("--view", default="ANT", choices=["ANT", "POST", "BOTH"])
    ap.add_argument("--img_h", type=int, default=512)
    ap.add_argument("--img_w", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    safe_model_name = args.model.replace("/", "_").replace(":", "_")
    exp_name = args.outdir if args.outdir else f"{safe_model_name}_{args.view}_h{args.img_h}_w{args.img_w}_bs{args.bs}_letterbox"
    outdir = Path("outputs/cls") / exp_name
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- STRATEGY B transforms: apply AFTER letterbox ----
    # train aug on padded PIL (no Resize!)
    train_after_letterbox = transforms.Compose([
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02))],
            p=0.7
        ),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.10, contrast=0.10)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    # val/test: deterministic
    val_after_letterbox = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_ds = CSVDataset(args.train_csv, view_filter=args.view, img_h=args.img_h, img_w=args.img_w,
                          after_letterbox_tf=train_after_letterbox)
    val_ds = CSVDataset(args.val_csv, view_filter=args.view, img_h=args.img_h, img_w=args.img_w,
                        after_letterbox_tf=val_after_letterbox)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model).to(device)

    # class balance: pos_weight = Nneg/Npos
    y_train = train_ds.df["label"].values.astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_auc = -1.0

    # save config
    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"[INFO] device={device}")
    print(f"[INFO] train={len(train_ds)} val={len(val_ds)} pos={n_pos} neg={n_neg} pos_weight={pos_weight.item():.3f}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] outdir={outdir}")
    print(f"[INFO] preprocess=LETTERBOX (keep aspect ratio) -> ToTensor -> Normalize")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logit = model(x).squeeze(1)
                loss = criterion(logit, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")

        val_metrics, val_pred = evaluate(model, val_loader, device)
        val_pred.to_csv(outdir / f"val_preds_epoch{epoch:02d}.csv", index=False)

        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} "
              f"val_auc={val_metrics['auc']:.4f} sens={val_metrics['sens']:.3f} "
              f"spec={val_metrics['spec']:.3f} f1={val_metrics['f1']:.3f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "model_name": args.model},
                outdir / "best.pt"
            )
            with (outdir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(val_metrics, f, ensure_ascii=False, indent=2)

    print(f"[DONE] best_auc={best_auc:.4f} saved to {outdir/'best.pt'}")


if __name__ == "__main__":
    main()
