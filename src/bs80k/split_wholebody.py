from pathlib import Path
import pandas as pd
import numpy as np

def stratified_split(df, seed=42, train_ratio=0.8, val_ratio=0.1):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for y in sorted(df["label"].unique()):
        idx = df.index[df["label"] == y].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]

def main():
    index_csv = Path("outputs/bs80k_index.csv")
    assert index_csv.exists(), f"Missing {index_csv}, run parse_index.py first."

    df = pd.read_csv(index_csv)

    # 只切 wholeBody（你当前主线：弱分类/检测都在全身图）
    df = df[df["folder"].isin(["wholeBodyANT", "wholeBodyPOST"])].copy()

    # 去重（按路径去重）
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    out_dir = Path("outputs/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = stratified_split(df, seed=42, train_ratio=0.8, val_ratio=0.1)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("[OK] Saved splits to outputs/splits/")
    print(" train:", len(train_df), " val:", len(val_df), " test:", len(test_df))
    print(" train label ratio:\n", train_df["label"].value_counts(normalize=True))

if __name__ == "__main__":
    main()
