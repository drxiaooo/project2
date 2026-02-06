import json
from pathlib import Path
import pandas as pd

def main():
    root = Path("outputs/cls")
    rows = []
    for exp in root.iterdir():
        if not exp.is_dir():
            continue
        mpath = exp / "best_metrics.json"
        if not mpath.exists():
            continue
        with mpath.open("r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "exp": exp.name,
            "auc": m.get("auc"),
            "sens": m.get("sens"),
            "spec": m.get("spec"),
            "f1": m.get("f1"),
            "acc": m.get("acc"),
            "tp": m.get("tp"),
            "tn": m.get("tn"),
            "fp": m.get("fp"),
            "fn": m.get("fn"),
            "n": m.get("n"),
        })

    df = pd.DataFrame(rows).sort_values(["auc", "sens"], ascending=[False, False])
    out = root / "leaderboard.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n[OK] saved: {out}")

if __name__ == "__main__":
    main()
