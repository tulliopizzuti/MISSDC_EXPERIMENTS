from pathlib import Path
import pandas as pd

SORT_COLUMNS = ["dataset", "missingness", "ratio", "repetition"]

for csv_file in Path(".").glob("*.csv"):
    print(f"Processing {csv_file.name}...")

    df = pd.read_csv(csv_file)

    missing = [c for c in SORT_COLUMNS if c not in df.columns]
    if missing:
        print(f"  Skipped: missing columns {missing}")
        continue

    df = df.sort_values(
        by=SORT_COLUMNS,
        ascending=True,
        kind="mergesort"
    )

    df.to_csv(csv_file, index=False)

print("Done.")