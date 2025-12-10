import json
from pathlib import Path

import pandas as pd

# Paths
SOURCE_CSV = Path(r"..\data\merged_East_labview_egauge_15min.csv")
SELECTION_JSON = Path(r"..\data\quality_analysis\manual_column_selection.json")
OUTPUT_CSV = SOURCE_CSV.with_name(f"final_essential_{SOURCE_CSV.name}")


def load_essential_headers(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        selections = json.load(f)

    essentials = selections["essential"]
    if len(essentials) == 0:
        raise ValueError("Essential header list is empty.")

    return essentials


def build_essential_csv():
    essentials = load_essential_headers(SELECTION_JSON)

    df = pd.read_csv(SOURCE_CSV, parse_dates=['timestamp'], index_col='timestamp')
    missing = [col for col in essentials if col not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns in source CSV: {missing}")

    essential_df = df[essentials]
    essential_df.to_csv(OUTPUT_CSV, index=True)  # timestamp as first column

    print(f"Saved essential columns to {OUTPUT_CSV}")
    print(f"Rows: {len(essential_df)}, Columns: {len(essentials)}")


if __name__ == "__main__":
    build_essential_csv()
