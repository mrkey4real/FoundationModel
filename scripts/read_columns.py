#!/usr/bin/env python
"""Read and print column names from the merged CSV file."""
import pandas as pd
from pathlib import Path

data_path = Path(r"E:\MOIRAI\data\final_essential_merged_East_labview_egauge_1min.csv")
df = pd.read_csv(data_path, nrows=5)
print("Total columns:", len(df.columns))
print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")
