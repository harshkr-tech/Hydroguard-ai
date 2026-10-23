

import pandas as pd

df = pd.read_csv(
    "data/raw/streamflow.csv",
    parse_dates=["date"]
).sort_values("date").reset_index(drop=True)

peak_indices = df["discharge_cfs"].nlargest(5).index.tolist()

print("Number of peak rows selected:", len(peak_indices))

for number, idx in enumerate(peak_indices, start=1):
    start = max(0, idx - 3)
    end = min(len(df), idx + 4)

    print(f"\n--- PEAK {number} ---")
    print(
        df.iloc[start:end][["date", "discharge_cfs"]]
        .to_string(index=False)
    )

