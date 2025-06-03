import pandas as pd

# Read the CSV; assuming the first column is an index and each row is an emotion vector
df = pd.read_csv("data/labels/godvectors.csv", index_col=0)

# Compute the rolling average (window of 3 rows) for each column.
# This will compute the mean for rows 0-2, then 1-3, 2-4, etc.
rolling_avg_df = df.rolling(window=3).mean().dropna()

# Optionally, reset the index so that the new CSV has a clean 0,1,2,... index.
rolling_avg_df.reset_index(drop=True, inplace=True)

# Define a fitting name for the new CSV file
new_csv_name = "GodVectors_RollingTripleAverages.csv"
rolling_avg_df.to_csv(new_csv_name, index=True)

print("New CSV file created:", new_csv_name)
