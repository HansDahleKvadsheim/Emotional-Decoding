import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# === GLOBAL SETTINGS ===
OUTPUT_FORMAT = "csv"     # "png" or "csv"
SCALE = True              # True = apply min-max scaling
MODEL = "lasso"           # "SVM", "ridge", "linear", "lasso" or "RF"
NETWORK_TYPE = "7N"      #"7N" or "17N"

INPUT_CSV = f'src/explanations/DFC/{MODEL}/intermediate_results/metrics/network_weighted_summary_{NETWORK_TYPE}.csv'
OUTPUT_BASE = f'src/explanations/DFC/{MODEL}/output/heatmaps/networks/{NETWORK_TYPE}/{"scaled" if SCALE else "raw"}'
os.makedirs(OUTPUT_BASE, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
networks_sorted = sorted(df['Network'].unique())

degree_df = df.pivot(index='Network', columns='Emotion', values='weighted_degree').reindex(index=networks_sorted)
betweenness_df = df.pivot(index='Network', columns='Emotion', values='betweenness').reindex(index=networks_sorted)

# Apply min-max scaling if specified
if SCALE:
    degree_df = pd.DataFrame(MinMaxScaler().fit_transform(degree_df),
                             index=degree_df.index, columns=degree_df.columns)
    betweenness_df = pd.DataFrame(MinMaxScaler().fit_transform(betweenness_df),
                                  index=betweenness_df.index, columns=betweenness_df.columns)


if OUTPUT_FORMAT == "png":
    # Plot weighted degree
    plt.figure(figsize=(12, 8))
    sns.heatmap(degree_df, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={'label': 'Scaled Weighted Degree' if SCALE else 'Weighted Degree'})
    plt.title(f'{"Min-Max Scaled " if SCALE else ""}Weighted Degree per Emotion (Network-Level)')
    plt.ylabel("Network")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE, "weighted_degree_heatmap.png"), dpi=300)
    plt.close()

    # Plot betweenness centrality
    plt.figure(figsize=(12, 8))
    sns.heatmap(betweenness_df, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={'label': 'Scaled Betweenness Centrality' if SCALE else 'Betweenness Centrality'})
    plt.title(f'{"Min-Max Scaled " if SCALE else ""}Betweenness Centrality per Emotion (Network-Level)')
    plt.ylabel("Network")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE, "betweenness_centrality_heatmap.png"), dpi=300)
    plt.close()

    print(f"PNG heatmaps saved to: {OUTPUT_BASE}")

elif OUTPUT_FORMAT == "csv":
    degree_df.to_csv(os.path.join(OUTPUT_BASE, "weighted_degree.csv"))
    betweenness_df.to_csv(os.path.join(OUTPUT_BASE, "betweenness_centrality.csv"))
    print(f"Scaled tables saved as CSV to: {OUTPUT_BASE}")

else:
    print("Invalid OUTPUT_FORMAT specified. Use 'png' or 'csv'.")
