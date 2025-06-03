import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

MODEL = "lasso"  # "SVR", "linear", "RF", "lasso" or "ridge"
OUTPUT_DIR = f'src/explanations/DFC/{MODEL}/output/heatmaps/ROI_heatmaps'
file_path = f"src/explanations/DFC/{MODEL}/intermediate_results/metrics/top_rois/top_5_rois_by_metric_7N.csv"  # Adjust path as needed

df = pd.read_csv(file_path)
df['ROI_Label'] = df['ROI_Label'].str.replace(r'^7Networks[_]*', '', regex=True)

# Filter data for each metric
degree_df = df[df['Metric'] == 'Weighted Degree'].pivot(index='ROI_Label', columns='Emotion', values='Value')
betweenness_df = df[df['Metric'] == 'Betweenness Centrality'].pivot(index='ROI_Label', columns='Emotion', values='Value')

# Sort ROIs by descending average value
degree_df_sorted = degree_df.sort_index()
betweenness_df_sorted = betweenness_df.sort_index()

# Heatmap for Weighted Degree
plt.figure(figsize=(12, 8))
sns.heatmap(degree_df_sorted, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Weighted Degree'})
plt.title("Top Weighted Degree per Emotion")
plt.ylabel("ROI Label")
plt.xlabel("Emotion")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, f"WR_heatmap_7N_5.png"), dpi=300)



# Heatmap for Betweenness Centrality
plt.figure(figsize=(12, 8))
sns.heatmap(betweenness_df_sorted, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Betweenness Centrality'})
plt.title("Top Betweenness Centrality per Emotion")
plt.ylabel("ROI Label")
plt.xlabel("Emotion")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, f"B_heatmap_7N_5.png"), dpi=300)