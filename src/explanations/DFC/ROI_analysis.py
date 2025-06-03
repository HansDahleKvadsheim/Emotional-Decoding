import os
import pandas as pd
import networkx as nx
from nilearn import datasets

#CONFIGS
USE_17_NETWORKS = False
MODEL = "lasso"  # "SVR", "linear", "RF", "lasso" or "ridge"
TOP_N = 5  # <<== CHANGE THIS to control how many top ROIs to extract
NUM_ROIS = 400 # <<== should always be 400 tbh
EMOTIONS = ['suprise', 'sadness', 'disgust', 'anger', 'fear', 'anticipation', 'joy', 'trust']
OCC_PERCENTILE = 0
FEATURE_PERCENTILE = 0

NETWORK_VERSION = '17N' if USE_17_NETWORKS else '7N'
MST_BASE = f'src/explanations/DFC/{MODEL}/intermediate_results/MSTs'
OUTPUT_DIR = f'src/explanations/DFC/{MODEL}/intermediate_results/metrics/top_rois'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_roi_labels():
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=NUM_ROIS, yeo_networks=17 if USE_17_NETWORKS else 7)
    return [label.decode() if isinstance(label, bytes) else label for label in atlas['labels']]

def analyze_top_rois_named():
    roi_labels = load_roi_labels()
    all_rows = []

    for emotion in EMOTIONS:
        edge_file = os.path.join(
            MST_BASE, NETWORK_VERSION, emotion,
            f"occ{OCC_PERCENTILE}_feat{FEATURE_PERCENTILE}",
            "edge_list.csv"
        )

        if not os.path.exists(edge_file):
            print(f"Missing MST edge list for: {emotion}")
            continue

        df_edges = pd.read_csv(edge_file)
        G = nx.Graph()
        G.add_weighted_edges_from(df_edges.values)

        deg = dict(G.degree(weight='weight'))
        btw = nx.betweenness_centrality(G)

        top_degree = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        top_betweenness = sorted(btw.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

        for rank, (roi_idx, value) in enumerate(top_degree, 1):
            all_rows.append({
                "Emotion": emotion.capitalize(),
                "Metric": "Weighted Degree",
                "Rank": rank,
                "ROI_Index": roi_idx,
                "ROI_Label": roi_labels[int(roi_idx)],
                "Value": value
            })

        for rank, (roi_idx, value) in enumerate(top_betweenness, 1):
            all_rows.append({
                "Emotion": emotion.capitalize(),
                "Metric": "Betweenness Centrality",
                "Rank": rank,
                "ROI_Index": roi_idx,
                "ROI_Label": roi_labels[int(roi_idx)],
                "Value": value
            })

    df_out = pd.DataFrame(all_rows)
    out_path = os.path.join(OUTPUT_DIR, f"top_{TOP_N}_rois_by_metric_{NETWORK_VERSION}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved top {TOP_N} ROI metrics with labels to: {out_path}")

if __name__ == "__main__":
    analyze_top_rois_named()
