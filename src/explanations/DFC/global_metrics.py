import os
import pandas as pd
import networkx as nx

# CONFIGS
USE_17_NETWORKS = False
MODEL = "linear"  # "SVR", "linear", "RF", "lasso" or "ridge"
OCC_PERCENTILE = 0
FEATURE_PERCENTILE = 0
EMOTIONS = ['suprise', 'sadness', 'disgust', 'anger', 'fear', 'anticipation', 'joy', 'trust']
VERBOSE = True

NETWORK_VERSION = '17N' if USE_17_NETWORKS else '7N'
MST_BASE = f'src/explanations/DFC/{MODEL}/intermediate_results/MSTs'
OUTPUT_DIR = f'src/explanations/DFC/{MODEL}/output/metrics/global_metrics'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f'global_metrics_{NETWORK_VERSION}_occ{OCC_PERCENTILE}_feat{FEATURE_PERCENTILE}.csv')


def compute_global_metrics(graph: nx.Graph):
    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
        efficiency = nx.global_efficiency(graph)
    else:
        diameter = float('nan')
        efficiency = float('nan')

    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight='weight'))
    modularity = nx.algorithms.community.modularity(graph, communities, weight='weight')

    return {
        "Modularity": modularity,
        "GlobalEfficiency": efficiency,
        "TreeDiameter": diameter
    }


def process_all_emotions():
    results = []

    for emotion in EMOTIONS:
        edge_path = os.path.join(MST_BASE, NETWORK_VERSION, emotion,
                                 f"occ{OCC_PERCENTILE}_feat{FEATURE_PERCENTILE}", "edge_list.csv")

        if not os.path.exists(edge_path):
            print(f"Missing MST file for emotion: {emotion}")
            continue

        df_edges = pd.read_csv(edge_path, header=None, names=['source', 'target', 'weight'])
        G = nx.Graph()
        G.add_weighted_edges_from(df_edges.values)

        metrics = compute_global_metrics(G)
        metrics['Emotion'] = emotion.capitalize()
        results.append(metrics)

        if VERBOSE:
            print(f"Processed global metrics for {emotion}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)

    if VERBOSE:
        print(f"\nGlobal metrics saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    process_all_emotions()