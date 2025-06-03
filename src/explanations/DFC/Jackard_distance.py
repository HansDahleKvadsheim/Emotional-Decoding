import os
import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations

# CONFIGS
USE_17_NETWORKS = False
MODEL = "lasso"  # "SVR", "linear", "RF", "lasso" or "ridge"
MST_BASE = f'src/explanations/DFC/{MODEL}/intermediate_results/MSTs'
NETWORK_VERSION = '17N' if USE_17_NETWORKS else '7N'
EMOTIONS = ['suprise', 'sadness', 'disgust', 'anger', 'fear', 'anticipation', 'joy', 'trust']
OCC_PERCENTILE = 0
FEATURE_PERCENTILE = 0
OUTPUT_PATH = f'src/explanations/DFC/{MODEL}/output/similarity/{NETWORK_VERSION}_jackard_distance_matrix_SVM.csv'

def load_graph(emotion):
    edge_file = os.path.join(MST_BASE, NETWORK_VERSION, emotion, f"occ{OCC_PERCENTILE}_feat{FEATURE_PERCENTILE}", "edge_list.csv")
    if not os.path.exists(edge_file):
        print(f"Missing edge list for: {emotion}")
        return None
    df = pd.read_csv(edge_file)
    G = nx.Graph()
    G.add_weighted_edges_from(df.values)
    return G

def weighted_jaccard(G1, G2):
    edges1 = {frozenset(e[:2]): e[2] for e in G1.edges(data='weight')}
    edges2 = {frozenset(e[:2]): e[2] for e in G2.edges(data='weight')}
    all_edges = set(edges1) | set(edges2)

    intersection = sum(min(edges1.get(e, 0), edges2.get(e, 0)) for e in all_edges)
    union = sum(max(edges1.get(e, 0), edges2.get(e, 0)) for e in all_edges)

    return 1 - intersection / union if union > 0 else 1  # distance = 1 - similarity

def main():
    graphs = {emotion: load_graph(emotion) for emotion in EMOTIONS}
    distance_matrix = pd.DataFrame(index=EMOTIONS, columns=EMOTIONS)

    for e1, e2 in combinations(EMOTIONS, 2):
        G1, G2 = graphs[e1], graphs[e2]
        if G1 is not None and G2 is not None:
            dist = weighted_jaccard(G1, G2)
            distance_matrix.at[e1, e2] = distance_matrix.at[e2, e1] = dist

    np.fill_diagonal(distance_matrix.values.astype(float), 0.0)
    distance_matrix.to_csv(OUTPUT_PATH)
    print(f"Weighted Jaccard distance matrix saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
