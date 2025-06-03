import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.algorithms.community import greedy_modularity_communities
from nilearn import datasets
from PIL import Image
import seaborn as sns

# CONFIGS
USE_17_NETWORKS = False
MODEL = "lasso"   #"linear", "RF", "SVR", "Ridge" or "lasso"
AVG_DFC_CSV = 'src/explanations/DFC/avg_DFC.csv'
IMPORTANCE_CSV = f'src/explanations/DFC/{MODEL}/importance.csv'
MST_BASE = f'src/explanations/DFC/{MODEL}/intermediate_results/MSTs'
METRICS_PLOT_DIR = f'src/explanations/DFC/{MODEL}/intermediate_results/metrics/plots'
OUTPUT_CSV_BASE = f'src/explanations/DFC/{MODEL}/intermediate_results/metrics/network_weighted_summary'
FINAL_OUTPUT_BASE = f'src/explanations/DFC/{MODEL}/output/signatures'
EMOTIONS = ['suprise', 'sadness', 'disgust', 'anger', 'fear', 'anticipation', 'joy', 'trust']
OCC_PERCENTILE = 0
FEATURE_PERCENTILE = 0
SHOW_PLOT = False
VERBOSE = True

NETWORK_VERSION = '17N' if USE_17_NETWORKS else '7N'
OUTPUT_CSV = f"{OUTPUT_CSV_BASE}_{NETWORK_VERSION}.csv"
FINAL_OUTPUT_DIR = os.path.join(FINAL_OUTPUT_BASE, f"{NETWORK_VERSION}_OCC{OCC_PERCENTILE}_FEAT{FEATURE_PERCENTILE}")
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

#NETWORK COLOR MAPPING
NETWORK_COLOR_MAPPING_17 = {
    "VisCent": "red", "VisPeri": "tomato",
    "SomMotA": "blue", "SomMotB": "skyblue",
    "DorsAttnA": "green", "DorsAttnB": "lime",
    "SalVentAttnA": "orange", "SalVentAttnB": "gold",
    "ContA": "brown", "ContB": "sienna", "ContC": "peru",
    "DefaultA": "pink", "DefaultB": "lightpink", "DefaultC": "hotpink",
    "LimbicA": "purple", "LimbicB": "orchid", "TempPar": "gray"
}

NETWORK_COLOR_MAPPING_7 = {
    'Vis': 'red', 'SomMot': 'blue', 'DorsAttn': 'green',
    'SalVentAttn': 'orange', 'Limbic': 'purple',
    'Cont': 'brown', 'Default': 'pink'
}

COLOR_MAPPING = NETWORK_COLOR_MAPPING_17 if USE_17_NETWORKS else NETWORK_COLOR_MAPPING_7

def load_atlas_labels():
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17 if USE_17_NETWORKS else 7)
    return [label.decode() if isinstance(label, bytes) else label for label in atlas['labels']]

def get_color_for_label(label):
    for key, color in COLOR_MAPPING.items():
        if key in label:
            return color
    return 'black'

def generate_graphs(labels, avg_DFC, upper_inds):
    occ_thresh = np.percentile(avg_DFC[upper_inds], OCC_PERCENTILE)
    occ_mask = avg_DFC >= occ_thresh
    if VERBOSE:
        print(f"OCC Threshold ({OCC_PERCENTILE}th percentile): {occ_thresh}")

    for emotion in EMOTIONS:
        imp_df = pd.read_csv(IMPORTANCE_CSV)
        imp = imp_df[emotion].to_numpy()
        full_imp = np.zeros((400, 400)) #generate empty matix
        full_imp[upper_inds] = imp  #fill upper triangle
        full_imp += full_imp.T #mirror
        mat_occ = full_imp * occ_mask #thresholding
        feat_thresh = np.percentile(mat_occ[upper_inds], FEATURE_PERCENTILE)
        mat_filt = np.where(mat_occ >= feat_thresh, mat_occ, 0)
        G = nx.from_numpy_array(mat_filt) #make graph
        tree = nx.maximum_spanning_tree(G) #kruskals algorithm
        tree.remove_nodes_from(list(nx.isolates(tree)))   #remove  isolated nodes. they are just noise
        save_graph_and_plot(tree, labels, emotion)  

def save_graph_and_plot(tree, labels, emotion):
    out_dir = os.path.join(MST_BASE, NETWORK_VERSION, emotion, f"occ{OCC_PERCENTILE}_feat{FEATURE_PERCENTILE}")
    os.makedirs(out_dir, exist_ok=True)
    edge_list_path = os.path.join(out_dir, 'edge_list.csv')
    nx.write_edgelist(tree, edge_list_path, data=['weight'], delimiter=',')

    try:
        pos = graphviz_layout(tree, prog='neato')
    except:
        pos = nx.spring_layout(tree)

    node_colors = [get_color_for_label(labels[n]) for n in tree.nodes()]
    fig, ax = plt.subplots(figsize=(13, 13))
    nx.draw(tree, pos, node_size=50, font_size=8, node_color=node_colors, with_labels=False, ax=ax)
    fig.suptitle(f"{emotion} OCC{OCC_PERCENTILE} FEAT{FEATURE_PERCENTILE}", fontsize=18, y=0.98)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=net, markerfacecolor=color, markersize=10) 
        for net, color in COLOR_MAPPING.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.93),
        ncol=5,
        frameon=False,
        fontsize=10,
        title="Networks"
    )
    fig.subplots_adjust(top=0.88)

    img_path = os.path.join(out_dir, 'spanning_tree.png')
    plt.savefig(img_path, bbox_inches='tight')
    if SHOW_PLOT:
        plt.show()
    plt.close()

    if VERBOSE:
        print(f"Saved tree and image to {out_dir}")

def analyze_graphs(labels):
    roi_to_network = {
        i: label.split('_')[2] if label.startswith(f"{17 if USE_17_NETWORKS else 7}Networks") else label
        for i, label in enumerate(labels)
    }

    all_results = []

    for emotion in EMOTIONS:
        edge_file = os.path.join(MST_BASE, NETWORK_VERSION, emotion, f"occ{OCC_PERCENTILE}_feat{FEATURE_PERCENTILE}", "edge_list.csv")
        if not os.path.exists(edge_file):
            print(f"Missing edge list for: {emotion}")
            continue

        df_edges = pd.read_csv(edge_file)
        G = nx.Graph()
        G.add_weighted_edges_from(df_edges.values)
        weighted_deg = dict(G.degree(weight='weight'))
        btw = nx.betweenness_centrality(G)

        communities = list(greedy_modularity_communities(G, weight='weight'))
        modularity = nx.algorithms.community.modularity(G, communities, weight='weight')
        efficiency = nx.global_efficiency(G)
        diameter = nx.diameter(G) if nx.is_connected(G) else float('nan')

        global_metrics = {
            "Emotion": emotion.capitalize(),
            "Modularity": modularity,
            "GlobalEfficiency": efficiency,
            "TreeDiameter": diameter
        }

        df = pd.DataFrame({
            'ROI': list(G.nodes),
            'weighted_degree': [weighted_deg[n] for n in G.nodes],
            'betweenness': [btw[n] for n in G.nodes],
            'Network': [roi_to_network[n] for n in G.nodes]
        })

        net_summary = df.groupby("Network")[["weighted_degree", "betweenness"]].mean()
        net_summary = net_summary.sort_values(by="weighted_degree", ascending=False)
        net_summary["Emotion"] = emotion.capitalize()
        for metric_name, value in global_metrics.items():
            net_summary[metric_name] = value

        all_results.append(net_summary.reset_index())

    summary_df = pd.concat(all_results, ignore_index=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    if VERBOSE:
        print("\n=== Weighted Network + Global Metrics Summary ===")
        print(summary_df)



def plot_metrics():
    summary_path = f"{OUTPUT_CSV_BASE}_{NETWORK_VERSION}.csv"
    summary_df = pd.read_csv(summary_path)

    metrics_plot_dir = os.path.join(METRICS_PLOT_DIR, NETWORK_VERSION)
    os.makedirs(metrics_plot_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    emotions = summary_df["Emotion"].unique()

    for emotion in emotions:
        df_emotion = summary_df[summary_df["Emotion"] == emotion]

        modularity = df_emotion.iloc[0]["Modularity"]
        efficiency = df_emotion.iloc[0]["GlobalEfficiency"]
        diameter = df_emotion.iloc[0]["TreeDiameter"]
        diameter_display = "NaN" if pd.isna(diameter) else int(diameter)
        text_str = f"Modularity: {modularity:.3f}    |    Global Efficiency: {efficiency:.3f}    |    Tree Diameter: {diameter_display}"

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 14),
                                 gridspec_kw={'height_ratios': [4, 4, 1]})
        fig.suptitle(f"Graph Metrics by Network - {emotion}", fontsize=18, y=0.94)

        # Weighted Degree Plot
        sns.barplot(
            data=df_emotion.sort_values("weighted_degree", ascending=False),
            x="weighted_degree",
            y="Network",
            ax=axes[0],
            palette="Reds_d"
        )
        axes[0].set_title("Average Weighted Degree", fontsize=14)
        axes[0].set_xlabel("Weighted Degree")
        axes[0].set_ylabel("Brain Network")

        # Betweenness Centrality Plot
        sns.barplot(
            data=df_emotion.sort_values("betweenness", ascending=False),
            x="betweenness",
            y="Network",
            ax=axes[1],
            palette="Blues_d"
        )
        axes[1].set_title("Average Betweenness Centrality", fontsize=14)
        axes[1].set_xlabel("Betweenness")
        axes[1].set_ylabel("Brain Network")

        # Global Metrics Text
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, text_str, fontsize=14, ha='center', va='center', weight='bold')

        plt.tight_layout(rect=[0, 0.05, 1, 0.93])

        out_path = os.path.join(metrics_plot_dir, f"{emotion}_graph_metrics.png")
        plt.savefig(out_path)
        plt.close()

        if VERBOSE:
            print(f"Saved metrics plot to: {out_path}")


#combine tree and metrics into one picture
def merge_images_vertically(image_path_top, image_path_bottom, output_path):
    img_top = Image.open(image_path_top)
    img_bottom = Image.open(image_path_bottom)
    max_width = max(img_top.width, img_bottom.width)
    total_height = img_top.height + img_bottom.height

    merged_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
    merged_img.paste(img_top, (0, 0))
    merged_img.paste(img_bottom, (0, img_top.height))
    merged_img.save(output_path)
    if VERBOSE:
        print(f"Merged image saved to: {output_path}")

def merge_all_images():
    metrics_plot_dir = os.path.join(METRICS_PLOT_DIR, NETWORK_VERSION)

    for emotion in EMOTIONS:
        tree_img_path = os.path.join(MST_BASE, NETWORK_VERSION, emotion, f"occ{OCC_PERCENTILE}_feat{FEATURE_PERCENTILE}", 'spanning_tree.png')
        metrics_plot_path = os.path.join(metrics_plot_dir, f"{emotion}_graph_metrics.png")
        output_path = os.path.join(FINAL_OUTPUT_DIR, f"{emotion}_final_output.png")

        if os.path.exists(tree_img_path) and os.path.exists(metrics_plot_path):
            merge_images_vertically(tree_img_path, metrics_plot_path, output_path)
        else:
            print(f"Missing images for emotion: {emotion}")


def main():
    labels = load_atlas_labels()
    avg_DFC = pd.read_csv(AVG_DFC_CSV).values
    upper_inds = np.triu_indices_from(avg_DFC, k=1)

    if VERBOSE:
        print(f"Loaded avg_DFC shape: {avg_DFC.shape}")

    generate_graphs(labels, avg_DFC, upper_inds)
    analyze_graphs(labels)
    plot_metrics()
    merge_all_images()


if __name__ == "__main__":
    main()
