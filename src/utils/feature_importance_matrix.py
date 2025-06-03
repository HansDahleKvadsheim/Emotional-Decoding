import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = "src/explanations/output"
EMOTION = "joy"
MATRIX_SIZE = 400
VERBOSE = True

def create_full_matrix_from_importance(importances, size=400):
    """
    Build a symmetric matrix from the upper triangular values of a feature importance vector.
    """
    full_matrix = np.zeros((size, size))
    full_matrix[np.triu_indices(size, k=1)] = importances
    full_matrix += np.triu(full_matrix, 1).T
    return full_matrix

def plot_full_heatmap(matrix, emotion, output_dir, show_plot=True):
    """
    Plot and save a heatmap of the full (unthresholded) connectivity/importance matrix.
    """
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap='plasma', interpolation='nearest')
    plt.colorbar(im, label='Feature Importance')
    plt.title(f"Full Feature Importance Matrix for '{emotion}'")
    plt.xlabel("ROI")
    plt.ylabel("ROI")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{emotion}_heatmap_full.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print("Saved heatmap to:", filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    # Load feature importance vector
    feature_importances = pd.read_csv('src/explanations/importance.csv')
    importance_vector = feature_importances[EMOTION]

    if VERBOSE:
        print(f"Loaded '{EMOTION}' importance vector with {len(importance_vector)} entries.")

    # Create full symmetric matrix
    full_matrix = create_full_matrix_from_importance(importance_vector.to_numpy(), size=MATRIX_SIZE)

    # Plot and save
    plot_full_heatmap(full_matrix, EMOTION, OUTPUT_DIR, show_plot=True)

if __name__ == "__main__":
    main()
