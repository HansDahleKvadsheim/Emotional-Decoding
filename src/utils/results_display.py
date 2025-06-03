import os
import numpy as np
import matplotlib.pyplot as plt

# Define the R² scores for each condition.
data = {
    "None": {
        "train": [0.42279743, 0.33557748, 0.30320004, 0.37365484, 0.3813171, 0.35966801, 0.37727207, 0.37727207],
        "test":  [0.12309655, 0.0471711,  0.04303738, 0.13548178, 0.13548178, 0.09898784, 0.12385877, 0.0891052]
    },
    "Vis": {
        "train": [0.41023698, 0.32798013, 0.29613297, 0.36581233, 0.37433522, 0.3508115, 0.37003397, 0.37787976],
        "test":  [0.11724934, 0.04212982, 0.03766372, 0.12522265, 0.09430655, 0.09449572, 0.11503264, 0.08775338]
    },
    "SomMot": {
        "train": [0.41858746, 0.34681346, 0.3058596,  0.37333828, 0.38059815, 0.36128242, 0.37280214, 0.38738003],
        "test":  [0.12216659, 0.04233451, 0.04390359, 0.12872964, 0.10213079, 0.09798139, 0.12385901, 0.08641728]
    },
    "DorsAttn": {
        "train": [0.42337474, 0.33242964, 0.30584067, 0.37577725, 0.38354072, 0.35701388, 0.38111297, 0.38304698],
        "test":  [0.12763278, 0.04511169, 0.04902118, 0.13448861, 0.11065168, 0.09520331, 0.12912286, 0.09061073]
    },
    "SalVentAttn": {
        "train": [0.41881065, 0.33453322, 0.30556906, 0.37006844, 0.37914357, 0.35853044, 0.3760239,  0.38197459],
        "test":  [0.11950728, 0.04972551, 0.04609953, 0.12669978, 0.10335648, 0.10080888, 0.1148876,  0.09823336]
    },
    "Limbic": {
        "train": [0.42555571, 0.33521568, 0.30610238, 0.37784046, 0.38516354, 0.35927516, 0.37978243, 0.38771746],
        "test":  [0.12207413, 0.04147852, 0.04023278, 0.13086283, 0.10214531, 0.09758749, 0.10658295, 0.08144465]
    },
    "Cont": {
        "train": [0.41424001, 0.33062564, 0.29848958, 0.36863959, 0.37540483, 0.35244098, 0.37514869, 0.3824616],
        "test":  [0.12574668, 0.04409045, 0.04285953, 0.13357216, 0.10323521, 0.09757525, 0.12245199, 0.09543805]
    },
    "Default": {
        "train": [0.44325309, 0.36239845, 0.32391051, 0.39849096, 0.40427324, 0.38277761, 0.40089045, 0.41143527],
        "test":  [0.12686407, 0.04028249, 0.04632647, 0.13941432, 0.10036255, 0.10594868, 0.12631081, 0.10525305]
    }
}

# Emotions are listed in the same order as in the data.
emotions = ["Anticipation", "Joy", "Trust", "Fear", "Surprise", "Sadness", "Anger", "Disgust"]

# Output directory.
output_dir = "alice_dataset/results"
os.makedirs(output_dir, exist_ok=True)

# The full brain (baseline) results.
baseline_train = data["None"]["train"]
baseline_test = data["None"]["test"]

# Loop over each excluded network (skip "None" since that is the full brain).
for network, results in data.items():
    if network == "None":
        continue

    ex_train = results["train"]
    ex_test = results["test"]

    # Create a figure with two subplots: one for training and one for testing.
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)
    index = np.arange(len(emotions))
    bar_width = 0.35

    # ---------------------------
    # Training R² Comparison Plot
    # ---------------------------
    bars_full_train = ax1.bar(index - bar_width/2, baseline_train, bar_width,
                              label="Full Brain", color="dodgerblue")
    bars_ex_train = ax1.bar(index + bar_width/2, ex_train, bar_width,
                            label=f"Excluded: {network}", color="orange")
    ax1.set_title("Training R² Comparison")
    ax1.set_xlabel("Emotions")
    ax1.set_xticks(index)
    ax1.set_xticklabels(emotions, rotation=45, ha="right")
    ax1.set_ylabel("R²")
    ax1.legend()

    # Annotate each bar in the training plot.
    for bar in bars_full_train + bars_ex_train:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    # ---------------------------
    # Testing R² Comparison Plot
    # ---------------------------
    bars_full_test = ax2.bar(index - bar_width/2, baseline_test, bar_width,
                             label="Full Brain", color="dodgerblue")
    bars_ex_test = ax2.bar(index + bar_width/2, ex_test, bar_width,
                           label=f"Excluded: {network}", color="orange")
    ax2.set_title("Testing R² Comparison")
    ax2.set_xlabel("Emotions")
    ax2.set_xticks(index)
    ax2.set_xticklabels(emotions, rotation=45, ha="right")
    ax2.legend()

    # Annotate each bar in the testing plot.
    for bar in bars_full_test + bars_ex_test:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    # Set an overall title for the figure.
    fig.suptitle(f"Comparison of Full Brain vs Excluded: {network}", fontsize=16)

    # Adjust layout so that titles and labels are not cut off.
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure.
    output_file = os.path.join(output_dir, f"{network}_comparison.png")
    plt.savefig(output_file)
    plt.close(fig)

print("Comparison plots saved in", output_dir)
