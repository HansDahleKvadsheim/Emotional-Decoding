import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

output_folder = "data/labels/plots"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv("data/labels/godvectors.csv", index_col=0)

df.columns = [col.capitalize() for col in df.columns]

color_mapping = {
    'Anticipation': 'blue',
    'Joy': 'green',
    'Trust': 'orange',
    'Fear': 'red',
    'Surprise': 'purple',
    'Sadness': 'brown',
    'Disgust': 'pink',
    'Anger': 'black'
}

discrete_values = np.arange(0.1, 1.1, 0.1)

for sentiment in df.columns:
    counts = df[sentiment].value_counts().sort_index()
    counts = counts.reindex(discrete_values, fill_value=0)

    plt.figure(figsize=(8, 6))
    plt.bar(discrete_values, counts, color=color_mapping[sentiment],
            edgecolor='black', width=0.08)
    plt.title(f"{sentiment}")
    plt.xlabel(sentiment)
    plt.ylabel("Frequency")
    
    # Set x-axis ticks to show each discrete value (0.1, 0.2, …, 1.0)
    plt.xticks(discrete_values, [f"{val:.1f}" for val in discrete_values])
    plt.xlim(0, 1)
    
    output_path = os.path.join(output_folder, f"{sentiment}_histogram.png")
    plt.savefig(output_path)
    plt.close()
