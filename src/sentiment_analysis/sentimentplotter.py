import matplotlib.pyplot as plt
import re
import numpy as np
import os

filename = 'llm-sentiment-analysis/gpt-4/sentiment.txt'

output_dir = 'plots'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = {
    'Anticipation': [],
    'Joy': [],
    'Trust': [],
    'Fear': [],
    'Surprise': [],
    'Sadness': [],
    'Disgust': [],
    'Anger': []
}

colors = {
    'Anticipation': 'blue',
    'Joy': 'green',
    'Trust': 'orange',
    'Fear': 'red',
    'Surprise': 'purple',
    'Sadness': 'brown',
    'Disgust': 'pink',
    'Anger': 'black'
}


pattern = re.compile(r'(\w+):\s*([\d.]+)')

with open(filename, 'r') as file:
    for line in file:
        matches = pattern.findall(line)
        if matches:
            for emotion, value in matches:
                data[emotion].append(float(value))


for emotion, values in data.items():
    plt.figure(figsize=(8, 4))
    plt.plot(values, color=colors[emotion], label=emotion)  
    plt.xlabel('Line Number')
    plt.ylabel('Value')
    plt.title(f'{emotion} Values over Lines')
    plt.ylim(0, 1)  
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f'{emotion}_plot.png')
    plt.savefig(plot_filename)
    plt.show()
    plt.close() 

