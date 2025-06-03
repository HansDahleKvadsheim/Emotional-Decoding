import pandas as pd

# === CONFIGURATION ===
RF_IMPORTANCE_PATH = "src/explanations/DFC/RF/importance.csv"  # Update this path
OUTPUT_FORMATTED_PATH = "src/explanations/DFC/RF/importance2.csv"  # Update this path if needed
EMOTIONS = ['anticipation', 'joy', 'trust', 'fear', 'suprise', 'sadness', 'disgust', 'anger']
N_EDGES = 79800  # Number of unique edges for a 400x400 symmetric matrix without duplicates

# === STEP 1: Load the RF_importance CSV ===
rf_df = pd.read_csv(RF_IMPORTANCE_PATH)

rf_df = rf_df.T

rf_df.to_csv(OUTPUT_FORMATTED_PATH, index = False)


rf_df = pd.read_csv(OUTPUT_FORMATTED_PATH)

rf_df.to_csv(OUTPUT_FORMATTED_PATH, index = True)