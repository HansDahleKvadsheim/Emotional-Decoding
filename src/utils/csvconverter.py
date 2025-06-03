import os
import numpy as np
import scipy.io as sio
import pandas as pd

def mat_to_csv(folder_path="Alice/Results/ROIsignal"):
    """
    Reads each .mat file in `folder_path`, takes the first non-internal variable,
    transposes it, and writes a CSV with columns:
      (unnamed index), ROISingal_0, ROISingal_1, ...
    (no 'img_timestamp' column at all).
    """
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".mat"):
            continue

        mat_path = os.path.join(folder_path, filename)
        print(f"Processing {mat_path}...")

        # Load the .mat file into a dictionary
        mat_contents = sio.loadmat(mat_path)

        # Identify the first non-internal key
        user_keys = [k for k in mat_contents.keys() if not k.startswith("__")]
        if not user_keys:
            print(f"  No user variables in {filename}. Skipped.")
            continue

        first_key = user_keys[0]
        data = mat_contents[first_key]

        data = np.atleast_2d(data)

        # Transpose so rows = time/samples, columns = ROI signals
        data = data.T  

        # Build ROI column names
        M = data.shape[1]
        roi_cols = [f"ROISignal_{i}" for i in range(M)]

        # Create DataFrame with those columns only
        df = pd.DataFrame(data, columns=roi_cols)

        # Construct output CSV filename
        csv_filename = os.path.splitext(filename)[0] + "_converted.csv"
        csv_path = os.path.join(folder_path, csv_filename)

        # Write CSV:
        #   - index=True => row index is the first unnamed column
        #   - index_label="" => forces an empty label => leading comma in the header
        df.to_csv(
            csv_path,
            index=True,
            index_label="",
            float_format="%.6f"
        )

        print(f"  -> Saved: {csv_path}")
if __name__ == "__main__":
    mat_to_csv()