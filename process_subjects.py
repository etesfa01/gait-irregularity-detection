import pandas as pd
import numpy as np
import os
from glob import glob
from scipy.signal import resample

# Define input and output directories
DATA_DIR = "raw_physionet"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_subject(file_path):
    # Example filename: GaCo03_01.txt → subject_id = "GaCo03"
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    subject_id = base_name.split("_")[0]  # remove "_01", "_02", etc.

    # Read the file
    df = pd.read_csv(file_path, sep="\t", header=None)
    df = df.dropna(axis=1, how='all')  # remove empty columns if any

    # Extract total VGRF (columns 18 and 19 in 1-based indexing → 17 and 18 in 0-based)
    left = df.iloc[:, 17]   # total left foot force
    right = df.iloc[:, 18]  # total right foot force

    # Resample each to 101 points (1 full gait cycle) ---
    left_resampled = resample(left, 101)
    right_resampled = resample(right, 101)

    # Normalize each foot’s force to its own max (0–1 per subject)
    left_norm = left_resampled / np.max(left_resampled)
    right_norm = right_resampled / np.max(right_resampled)

    # Save to CSV for later ML pipeline
    left_path = os.path.join(OUTPUT_DIR, f"{subject_id}_left.csv")
    right_path = os.path.join(OUTPUT_DIR, f"{subject_id}_right.csv")
    pd.Series(left_norm).to_csv(left_path, index=False, header=False)
    pd.Series(right_norm).to_csv(right_path, index=False, header=False)

    return subject_id


# Process only normal walking trials (_01.txt)
files = [f for f in glob(os.path.join(DATA_DIR, "*.txt")) if "_01.txt" in f]

# Run the processing
ids = [process_subject(f) for f in files]

# Save a log of processed subjects
pd.DataFrame({"subject_id": ids}).to_csv("processed_subjects.csv", index=False)

print(f"Resampled and processed {len(ids)} subjects into 101-point left/right CSV files.")
print(f"Output saved to '{OUTPUT_DIR}/' and log to 'processed_subjects.csv'")
