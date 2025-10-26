import pandas as pd

labels = pd.read_csv("data/labels.csv")
processed = pd.read_csv("processed_subjects.csv")

labels.columns = labels.columns.str.strip()
processed.columns = processed.columns.str.strip()

# Sets for comparison
labels_set = set(labels["subject_id"].unique())
processed_set = set(processed["subject_id"].unique())

missing_in_processed = sorted(labels_set - processed_set)
extra_in_processed = sorted(processed_set - labels_set)

print(f" Missing in processed ({len(missing_in_processed)}):", missing_in_processed)
print(f" Extra in processed ({len(extra_in_processed)}):", extra_in_processed)
