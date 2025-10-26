import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# Ensure report folder exists for saving figures
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 120

DATA_DIR = "data"


def load_physionet_style(DATA_DIR: str):
    labels_path = os.path.join(DATA_DIR, "labels.csv")
    if not os.path.exists(labels_path):
        return None

    labels = pd.read_csv(labels_path)
    records = []
    for _, row in labels.iterrows():
        sid = str(row["subject_id"])
        y = int(row["label"])
        left_path = os.path.join(DATA_DIR, f"{sid}_left.csv")
        right_path = os.path.join(DATA_DIR, f"{sid}_right.csv")
        if not (os.path.exists(left_path) and os.path.exists(right_path)):
            continue
        # Load as 1D arrays
        left = pd.read_csv(left_path, header=None).values.flatten()
        right = pd.read_csv(right_path, header=None).values.flatten()
        if len(left) != 101 or len(right) != 101:
            continue
        records.append((sid, y, left, right))
    if len(records) == 0:
        return None
    return records


def synthetic_double_hump(n_points=101, asym=0.0, noise=0.02):
    x = np.linspace(0, 1, n_points)
    # Two gaussian-like humps to mimic GRF vertical pattern
    hump1 = np.exp(-((x - 0.2) ** 2) / 0.01)
    hump2 = np.exp(-((x - 0.6) ** 2) / 0.015)
    base = 0.2 + 0.9 * hump1 + 1.1 * hump2
    # Add asymmetry by scaling second hump
    base = base * (1.0 + asym * (x - 0.5))
    # Add noise
    base += np.random.normal(0, noise, size=n_points)
    # Normalize roughly to body weight units
    base = base / np.max(base)
    return base


def generate_synthetic_dataset(n_subjects=80, impaired_ratio=0.5, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_subjects):
        sid = f"S{i:03d}"
        is_impaired = 1 if rng.random() < impaired_ratio else 0
        # Healthy: low asymmetry/noise; Impaired: higher asymmetry and variability
        if is_impaired:
            asym_L = rng.uniform(0.05, 0.15)
            asym_R = rng.uniform(-0.15, -0.05)
            noise = rng.uniform(0.03, 0.06)
        else:
            asym_L = rng.uniform(-0.02, 0.02)
            asym_R = rng.uniform(-0.02, 0.02)
            noise = rng.uniform(0.01, 0.03)

        left = synthetic_double_hump(asym=asym_L, noise=noise)
        right = synthetic_double_hump(asym=asym_R, noise=noise)
        records.append((sid, is_impaired, left, right))
    return records


records = load_physionet_style(DATA_DIR)
if records is None:
    print("No real data detected in ./data — generating synthetic dataset so the notebook is runnable.")
    records = generate_synthetic_dataset()
else:
    print(f"Loaded {len(records)} subject records from data/.")
len(records)


def extract_features(left: np.ndarray, right: np.ndarray) -> dict:
    feats = {}
    # Left features
    feats["mean_L"] = float(np.mean(left))
    feats["std_L"] = float(np.std(left))
    feats["area_L"] = float(np.sum(left))
    feats["max_L"] = float(np.max(left))
    feats["min_L"] = float(np.min(left))
    # Right features
    feats["mean_R"] = float(np.mean(right))
    feats["std_R"] = float(np.std(right))
    feats["area_R"] = float(np.sum(right))
    feats["max_R"] = float(np.max(right))
    feats["min_R"] = float(np.min(right))
    # Symmetry
    feats["sym_mean"] = abs(feats["mean_L"] - feats["mean_R"])
    feats["sym_area"] = abs(feats["area_L"] - feats["area_R"])
    feats["sym_max"] = abs(feats["max_L"] - feats["max_R"])
    feats["sym_min"] = abs(feats["min_L"] - feats["min_R"])
    return feats


rows = []
for sid, label, left, right in records:
    f = extract_features(left, right)
    f["subject_id"] = sid
    f["label"] = int(label)
    rows.append(f)

df = pd.DataFrame(rows)
print(df.head())
print("\nClass balance:\n", df["label"].value_counts())

from sklearn.metrics import accuracy_score

X = df.drop(columns=["subject_id", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print("CV Accuracy Mean:", cv_scores.mean())
print("CV Accuracy Std:", cv_scores.std())

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots()
disp.plot(ax=ax)
ax.set_title("Confusion Matrix - Gait Irregularity Detection")
fig.tight_layout()
figpath = os.path.join(REPORT_DIR, "confusion_matrix.png")
fig.savefig(figpath)
print("Saved:", figpath)
plt.show()

# Reconstruct class-wise stacks of waveforms from original records
healthy_L, healthy_R, impaired_L, impaired_R = [], [], [], []

for (sid, label, left, right) in records:
    if label == 0:
        healthy_L.append(left)
        healthy_R.append(right)
    else:
        impaired_L.append(left)
        impaired_R.append(right)

healthy_L = np.array(healthy_L) if len(healthy_L) else np.zeros((1, 101))
healthy_R = np.array(healthy_R) if len(healthy_R) else np.zeros((1, 101))
impaired_L = np.array(impaired_L) if len(impaired_L) else np.zeros((1, 101))
impaired_R = np.array(impaired_R) if len(impaired_R) else np.zeros((1, 101))

x = np.linspace(0, 100, 101)

fig1, ax1 = plt.subplots()
ax1.plot(x, healthy_L.mean(axis=0), label="Healthy Left")
ax1.plot(x, healthy_R.mean(axis=0), label="Healthy Right")
ax1.set_title("Average Healthy Gait (Forces)")
ax1.set_xlabel("Gait Cycle (%)")
ax1.set_ylabel("Normalized Force (BW)")
ax1.legend()
fig1.tight_layout()
p1 = os.path.join(REPORT_DIR, "avg_healthy.png")
fig1.savefig(p1)
print("Saved:", p1)
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(x, impaired_L.mean(axis=0), label="Impaired Left")
ax2.plot(x, impaired_R.mean(axis=0), label="Impaired Right")
ax2.set_title("Average Impaired Gait (Forces)")
ax2.set_xlabel("Gait Cycle (%)")
ax2.set_ylabel("Normalized Force (BW)")
ax2.legend()
fig2.tight_layout()
p2 = os.path.join(REPORT_DIR, "avg_impaired.png")
fig2.savefig(p2)
print("Saved:", p2)
plt.show()

# Access the trained model:
rf = pipe.named_steps["model"]
# Get feature names in the same order we fed to the model
feat_names = list(X.columns)
importances = rf.feature_importances_
order = np.argsort(importances)[::-1]

top_k = 10
fig3, ax3 = plt.subplots()
ax3.bar(range(top_k), importances[order][:top_k])
ax3.set_xticks(range(top_k))
ax3.set_xticklabels([feat_names[i] for i in order[:top_k]], rotation=45, ha="right")
ax3.set_title("Top Feature Importances (Random Forest)")
ax3.set_ylabel("Importance")
fig3.tight_layout()
p3 = os.path.join(REPORT_DIR, "feature_importance.png")
fig3.savefig(p3)
print("Saved:", p3)
plt.show()
