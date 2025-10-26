
# AI-Powered Gait Irregularity Detection (PhysioNet-Style, Pipeline-First)

**Author:** Eden  
**Tech:** Python, scikit-learn, Pipelines, matplotlib

## Overview
This project detects **normal vs. impaired** gait using **force/pressure time-series**.
It bridges **kinesiology** and **data science**, demonstrating how biomechanical principles can inform machine learning models for healthcare and prosthetic research.
**Key Highlights**:
- **Biomechanics-inspired features**: peaks, variability, symmetry
- **Leakage-safe ML** via scikit-learn **Pipelines**
- **Figures and results** saved under `reports/`

## Data (Two Options)
### Option A — Real PhysioNet-Style Files (https://physionet.org/content/gaitpdb/1.0.0/)
Create a local `data/` folder with:
- `labels.csv` — two columns: `subject_id,label` where `label` is `0=healthy`, `1=impaired`
- For each subject, two files:
  - `SUBJECT_left.csv`
  - `SUBJECT_right.csv`
Each file should contain **101 rows** (0–100% gait cycle), either a single column or comma-separated values.

Example `labels.csv`:
```
subject_id,label
GaPt03,1
SiCo21,0
GaPt06,1
```

### Option B — Synthetic Fallback (Default)
If `data/` is empty, the notebook **auto-generates** a realistic dataset so the full pipeline runs end-to-end (useful for demos and reproducibility).

## How to Run
1. Open `gait_irregularity_detection.ipynb` in Jupyter or Colab.
2. (Optional) Add real data to `data/` as described above.
3. Run all cells.  
4. Check `reports/` for saved figures:
   - `confusion_matrix.png`
   - `avg_healthy.png`
   - `avg_impaired.png`
   - `feature_importance.png`

## Methods
- **Features**: Mean, Std, Area (sum), Max/Min per foot; **symmetry indices** |mean_L-mean_R|, |area_L-area_R|, |max_L-max_R|, |min_L-min_R|
- **Model**: `Pipeline([StandardScaler(), RandomForestClassifier()])`
- **Validation**: Train/Test split + **StratifiedKFold (5-fold)** cross-validation

## Extensions
- Replace hand-crafted features with **1D CNN** on raw waveforms.
- Multi-class classification for specific conditions.
- **Streamlit** dashboard for interactive visualization.

---

**Files created by this project**  
- `gait_detection.ipynb` — Full runnable notebook  
- `reports/` — Saved figures and a visual report  
