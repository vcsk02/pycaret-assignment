# PyCaret “Low Code” Data Mining Assignment

This repository contains a Google Colab notebook that demonstrates end-to-end data mining with the **PyCaret** library across supervised learning, clustering, anomaly detection, and time-series forecasting. All examples use the **Fitness & Activity** dataset.

> **Primary file:** `pycaret_assignment.ipynb` (complete, executable workflow)  
> **Extras:** saved plots/outputs and two small Gradio apps for interactive demos.

---

## Table of Contents

- [Dataset](#dataset)
- [Quick Start (Colab)](#quick-start-colab)
- [Environment Notes & Troubleshooting](#environment-notes--troubleshooting)
- [Tasks Overview](#tasks-overview)
  - [Task A — Supervised Learning](#task-a--supervised-learning)
  - [Task B — Unsupervised Learning: Clustering](#task-b--unsupervised-learning-clustering)
  - [Task C — Unsupervised Learning: Anomaly Detection](#task-c--unsupervised-learning-anomaly-detection)
  - [Task D — Association Rules (Skipped)](#task-d--association-rules-skipped)
  - [Task E — Time Series Forecasting](#task-e--time-series-forecasting)
- [Repository Structure](#repository-structure)
- [Reproducibility Tips](#reproducibility-tips)
- [License](#license)

---

## Dataset

- **Source:** Fitness & Activity Dataset on Kaggle  
- **Link:** _Add the Kaggle URL here_  
- **Typical fields:** `steps`, `duration_min`, `heart_rate_avg`, `calories_burned`, `activity_type`, timestamps, and derived features.

---

## Quick Start (Colab)

1. **Open the notebook**  
   Upload or open `pycaret_assignment.ipynb` in Google Colab.

2. **STEP 1 — Install Python 3.11 and PyCaret 3.3.2**  
   Run the “STEP 1” cell in the notebook (installs Python 3.11 and `pycaret[full]==3.3.2`), then **restart the runtime**.
   ```bash
   # (Shown in the notebook)
   !apt-get update -y
   !apt-get install -y python3.11 python3.11-venv python3.11-dev
   !update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
   !python3 -m pip install --upgrade pip
   !python3 -m pip install "pycaret[full]==3.3.2" gradio
   ```

3. **STEP 2 — Validate Environment**  
   Confirm versions before running tasks:
   ```python
   import sys, pycaret
   print(sys.version)
   print(pycaret.__version__)
   ```

4. **Run the tasks (A–E)**  
   Execute the notebook cells in order. Saved plots and artifacts are written to the repository folders as configured in the notebook.

5. **Launch Gradio apps**  
   The final section starts two small web apps for the regression and multiclass models.

---

## Environment Notes & Troubleshooting

- **Colab default (Python 3.12) is incompatible** with `pycaret==3.3.2`. The notebook switches to **Python 3.11**.
- **Session restart required** after STEP 1 so that Colab uses Python 3.11.
- **Import aliases to avoid name conflicts:**  
  `import pycaret.classification as pc`, `pycaret.regression as pr`, etc., to sidestep `setup()` collisions.
- **Disable experiment logging in Colab:**  
  Use `log_experiment=False` in `setup()` to avoid an `mlflow` `AttributeError`.
- **Feature importance plotting:**  
  Wrap `plot_model(model, plot='feature')` in `try/except` to handle models that don’t expose feature importance.
- **PyCaret 3.x API changes:**  
  - Anomaly Detection: `contamination` is passed to `create_model` (not `setup`).  
  - Time Series: remove deprecated `exogenous_variables` argument; exogenous features are inferred from data passed to `setup()`.

---

## Tasks Overview

### Task A — Supervised Learning

**Dataset target fields are engineered in-notebook.** Models are selected via `compare_models()` and then tuned with `tune_model()`.

#### A-1: Binary Classification
- **Goal:** Predict whether a session resulted in **high calorie burn** (> 300 calories).  
- **Target:** `high_calorie_burn` (1/0)  
- **Modeling:** `tune_model(compare_models(optimize='AUC'))`  
- **Outputs:** Leaderboard, tuned model, confusion matrix, ROC, feature plots (when supported).

#### A-2: Multiclass Classification
- **Goal:** Predict `activity_type` (e.g., Cycling, Swimming, Yoga) from workout metrics.  
- **Target:** `activity_type`  
- **Modeling:** `tune_model(compare_models(optimize='Accuracy'))`  
- **Outputs:** Leaderboard, tuned model, class report, confusion matrix.  
- **Saved artifact:** Used by the **Multiclass Gradio App**.

#### A-3: Regression
- **Goal:** Predict exact `calories_burned`.  
- **Target:** `calories_burned` (numeric)  
- **Modeling:** `tune_model(compare_models(optimize='MAE'))`  
- **Outputs:** Leaderboard, tuned model, residuals, error plots.  
- **Saved artifact:** Used by the **Regression Gradio App**.

---

### Task B — Unsupervised Learning: Clustering

- **Goal:** Discover user clusters from metrics like `steps`, `heart_rate_avg`, `duration_min`.  
- **Algorithm:** **K-Means**, `num_clusters=4`  
- **Plots:** Cluster plot (PCA), **Elbow plot** for K selection.

---

### Task C — Unsupervised Learning: Anomaly Detection

- **Goal:** Flag unusual workout sessions that deviate strongly from typical patterns.  
- **Algorithm:** **Isolation Forest** (`iforest`)  
- **Plots:** **UMAP** visualization of detected anomalies vs. normal sessions.

---

### Task D — Association Rules (Skipped)

This task requires **PyCaret 2.3.5**, which depends on builds (e.g., `scipy`) incompatible with modern Colab + Python 3.11.  
All other tasks run successfully with **PyCaret 3.3.2**.

---

### Task E — Time Series Forecasting

#### E-1: Univariate Forecasting
- **Goal:** Forecast next **7 days** of total `steps` using only historical steps.  
- **Procedure:** PyCaret time series `setup()` → model selection → forecast and plots.

#### E-2: Univariate Forecasting with Exogenous Variables
- **Goal:** Forecast next **7 days** of `steps` using `duration_min` as an external regressor.  
- **Exogenous handling:** Provided to the time-series pipeline (inferred automatically by PyCaret 3.x).  
- **Outputs:** Forecast table/plot, diagnostics.

---

## Repository Structure

```
.
├─ pycaret_assignment.ipynb        # Main Colab notebook (all tasks)
├─ models/                         # Saved models/pipelines (created by notebook)
├─ outputs/                        # Exported plots, figures, and artifacts
└─ apps/                           # (Optional) Gradio helper scripts if saved
```

> Note: Folders are created on demand by the notebook. Names may vary slightly depending on how you run/save artifacts.

---

## Reproducibility Tips

- Use `session_id` (random seed) in `setup()` for deterministic splits when supported.
- Keep `log_experiment=False` in Colab to avoid mlflow conflicts.
- When rerunning, clear outputs and re-execute cells in order (after confirming Python 3.11 + PyCaret 3.3.2).

---

## License

_Add your preferred license here (e.g., MIT, Apache-2.0)._

---

### Deliverables Checklist

- ✅ **Colab Notebook:** `pycaret_assignment.ipynb` with all tasks A–E (except D, skipped).  
- ✅ **Video Tutorial:** Walkthrough explaining the execution and outputs.  
- ✅ **GitHub Repository:** Organized with code and saved outputs (plots), plus this `README.md`.

---

**Acknowledgments:** Thanks to the PyCaret team and the Kaggle community for datasets and tooling.
