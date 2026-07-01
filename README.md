# MEDIX — Medical Early Disease Intelligence eXpert
 
A machine learning pipeline that predicts risk across **8 diseases** from a single unified model, trained on **12,168 real patient records** pulled live from public medical data sources (UCI ML Repository, Kaggle mirrors, the Framingham Heart Study, and scikit-learn's built-in datasets — no synthetic data).
 
The notebook runs end-to-end in Google Colab: it downloads the data, engineers domain-specific clinical features, trains and tunes 19 individual models plus 2 ensembles, explains every prediction with SHAP, and exposes a `predict_patient()` function that turns raw lab values into a risk score, a risk tier, and plain-language recommendations.
 
> ⚠️ **Disclaimer**: MEDIX is a screening and educational tool, not a diagnostic device. It does not replace professional medical advice. Always consult a licensed physician.
 
---
 
## Why MEDIX
 
Most disease-prediction notebooks train one model on one dataset. MEDIX instead merges **eight separate disease datasets into a single feature space** and trains one model family to recognize risk patterns across all of them simultaneously — closer to how a single screening intake might flag several conditions at once from overlapping biomarkers (age, blood pressure, glucose, cholesterol, etc.).
 
## Diseases Covered
 
| Disease | Source | Samples | Positive Rate |
|---|---|---:|---:|
| Diabetes | `plotly/datasets` GitHub mirror (Pima Indians Diabetes) | 768 | 34.9% |
| Heart Disease | `sharmaroshan/Heart-UCI-Dataset` (UCI Cleveland) | 303 | 54.5% |
| Breast Cancer | `sklearn.datasets.load_breast_cancer` (Wisconsin) | 569 | 37.3% |
| Parkinson's Disease | UCI ML Repository (id 174) | 195 | 75.4% |
| Liver Disease | UCI ML Repository — ILPD (id 225) | 583 | 71.4% |
| Chronic Kidney Disease | UCI ML Repository — CKD (id 336) | 400 | 62.5% |
| Stroke | Kaggle stroke-prediction dataset (GitHub mirror) | 5,110 | 4.9% |
| Hypertension / 10-yr CHD | Framingham Heart Study (GitHub mirror) | 4,240 | 15.2% |
 
All eight sources are fetched automatically at runtime, with a fallback mirror for diabetes, heart disease, stroke, and hypertension. If a source is unreachable, the cell raises a clear error rather than silently substituting synthetic data.
 
**Combined dataset**: 12,168 rows × 129 raw columns → 142 columns after feature engineering, spanning 8 disease labels, 19.3% overall positive rate.
 
---
 
## Pipeline
 
```
Auto-Download (8 sources)
        │
        ▼
Load, Clean & Merge  →  combined: (12,168 × 129)
        │
        ▼
EDA  →  class balance, missing values, correlations, distributions
        │
        ▼
Feature Engineering
   • categorical encoding, KNN imputation (127 cols)
   • 12+ domain-specific ratios (glucose/insulin, BUN/creatinine,
     AST/ALT, pulse pressure, smoking index, age groups, …)
   • outlier clipping (1st–99th percentile)
        │
        ▼
Train/Test Split (80/20, stratified) → RobustScaler → SMOTE balancing
        │
        ▼
15 Base Models  +  Optuna Hyperparameter Tuning (XGBoost, LightGBM,
CatBoost, Decision Tree — 10 trials each)
        │
        ▼
19 Trained Models  →  Stacking Ensemble  →  Voting Ensemble  =  21 Models
        │
        ▼
Evaluation (Accuracy, ROC-AUC, F1, MCC, Balanced Accuracy)
   + 5-Fold Cross-Validation Stability
        │
        ▼
SHAP Explainability  +  Decision Tree Visualisation (depth tuning,
full tree render, per-disease trees, feature importance, decision-path trace)
        │
        ▼
Patient Risk Prediction Engine  →  Interactive Plotly Dashboard
        │
        ▼
Save Artefacts (.pkl models, leaderboard CSVs, meta.json) → Zip → Download
```
 
## Models Trained (21 total)
 
**Tree-based**: Decision Tree (Gini), Decision Tree (Entropy), Extra Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging (DT)
**Boosting**: XGBoost, LightGBM, CatBoost — each trained both at default settings and Optuna-tuned (10 trials, 5-fold CV, ROC-AUC objective)
**Linear / probabilistic**: Logistic Regression, Naive Bayes
**Distance-based**: K-Nearest Neighbors
**Neural**: MLP (256-128-64, ReLU, early stopping)
**Ensembles**: Stacking Classifier (top 4 base models + logistic meta-learner), Voting Classifier (soft voting, top 5 base models)
 
---
 
## Results
 
### Leaderboard (held-out test set, top 10 of 21 models)
 
| Rank | Model | Accuracy | ROC-AUC | F1 | MCC | Balanced Acc |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Voting Ensemble** | 0.8776 | **0.8792** | 0.8711 | 0.5761 | 0.7590 |
| 2 | Random Forest | 0.8739 | 0.8774 | 0.8693 | 0.5712 | 0.7657 |
| 3 | Extra Trees | 0.8677 | 0.8754 | 0.8629 | 0.5496 | 0.7554 |
| 4 | LightGBM | 0.8759 | 0.8740 | 0.8698 | 0.5717 | 0.7588 |
| 5 | CatBoost | 0.8735 | 0.8734 | 0.8675 | 0.5640 | 0.7565 |
| 6 | Stacking Ensemble | 0.8767 | 0.8732 | 0.8685 | 0.5676 | 0.7480 |
| 7 | LightGBM (Tuned) | 0.8735 | 0.8732 | 0.8669 | 0.5619 | 0.7533 |
| 8 | XGBoost | 0.8722 | 0.8726 | 0.8667 | 0.5616 | 0.7574 |
| 9 | Gradient Boosting | 0.8661 | 0.8685 | 0.8620 | 0.5476 | 0.7576 |
| 10 | XGBoost (Tuned) | 0.8665 | 0.8671 | 0.8605 | 0.5408 | 0.7473 |
 
**Best model: Voting Ensemble** — 87.76% accuracy, 0.8792 ROC-AUC, 0.8711 F1, 0.5761 MCC on the held-out test set.
 
### Per-Disease Breakdown (Voting Ensemble)
 
| Disease | Test Samples | Accuracy | ROC-AUC | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Chronic Kidney Disease | 80 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Breast Cancer | 128 | 0.9297 | 0.9835 | 0.9303 | 0.8549 |
| Heart Disease | 62 | 0.8065 | 0.8740 | 0.8065 | 0.6125 |
| Diabetes | 146 | 0.7534 | 0.8391 | 0.7509 | 0.4813 |
| Parkinson's Disease | 40 | 0.8000 | 0.8339 | 0.7933 | 0.4726 |
| Stroke | 1,006 | 0.9503 | 0.8163 | 0.9305 | 0.1770 |
| Liver Disease | 110 | 0.7000 | 0.7504 | 0.6478 | 0.3087 |
| Hypertension / CHD | 862 | 0.8260 | 0.6704 | 0.7835 | 0.0516 |
 
Stroke and Hypertension/CHD show high accuracy but low MCC — both are heavily imbalanced datasets (4.9% and 15.2% positive respectively), so accuracy alone overstates real-world usefulness there; ROC-AUC and MCC are the more honest metrics for those two.
 
### Cross-Validation Stability
 
5-fold CV (on the SMOTE-balanced training set) confirms the top models generalize consistently, with very low variance across folds:
 
| Model | Mean ROC-AUC | Std Dev |
|---|---:|---:|
| Stacking Ensemble | 0.9893 | 0.0007 |
| Extra Trees | 0.9882 | 0.0005 |
| Voting Ensemble | 0.9851 | 0.0008 |
| Random Forest | 0.9816 | 0.0009 |
| LightGBM (Tuned) | 0.9804 | 0.0011 |
| LightGBM | 0.9802 | 0.0015 |
| XGBoost | 0.9795 | 0.0013 |
| CatBoost | 0.9767 | 0.0010 |
 
*Note: these CV scores are computed on the SMOTE-balanced training data, so they read higher than the held-out test-set leaderboard above — that gap is expected and the test-set numbers are the more representative estimate of real-world performance.*
 
---
 
## Explainability
 
- **SHAP (TreeExplainer)** on the best tuned boosting model — global feature importance (bar + beeswarm), dependence plots, and a per-patient waterfall plot showing exactly which biomarkers pushed an individual prediction up or down.
- **Decision Tree analysis** — a CV-driven depth-tuning curve (sweet spot at depth 8, 0.9080 CV ROC-AUC), a rendered depth-3 tree for clinician readability, per-disease trees trained independently (Chronic Kidney Disease and Parkinson's score highest in isolation, Hypertension lowest), a global Gini feature-importance ranking, and a step-by-step decision-path trace for a sample high-risk patient.
---
 
## Patient Risk Prediction Engine
 
```python
demo_patient = dict(
    Pregnancies=6, Glucose=148, BloodPressure=72, SkinThickness=35,
    Insulin=200, BMI=33.6, DiabetesPedigreeFunction=0.627, Age=50
)
 
result = predict_patient(demo_patient)
# {
#   'risk_probability': 23.0,
#   'risk_level': '🟢 LOW RISK',
#   'shap_top5': [...],
#   'recommendations': [...]
# }
```
 
Any subset of the 142 model features can be passed in — missing values are automatically median-imputed from the training distribution. The function prints a formatted risk report (probability, tier, top-5 SHAP-ranked biomarkers, tier-appropriate recommendations, and a medical disclaimer) and returns the same data as a dict for programmatic use.
 
Risk tiers: 🟢 Low (<25%) · 🟡 Moderate (25–50%) · 🟠 High (50–75%) · 🔴 Critical (≥75%)
 
## Interactive Dashboard
 
A 6-panel Plotly dashboard (`risk_dashboard.html`) renders per prediction: a risk-score gauge, top-10 SHAP feature contributions, the model leaderboard, per-disease ROC-AUC bars, a labeled risk scale with the patient's position marked, and a table of the raw input values.
 
## Standalone Inference
 
After downloading the artefacts zip, predictions can be made independently of the training notebook:
 
```python
def infer(patient_dict: dict) -> dict:
    """Load saved model/scaler/encoders and predict from a dict of lab values."""
    ...
 
infer({"Glucose": 170, "BMI": 34, "Age": 55, "BloodPressure": 85, "Insulin": 220})
# {'risk_probability': 15.67, 'risk_level': '🟢 LOW RISK'}
```
 
---
 
## Tech Stack
 
| Category | Tools |
|---|---|
| Core | Python, NumPy, Pandas |
| ML / Tuning | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| Persistence | joblib |
| Environment | Google Colab |
 
## Project Structure
 
```
MEDIX_Multi_Disease_Risk_Prediction.ipynb   # the full pipeline, run top to bottom
datasets/                                    # auto-downloaded CSVs (8 files, gitignored)
artefacts/
  ├── best_model.pkl
  ├── scaler.pkl
  ├── imputer.pkl
  ├── label_encoders.pkl
  ├── disease_label_encoder.pkl
  ├── shap_explainer.pkl
  ├── model_leaderboard.csv
  ├── per_disease_breakdown.csv
  └── meta.json
risk_dashboard.html                          # interactive Plotly dashboard
MultiDiseaseRiskForecast_v2_artefacts.zip    # all artefacts, packaged for download
```
 
## Getting Started
 
1. Open `MEDIX_Multi_Disease_Risk_Prediction.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells top to bottom — dependencies install automatically in the first cell.
3. Datasets download automatically; no manual data prep or API keys required.
4. The full pipeline (data → 21 models → SHAP → dashboard → artefacts) completes in a single run.
5. Use `predict_patient({...})` inside the notebook, or download the artefacts zip and use the standalone `infer({...})` function anywhere else.
**Runtime note**: `N_TRIALS = 10` controls Optuna tuning depth (trials per model for XGBoost, LightGBM, CatBoost, and Decision Tree). Raise it (e.g. to 40–80) for marginally better tuned-model scores at the cost of longer runtime.
 
---
 
## Roadmap
 
- [ ] Render full Random Forest trees (individual estimators) alongside the existing Decision Tree visualizations
- [ ] Gradio UI for interactive, no-code patient input and live risk prediction
- [ ] Model versioning / experiment tracking

 
