# Diabetes Onset Prediction — Logistic Regression + Hyperparameter Tuning

**Problem.** Predict diabetes onset from clinical features to support early intervention.

**Data.** `data/raw/diabetes_clean.csv` (columns include: pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age, diabetes).

**Approach.**
- Baseline **Logistic Regression**; ROC curve + confusion matrix.
- Compared against **KNN** (logistic performed better across metrics).
- **Hyperparameter tuning** with `GridSearchCV` and `RandomizedSearchCV`.
- Evaluated with train/test split; tracked accuracy, precision, recall, F1, ROC-AUC.

**Results.**
- Logistic > KNN on all reported metrics.
- ROC-AUC ≈ 0.801; accuracy ≈ 0.68; balanced performance across classes.

**What I Learned.**
- Interpreting coefficients and thresholds via ROC.
- Why CV-based tuning improves generalization.
- How metric choice (F1 vs AUC) shifts model selection.

## Quick Start
```bash
# clone if standalone
git clone https://github.com/Joe-Naz01/fine_tuning_supervised.git
cd fine_tuning_supervised

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
