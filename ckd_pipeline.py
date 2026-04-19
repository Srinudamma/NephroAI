"""
CKD Prediction Pipeline — FIXED VERSION
Trains all models, computes feature importances, saves artifacts.

Fixes applied:
  1. Lambda closure bug fixed (enc=le default arg)
  2. Saves `evaluation_results` (full metrics) instead of partial `results`
  3. get_feature_importance now correctly unpacks StackingClassifier estimator tuples
  4. Models trained only ONCE — predictions reused across loops
  5. plt.show() removed for headless/server compatibility
  6. local_explain signature cleaned up (removed unused background args)
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, classification_report,
                              confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Then replace all bare filenames with:
df = pd.read_csv(os.path.join(BASE_DIR, 'ChronicKidneyDisease.csv'))

# And the save paths:
save_path = os.path.join(BASE_DIR, 'ckd_artifacts.pkl')
json_path = os.path.join(BASE_DIR, 'ckd_model_info.json')


# ─── Dataset Loading ──────────────────────────────────────────────────────────

print("Loading ChronicKidneyDisease.csv dataset...")

df = pd.read_csv('ChronicKidneyDisease.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Rename columns for consistency
df = df.rename(columns={
    'classification': 'class',
    'wc':  'wbcc',
    'rc':  'rbcc'
})

# Drop unnecessary column
df = df.drop('id', axis=1, errors='coerce')
df = df.fillna(0)

# Clean target column
df['class'] = df['class'].astype(str).str.strip().str.lower()
df['class'] = df['class'].map({'ckd': 'ckd', 'notckd': 'notckd'})

# Remove rows with unknown class
df = df[df['class'].isin(['ckd', 'notckd'])]

# Replace weird missing values
df = df.replace(['?', ' ', ''], np.nan)

# Convert numeric columns
NUMERIC_COLS = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
                'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts()}")
print("\nMissing values:\n", df.isnull().sum())

# ─── Preprocessing ────────────────────────────────────────────────────────────

CAT_COLS = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

X = df.drop('class', axis=1).copy()
y = (df['class'] == 'ckd').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── FIX 1: Lambda closure bug fixed with `enc=le` default argument ───────────
encoders = {}

for col in CAT_COLS:
    le = LabelEncoder()

    # Fit ONLY on training data
    mask = X_train[col].notna()
    le.fit(X_train.loc[mask, col].astype(str))

    # FIX: capture `le` in default arg to avoid closure over loop variable
    X_train[col] = X_train[col].map(
        lambda x, enc=le: enc.transform([str(x)])[0] if pd.notna(x) else np.nan
    )
    X_test[col] = X_test[col].map(
        lambda x, enc=le: enc.transform([str(x)])[0] if str(x) in enc.classes_ else np.nan
    )

    encoders[col] = le

# ─── Imputation (fit only on train) ──────────────────────────────────────────

print("\nApplying Iterative Imputer...")

imputer = IterativeImputer(max_iter=10, random_state=42)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)

# ─── Scaling (fit only on train) ─────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

# ─── SMOTE (manual, train only) ──────────────────────────────────────────────

def manual_smote(X, y, k=5, random_state=42):
    """Minimal SMOTE implementation — no imbalanced-learn dependency."""
    np.random.seed(random_state)
    minority_idx = np.where(y == 0)[0]
    majority_idx = np.where(y == 1)[0]

    if len(minority_idx) >= len(majority_idx):
        return X, y

    n_synthetic = len(majority_idx) - len(minority_idx)
    X_min = X[minority_idx]
    synthetic = []

    for _ in range(n_synthetic):
        idx    = np.random.randint(len(X_min))
        sample = X_min[idx]
        dists  = np.linalg.norm(X_min - sample, axis=1)
        dists[idx] = np.inf
        nn_idx = np.argsort(dists)[:k]
        nn     = X_min[np.random.choice(nn_idx)]
        alpha  = np.random.rand()
        synthetic.append(sample + alpha * (nn - sample))

    synthetic = np.array(synthetic)
    X_res = np.vstack([X.values if hasattr(X, 'values') else X, synthetic])
    y_res = np.concatenate([y, np.zeros(n_synthetic, dtype=int)])
    return X_res, y_res


print("Applying SMOTE (Train Only)...")
X_train_res, y_train_res = manual_smote(X_train_scaled, y_train.values)
print("After SMOTE:", np.bincount(y_train_res))

# ─── Feature Selection via RFE (train only) ───────────────────────────────────

print("\nApplying RFE...")

rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=15)
X_train_sel = rfe.fit_transform(X_train_res, y_train_res)
X_test_sel  = rfe.transform(X_test_scaled)

selected_features = X_train.columns[rfe.support_]
print("Selected Features:", list(selected_features))

# ─── FIX 2: Train each model ONCE, reuse predictions ─────────────────────────

print("\nTraining and evaluating base models (single pass)...")

base_model_configs = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(probability=True, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
}

evaluation_results = []   # FIX 2: single list with FULL metrics
trained_base_models = {}  # FIX 2: store trained models to reuse predictions
base_predictions    = {}  # FIX 2: store predictions to reuse in ROC curve

for name, mdl in base_model_configs.items():
    mdl.fit(X_train_sel, y_train_res)
    trained_base_models[name] = mdl

    y_pred = mdl.predict(X_test_sel)
    y_prob = mdl.predict_proba(X_test_sel)[:, 1]
    base_predictions[name] = (y_pred, y_prob)   # cache

    evaluation_results.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "AUC":       roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(evaluation_results).sort_values(by="AUC", ascending=False)
print("\n📊 Experimental Results Table:\n")
print(results_df.to_string(index=False))

# ─── Ensemble Models ──────────────────────────────────────────────────────────

print("\nTraining Ensemble Models...")

voting_clf = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb',  GradientBoostingClassifier(random_state=42)),
        ('lr',  LogisticRegression(max_iter=1000))
    ],
    voting='soft'
)

stacking_clf = StackingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb',  GradientBoostingClassifier(random_state=42)),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression(),
    cv=3
)

voting_clf.fit(X_train_sel, y_train_res)
stacking_clf.fit(X_train_sel, y_train_res)

voting_prob   = voting_clf.predict_proba(X_test_sel)[:, 1]
stacking_prob = stacking_clf.predict_proba(X_test_sel)[:, 1]

print(f"Voting AUC:   {roc_auc_score(y_test, voting_prob):.4f}")
print(f"Stacking AUC: {roc_auc_score(y_test, stacking_prob):.4f}")

final_model = stacking_clf
print("\n✅ Final Model: Stacking Classifier")

# ─── ROC Curve (FIX 2: reuse cached predictions, no retraining) ───────────────

import matplotlib
matplotlib.use('Agg')   # FIX 5: headless backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
for name, (_, y_prob) in base_predictions.items():   # reuse cached probs
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()   # FIX 5: close instead of show()
print("Saved: roc_curve.png")

# ─── Confusion Matrix ────────────────────────────────────────────────────────

y_pred_final = final_model.predict(X_test_sel)
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix — Stacking Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()   # FIX 5
print("Saved: confusion_matrix.png")

# ─── Feature Importances ──────────────────────────────────────────────────────

print("\nComputing feature importances...")

def permutation_importance_fn(model, X, y, feature_names, n_repeats=5):
    """Permutation-based importance fallback."""
    base_score = roc_auc_score(y, model.predict_proba(X)[:, 1])
    importances = {}
    for i, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            try:
                s = roc_auc_score(y, model.predict_proba(X_perm)[:, 1])
            except Exception:
                s = base_score
            scores.append(base_score - s)
        importances[name] = float(np.mean(scores))
    return importances


def get_feature_importance(model, X, y, feature_names):
    """
    Get normalised feature importances from model.
    FIX 3: correctly unpacks StackingClassifier estimator tuples.
    """
    importances = {}

    if hasattr(model, 'feature_importances_'):
        # Direct tree-based importance
        fi = model.feature_importances_
        importances = {name: float(fi[i]) for i, name in enumerate(feature_names)}

    elif hasattr(model, 'estimators_'):
        # Ensemble — FIX 3: handle (name, estimator) tuples from StackingClassifier
        all_fi = []
        for est in model.estimators_:
            actual_est = est[1] if isinstance(est, tuple) else est   # ← FIX 3
            if hasattr(actual_est, 'feature_importances_'):
                all_fi.append(actual_est.feature_importances_)

        if all_fi:
            fi = np.mean(all_fi, axis=0)
            importances = {name: float(fi[i]) for i, name in enumerate(feature_names)}
        else:
            importances = permutation_importance_fn(model, X, y, feature_names)
    else:
        importances = permutation_importance_fn(model, X, y, feature_names)

    # Normalise
    total = sum(abs(v) for v in importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}

    return importances


fi = get_feature_importance(final_model, X_test_sel, y_test.values, selected_features)
fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

print("Feature importances:")
for feat, importance in fi_sorted.items():
    print(f"  {feat:12s}: {importance:.4f}")

# ─── Local Explanation Function (FIX 6: remove unused background args) ────────

def local_explain(model, sample, feature_names):
    """
    LIME-style local explanation by feature perturbation.
    FIX 6: removed unused background_mean / background_std parameters.
    """
    base_pred = model.predict_proba(sample.reshape(1, -1))[0, 1]
    contributions = {}
    for i, name in enumerate(feature_names):
        perturbed    = sample.copy()
        perturbed[i] = 0.0    # scaled mean = 0
        new_pred     = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
        contributions[name] = float(base_pred - new_pred)
    return contributions, base_pred

# ─── Save Artifacts ───────────────────────────────────────────────────────────

print(f"\nDEBUG imputer type: {type(imputer)}")
print("Saving artifacts...")

feature_stats = {
    col: {
        'mean': float(X_train_imp[col].mean()),
        'std':  float(X_train_imp[col].std()),
        'min':  float(X_train_imp[col].min()),
        'max':  float(X_train_imp[col].max()),
    }
    for col in X_train.columns
}

artifacts = {
    "model":             final_model,
    "imputer":           imputer,
    "scaler":            scaler,
    "encoders":          encoders,
    "rfe":               rfe,
    "selected_features": list(selected_features),
    "all_features":      list(X.columns),
    "numeric_cols":      NUMERIC_COLS,
    "cat_cols":          CAT_COLS,
    "model_results":     evaluation_results,   # FIX 2: full metrics
    "best_model_name":   "Stacking",
    "global_importances": fi_sorted,
    "feature_stats":     feature_stats
}

save_path = os.path.join(os.getcwd(), "ckd_artifacts.pkl")
with open(save_path, "wb") as f:
    pickle.dump(artifacts, f)

# JSON sidecar (non-model info for web use)
json_artifacts = {
    'selected_features':  list(selected_features),
    'all_features':       list(X.columns),
    'numeric_cols':       NUMERIC_COLS,
    'cat_cols':           CAT_COLS,
    'model_results':      evaluation_results,   # FIX 2: full metrics
    'best_model_name':    "Stacking",
    'global_importances': fi_sorted,
    'feature_stats':      feature_stats
}

json_path = os.path.join(os.getcwd(), "ckd_model_info.json")
with open(json_path, "w") as f:
    json.dump(json_artifacts, f, indent=2)

print("Artifacts saved: ckd_artifacts.pkl, ckd_model_info.json")
print("\n✅ Pipeline complete!")