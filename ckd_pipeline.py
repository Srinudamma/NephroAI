"""
CKD Prediction Pipeline
Trains all models, computes feature importances, saves artifacts.
"""

import numpy as np
import pandas as pd
import json
import pickle
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.pipeline import Pipeline

# ─── Dataset: REAL CKD Dataset Loading ───────────────────────────────────────

print("Loading ChronicKidneyDisease.csv dataset...")

df = pd.read_csv('ChronicKidneyDisease.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Rename columns for consistency (important for pipeline)
df = df.rename(columns={
    'classification': 'class',
    'wc': 'wbcc',
    'rc': 'rbcc'
})

# Drop unnecessary column
df = df.drop('id', axis=1, errors='ignore')

# Clean target column
df['class'] = df['class'].astype(str).str.strip().str.lower()
df['class'] = df['class'].map({
    'ckd': 'ckd',
    'notckd': 'notckd'
})

# Remove rows with unknown class (if any)
df = df[df['class'].isin(['ckd', 'notckd'])]

# Replace weird missing values (?, blanks)
df = df.replace(['?', ' ', ''], np.nan)

# Convert numeric columns properly
NUMERIC_COLS = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc']
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts()}")
print("\nMissing values:\n", df.isnull().sum())

# ─── Preprocessing ────────────────────────────────────────────────────────────

NUMERIC_COLS = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc']
CAT_COLS     = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']

X = df.drop('class', axis=1).copy()
y = (df['class'] == 'ckd').astype(int)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Encode categoricals
encoders = {}

for col in CAT_COLS:
    le = LabelEncoder()

    # Fit ONLY on training data
    mask = X_train[col].notna()
    le.fit(X_train.loc[mask, col].astype(str))

    # Transform train
    X_train[col] = X_train[col].map(
        lambda x: le.transform([str(x)])[0] if pd.notna(x) else np.nan
    )

    # Transform test (handle unseen values safely)
    X_test[col] = X_test[col].map(
        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else np.nan
    )

    encoders[col] = le

# ─── IMPUTATION (FIT ONLY ON TRAIN) ─────────────────────

print("\nApplying Iterative Imputer...")

imputer = IterativeImputer(max_iter=10, random_state=42)

X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

# Convert back to DataFrame
X_train_imp = pd.DataFrame(X_train_imp, columns=X_train.columns)
X_test_imp  = pd.DataFrame(X_test_imp, columns=X_test.columns)


# ─── SCALING (FIT ONLY ON TRAIN) ───────────────────────

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

# ─── SMOTE (manual implementation without imbalanced-learn) ──────────────────

def manual_smote(X, y, k=5, random_state=42):
    """Minimal SMOTE implementation."""
    np.random.seed(random_state)
    minority_idx = np.where(y == 0)[0]
    majority_idx = np.where(y == 1)[0]
    
    if len(minority_idx) >= len(majority_idx):
        return X, y
    
    n_synthetic = len(majority_idx) - len(minority_idx)
    X_min = X[minority_idx]
    synthetic = []
    
    for _ in range(n_synthetic):
        idx = np.random.randint(len(X_min))
        sample = X_min[idx]
        # Find k nearest neighbors
        dists = np.linalg.norm(X_min - sample, axis=1)
        dists[idx] = np.inf
        nn_idx = np.argsort(dists)[:k]
        nn = X_min[np.random.choice(nn_idx)]
        alpha = np.random.rand()
        synthetic.append(sample + alpha * (nn - sample))
    
    synthetic = np.array(synthetic)
    X_res = np.vstack([X.values if hasattr(X,'values') else X,
                       synthetic])
    y_res = np.concatenate([y, np.zeros(n_synthetic, dtype=int)])
    return X_res, y_res

print("Applying SMOTE (Train Only)...")

X_train_arr = X_train_scaled
y_train_arr = y_train.values

X_train_res, y_train_res = manual_smote(X_train_arr, y_train_arr)

print("After SMOTE:", np.bincount(y_train_res))

# ─── Feature Selection via RFE ────────────────────────────────────────────────
# ─── RFE (TRAIN ONLY) ─────────────────────────

print("\nApplying RFE...")

rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=15)

X_train_sel = rfe.fit_transform(X_train_res, y_train_res)
X_test_sel  = rfe.transform(X_test_scaled)

selected_features = X_train.columns[rfe.support_]

print("Selected Features:", selected_features)

# ─── Model Training ───────────────────────────────────────────────────────────

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(probability=True, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
}

print("\nTraining models...")

results = []

for name, model in base_models.items():
    model.fit(X_train_sel, y_train_res)

    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)

print("\n📊 Results:\n", results_df)

# ─── ENSEMBLE MODELS (CORRECTED) ─────────────────────────

print("\nEvaluating Ensembles...")

# Define models (keep your config, just updated n_estimators optional)
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ],
    voting='soft'
)

stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression(),
    cv=3
)

# Train ONLY on TRAIN data
voting_clf.fit(X_train_sel, y_train_res)
stacking_clf.fit(X_train_sel, y_train_res)

# Evaluate on TEST data
voting_prob = voting_clf.predict_proba(X_test_sel)[:, 1]
stacking_prob = stacking_clf.predict_proba(X_test_sel)[:, 1]

v_auc = roc_auc_score(y_test, voting_prob)
s_auc = roc_auc_score(y_test, stacking_prob)

print("Voting AUC:", v_auc)
print("Stacking AUC:", s_auc)
# Final model = Stacking (fixed)
final_model = stacking_clf
final_model.fit(X_train_sel, y_train_res)

print("\n✅ Final Model: Stacking Classifier")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
evaluation_results = []

print("\nEvaluating Models on Test Set...\n")

for name, model in base_models.items():
    model.fit(X_train_sel, y_train_res)

    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]

    evaluation_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    })

# Convert to DataFrame
results_df = pd.DataFrame(evaluation_results)
results_df = results_df.sort_values(by="AUC", ascending=False)

print("\n📊 Experimental Results Table:\n")
print(results_df)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

plt.figure()
for name, model in base_models.items():
    model.fit(X_train_sel, y_train_res)

    y_prob = model.predict_proba(X_test_sel)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()

plt.savefig("roc_curve.png")
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix


y_pred = final_model.predict(X_test_sel)

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

# ─── Feature Importances (SHAP-equivalent via permutation + tree importance) ──

print("\nComputing feature importances...")

def get_feature_importance(model, X, y, feature_names):
    """Get feature importances from model or permutation."""
    importances = {}
    
    # Tree-based importance
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        for i, name in enumerate(feature_names):
            importances[name] = float(fi[i])
    elif hasattr(model, 'estimators_'):
        # Ensemble - average
        all_fi = []
        for est in model.estimators_:
            if hasattr(est, 'feature_importances_'):
                all_fi.append(est.feature_importances_)
        if all_fi:
            fi = np.mean(all_fi, axis=0)
            for i, name in enumerate(feature_names):
                importances[name] = float(fi[i])
        else:
            # Permutation fallback
            importances = permutation_importance(model, X, y, feature_names)
    else:
        importances = permutation_importance(model, X, y, feature_names)
    
    # Normalize
    total = sum(abs(v) for v in importances.values())
    if total > 0:
        importances = {k: v/total for k, v in importances.items()}
    return importances

def permutation_importance(model, X, y, feature_names, n_repeats=5):
    base_score = roc_auc_score(y, model.predict_proba(X)[:,1])
    importances = {}
    for i, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            try:
                s = roc_auc_score(y, model.predict_proba(X_perm)[:,1])
            except:
                s = base_score
            scores.append(base_score - s)
        importances[name] = float(np.mean(scores))
    return importances

# Use original (non-SMOTE) data for importance evaluation
fi = get_feature_importance(final_model, X_test_sel, y_test.values, selected_features)
fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

print("Feature importances:")
for feat, importance in fi_sorted.items():
    print(f"  {feat:12s}: {importance:.4f}")

# ─── Per-sample local explanation function ────────────────────────────────────

def local_explain(model, sample, feature_names, background_mean, background_std):
    """
    Compute local feature contributions using linear approximation (LIME-style).
    Perturbs each feature and measures prediction change.
    """
    base_pred = model.predict_proba(sample.reshape(1,-1))[0,1]
    contributions = {}
    
    for i, name in enumerate(feature_names):
        # Perturb feature to mean (neutralize)
        perturbed = sample.copy()
        perturbed[i] = 0.0  # scaled mean = 0
        new_pred = model.predict_proba(perturbed.reshape(1,-1))[0,1]
        contributions[name] = float(base_pred - new_pred)
    
    return contributions, base_pred


# ─── Save Artifacts ───────────────────────────────────────────────────────────
print("DEBUG imputer type:", type(imputer))

print("\nSaving artifacts...")

feature_stats = {
    col: {
        'mean': float(X_train_imp[col].mean()),
        'std': float(X_train_imp[col].std()),
        'min': float(X_train_imp[col].min()),
        'max': float(X_train_imp[col].max()),
    }
    for col in X_train.columns
}

artifacts = {
    "model": final_model,
    "imputer": imputer,        # ensure object stored
    "scaler": scaler,
    "encoders": encoders,
    "rfe": rfe,
    "selected_features": list(selected_features),
    "all_features": list(X.columns),
    "numeric_cols": NUMERIC_COLS,
    "cat_cols": CAT_COLS,
    "model_results": results,
    "best_model_name": "Stacking",
    "global_importances": fi_sorted,
    "feature_stats": feature_stats
}
import os

save_path = os.path.join(os.getcwd(), "ckd_artifacts.pkl")

with open(save_path, "wb") as f:
    pickle.dump(artifacts, f)

# Also save as JSON for the web app
json_artifacts = {
    'selected_features': list(selected_features),
    'all_features': list(X.columns),
    'numeric_cols': NUMERIC_COLS,
    'cat_cols': CAT_COLS,
    'model_results': results,
    'best_model_name': "Stacking",
    'global_importances': fi_sorted,
    "feature_stats": feature_stats
}

json_path = os.path.join(os.getcwd(), "ckd_model_info.json")

with open(json_path, "w") as f:
    json.dump(json_artifacts, f, indent=2)

print("Artifacts saved: ckd_artifacts.pkl, ckd_model_info.json")
print("\n✅ Pipeline complete!")
