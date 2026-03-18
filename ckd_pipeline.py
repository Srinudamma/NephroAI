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

# Encode categoricals
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    mask = X[col].notna()
    encoded = le.fit_transform(X.loc[mask, col].astype(str))
    X[col] = X[col].astype(object)
    X.loc[mask, col] = encoded.astype(float)
    X[col] = pd.to_numeric(X[col], errors='coerce')
    encoders[col] = le

# Iterative imputation
print("\nApplying Iterative Imputer...")
imputer = IterativeImputer(max_iter=10, random_state=42)
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns)

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

print("Applying SMOTE...")
X_arr = X_scaled.values
X_res, y_res = manual_smote(X_arr, y.values)
print(f"After SMOTE: {X_res.shape[0]} samples, class balance: {np.bincount(y_res)}")

# ─── Feature Selection via RFE ────────────────────────────────────────────────

print("\nApplying RFE with Random Forest...")
rf_rfe = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(rf_rfe, n_features_to_select=15, step=1)
rfe.fit(X_res, y_res)

selected_features = X_scaled.columns[rfe.support_].tolist()
print(f"Selected features ({len(selected_features)}): {selected_features}")

X_sel = X_res[:, rfe.support_]
X_orig_sel = X_arr[:, rfe.support_]  # original (pre-SMOTE) for evaluation

# ─── Model Training ───────────────────────────────────────────────────────────

print("\nTraining models...")

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(probability=True, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in base_models.items():
    scores = cross_val_score(model, X_sel, y_res, cv=cv, scoring='roc_auc')
    results[name] = {'mean_auc': scores.mean(), 'std_auc': scores.std()}
    print(f"  {name:25s}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# Ensemble: Voting
voting_clf = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
    ],
    voting='soft'
)
v_scores = cross_val_score(voting_clf, X_sel, y_res, cv=cv, scoring='roc_auc')
results['Voting Ensemble'] = {'mean_auc': v_scores.mean(), 'std_auc': v_scores.std()}
print(f"  {'Voting Ensemble':25s}: AUC = {v_scores.mean():.4f} ± {v_scores.std():.4f}")

# Ensemble: Stacking
stacking_clf = StackingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3
)
s_scores = cross_val_score(stacking_clf, X_sel, y_res, cv=cv, scoring='roc_auc')
results['Stacking Ensemble'] = {'mean_auc': s_scores.mean(), 'std_auc': s_scores.std()}
print(f"  {'Stacking Ensemble':25s}: AUC = {s_scores.mean():.4f} ± {s_scores.std():.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['mean_auc'])
print(f"\nBest model: {best_name} (AUC={results[best_name]['mean_auc']:.4f})")

# Train best model on full SMOTE data
print(f"\nTraining final model ({best_name}) on full dataset...")

model_map = {
    'Random Forest':      RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=200, random_state=42),
    'AdaBoost':           AdaBoostClassifier(n_estimators=200, random_state=42),
    'Logistic Regression':LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes':        GaussianNB(),
    'SVM':                SVC(probability=True, random_state=42),
    'Decision Tree':      DecisionTreeClassifier(max_depth=6, random_state=42),
    'Voting Ensemble':    voting_clf,
    'Stacking Ensemble':  stacking_clf,
}

final_model = model_map.get(best_name, RandomForestClassifier(n_estimators=200, random_state=42))
final_model.fit(X_sel, y_res)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\nCreating Train-Test Split for Evaluation...")

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y_res, test_size=0.2, random_state=42, stratify=y_res
)

evaluation_results = []

print("\nEvaluating Models on Test Set...\n")

for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

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
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

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

# Use best model
best_model = final_model

y_pred = best_model.predict(X_test)
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
fi = get_feature_importance(final_model, X_orig_sel, y.values, selected_features)
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

artifacts = {
    "model": final_model,
    "imputer": imputer,        # ensure object stored
    "scaler": scaler,
    "encoders": encoders,
    "rfe": rfe,
    "selected_features": selected_features,
    "all_features": list(X.columns),
    "numeric_cols": NUMERIC_COLS,
    "cat_cols": CAT_COLS,
    "model_results": results,
    "best_model_name": best_name,
    "global_importances": fi_sorted,
    'feature_stats': {
        col: {
            'mean': float(X_imp[col].mean()),
            'std': float(X_imp[col].std()),
            'min': float(X_imp[col].min()),
            'max': float(X_imp[col].max()),
        } for col in X.columns
    }
}
import os

save_path = os.path.join(os.getcwd(), "ckd_artifacts.pkl")

with open(save_path, "wb") as f:
    pickle.dump(artifacts, f)

# Also save as JSON for the web app
json_artifacts = {
    'selected_features': selected_features,
    'all_features': list(X.columns),
    'numeric_cols': NUMERIC_COLS,
    'cat_cols': CAT_COLS,
    'model_results': {k: {'mean_auc': float(v['mean_auc']), 'std_auc': float(v['std_auc'])}
                      for k,v in results.items()},
    'best_model_name': best_name,
    'global_importances': fi_sorted,
    'feature_stats': {
        col: {
            'mean': float(X_imp[col].mean()),
            'std': float(X_imp[col].std()),
            'min': float(X_imp[col].min()),
            'max': float(X_imp[col].max()),
        } for col in X.columns
    }
}

json_path = os.path.join(os.getcwd(), "ckd_model_info.json")

with open(json_path, "w") as f:
    json.dump(json_artifacts, f, indent=2)

print("Artifacts saved: ckd_artifacts.pkl, ckd_model_info.json")
print("\n✅ Pipeline complete!")
