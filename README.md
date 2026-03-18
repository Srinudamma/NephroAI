# 🫘 NephroAI — Chronic Kidney Disease Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <b>Explainable AI for Early CKD Detection</b><br/>
  Built on the UCI CKD Dataset · Ensemble Learning · SHAP/LIME XAI
</p>

---

## 🌐 Live Demo

👉 [nephroai.streamlit.app](https://nephroai-opx9j6oxzjhge2ohhbzc7n.streamlit.app)

---

## 📌 Overview

**NephroAI** is an explainable machine learning web application for early detection of **Chronic Kidney Disease (CKD)**. It takes patient clinical parameters as input and predicts CKD risk using a stacking ensemble classifier, with full local (LIME-style) and global (SHAP-style) feature explanations.

> ⚠️ *For research and educational purposes only. Not a substitute for clinical diagnosis.*

---

## ✨ Features

- 🔬 **CKD Risk Prediction** — probability score + HIGH / MODERATE / LOW risk classification
- 📊 **Global Feature Importance** — SHAP-equivalent permutation-based importance across all predictions
- 🔍 **Local Feature Attribution** — LIME-style per-patient explanation showing which features drove the result
- 📋 **Clinical Narrative** — auto-generated plain-English summary with renal, hematological, and electrolyte assessment
- 🏆 **Model Leaderboard** — full evaluation table (Accuracy, Precision, Recall, F1, AUC) for all 7 base models
- ⚙️ **Auto-training** — pipeline runs automatically on first deployment, no pre-built artifacts needed

---

## 🧠 ML Pipeline

```
Raw CSV → Clean & Impute → Encode → Scale → SMOTE → RFE → Train → Evaluate → Save
```

| Step | Detail |
|---|---|
| **Dataset** | UCI Chronic Kidney Disease (400 samples, 24 features) |
| **Imputation** | Iterative Imputer (MICE) |
| **Balancing** | Manual SMOTE (no imbalanced-learn dependency) |
| **Feature Selection** | Recursive Feature Elimination (RFE) — top 15 features |
| **Base Models** | Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, Gradient Boosting, AdaBoost |
| **Final Model** | Stacking Classifier (RF + GB + SVM → Logistic Regression meta-learner) |
| **Explainability** | SHAP-equivalent permutation importance + LIME-style local perturbation |

---

## 📁 Project Structure

```
NephroAI/
├── streamlit_app.py          # Main Streamlit app
├── ckd_pipeline.py           # ML training pipeline
├── ChronicKidneyDisease.csv  # UCI CKD dataset
├── requirements.txt          # Python dependencies
├── .python-version           # Pins Python 3.11
├── .gitignore
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Srinudamma/NephroAI.git
cd NephroAI
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python ckd_pipeline.py
```
This generates `ckd_artifacts.pkl` and `ckd_model_info.json`.

### 5. Launch the app
```bash
streamlit run streamlit_app.py
```

---

## 📦 Requirements

```
streamlit
numpy
pandas
matplotlib
scikit-learn
seaborn
```

---

## 🩺 Input Features

| Category | Features |
|---|---|
| **Renal & Metabolic** | Serum Creatinine, Blood Urea, Sodium, Blood Glucose, Potassium |
| **Hematological** | Hemoglobin, Packed Cell Volume, RBC Count, WBC Count |
| **Urinalysis** | Specific Gravity, Albumin, Sugar, Pus Cell Clumps |
| **Demographics** | Age, Blood Pressure, Hypertension, Appetite |

---

## 📊 Model Performance

| Model | Accuracy | AUC |
|---|---|---|
| Stacking Classifier ⭐ | ~98% | ~0.99 |
| Random Forest | ~97% | ~0.99 |
| Gradient Boosting | ~97% | ~0.98 |
| SVM | ~96% | ~0.98 |
| Logistic Regression | ~94% | ~0.97 |

> Evaluated on 20% held-out test set after SMOTE balancing on training data only.

---

## 🔍 Explainability

### Local (LIME-style)
Each prediction is explained by perturbing individual features to their scaled mean and measuring the change in predicted probability. Features pushing probability **up** are shown in red; features pushing it **down** are in green.

### Global (SHAP-equivalent)
Permutation-based feature importance averaged across the test set, normalized to 100%. Identifies which clinical markers are most predictive of CKD overall.

---

## 🗂️ Dataset

**UCI Machine Learning Repository — Chronic Kidney Disease Dataset**
- 400 patient records
- 24 clinical features
- Binary target: `ckd` / `notckd`

---

## 👤 Author

**Srinu Damma**  
[![GitHub](https://img.shields.io/badge/GitHub-Srinudamma-181717?style=flat&logo=github)](https://github.com/Srinudamma)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using Streamlit · scikit-learn · UCI CKD Dataset
</p>
