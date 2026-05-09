"""
Heart Disease Prediction - ML Training Pipeline
================================================
Trains: Logistic Regression, Decision Tree, Random Forest, Neural Network
Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC
Outputs: Saved models, plots, and results CSV
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import joblib

# ── Tensorflow / Keras ─────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
    print(f"TensorFlow {tf.__version__} loaded.")
except ImportError:
    HAS_TF = False
    print("TensorFlow not available – Neural Network model will be skipped.")

# ── Directories ────────────────────────────────────────────────────────────────
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

PALETTE = "#1e3a5f"
ACCENT  = "#e63946"

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & UNDERSTANDING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  HEART DISEASE PREDICTION – ML PIPELINE")
print("="*60)

df = pd.read_csv("heart.csv")
df.dropna(inplace=True)          # drop any blank trailing rows
df = df.reset_index(drop=True)

print(f"\n[DATA] Shape: {df.shape}")
print(df.head())
print("\n[INFO] Dtypes & Missing Values:")
print(df.isnull().sum())
print("\n[STAT] Descriptive Statistics:")
print(df.describe().round(2))

# Column descriptions for reference
COL_INFO = {
    "age":      "Age of patient",
    "sex":      "Sex (1=male, 0=female)",
    "cp":       "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol":     "Serum cholesterol (mg/dl)",
    "fbs":      "Fasting blood sugar > 120 mg/dl (1=True)",
    "restecg":  "Resting ECG results (0-2)",
    "thalach":  "Max heart rate achieved",
    "exang":    "Exercise induced angina (1=Yes)",
    "oldpeak":  "ST depression induced by exercise",
    "slope":    "Slope of peak exercise ST segment (0-2)",
    "ca":       "Number of major vessels (0-4)",
    "thal":     "Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)",
    "target":   "Heart disease (1=Yes, 0=No)"
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[EDA] Generating plots...")
sns.set_style("whitegrid")

# ── 2a. Target distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['target'].value_counts()
bars = ax.bar(['No Disease (0)', 'Heart Disease (1)'],
              counts.values,
              color=['#457b9d', '#e63946'], edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{count}\n({count/len(df)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Count')
ax.set_ylim(0, max(counts.values) * 1.2)
plt.tight_layout()
plt.savefig("outputs/plots/01_target_distribution.png", dpi=150)
plt.close()

# ── 2b. Age distribution by target ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, color, name in [(1,'#e63946','Heart Disease'), (0,'#457b9d','No Disease')]:
    subset = df[df['target'] == label]['age']
    ax.hist(subset, bins=20, alpha=0.7, color=color, label=name, edgecolor='white')
ax.set_title('Age Distribution by Target', fontsize=13, fontweight='bold')
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig("outputs/plots/02_age_distribution.png", dpi=150)
plt.close()

# ── 2c. Correlation heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5,
            annot_kws={"size": 8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("outputs/plots/03_correlation_heatmap.png", dpi=150)
plt.close()

# ── 2d. Feature distributions (grid) ──────────────────────────────────────────
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    for label, color in [(1,'#e63946'), (0,'#457b9d')]:
        axes[i].hist(df[df['target']==label][col], bins=20,
                     alpha=0.6, color=color, edgecolor='white')
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel(col)
axes[-1].axis('off')
fig.suptitle('Numerical Feature Distributions by Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/plots/04_feature_distributions.png", dpi=150)
plt.close()

# ── 2e. Categorical feature vs target ─────────────────────────────────────────
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df['target'])
    ct.plot(kind='bar', ax=axes[i], color=['#457b9d','#e63946'],
            edgecolor='white', legend=(i==0))
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=0)
fig.suptitle('Categorical Features vs Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/plots/05_categorical_features.png", dpi=150)
plt.close()

print("[EDA] All EDA plots saved to outputs/plots/")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[SPLIT] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, "outputs/models/scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING  &  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
results = {}

def evaluate(name, model, X_tr, X_te, y_tr, y_te, use_proba=True):
    """Fit model and collect metrics."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1] if use_proba else y_pred.astype(float)

    metrics = {
        "Accuracy":  round(accuracy_score(y_te, y_pred) * 100, 2),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_te, y_pred, zero_division=0) * 100, 2),
        "F1-Score":  round(f1_score(y_te, y_pred, zero_division=0) * 100, 2),
        "ROC-AUC":   round(roc_auc_score(y_te, y_prob) * 100, 2),
    }
    results[name] = metrics
    print(f"\n[{name}] Results:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.2f}%")
    print(classification_report(y_te, y_pred, target_names=["No Disease","Heart Disease"]))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=["No Disease","Heart Disease"],
                yticklabels=["No Disease","Heart Disease"])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{name} – Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"outputs/plots/cm_{name.replace(' ','_')}.png", dpi=150)
    plt.close()

    return model, y_prob

# ══════════════════════════════════════════════════════════════════════════════
# 4a. Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-"*50)
print("  MODEL 1: Logistic Regression")
print("-"*50)
lr_params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
lr_gs = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                     lr_params, cv=5, scoring='accuracy', n_jobs=-1)
lr_model, lr_prob = evaluate("Logistic Regression", lr_gs,
                              X_train_sc, X_test_sc, y_train, y_test)
print(f"  Best params: {lr_gs.best_params_}")
joblib.dump(lr_gs.best_estimator_, "outputs/models/logistic_regression.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 4b. Decision Tree
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-"*50)
print("  MODEL 2: Decision Tree")
print("-"*50)
dt_params = {'max_depth': [3, 5, 7, 10, None],
             'min_samples_split': [2, 5, 10],
             'criterion': ['gini', 'entropy']}
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                     dt_params, cv=5, scoring='accuracy', n_jobs=-1)
dt_model, dt_prob = evaluate("Decision Tree", dt_gs,
                              X_train_sc, X_test_sc, y_train, y_test)
print(f"  Best params: {dt_gs.best_params_}")
joblib.dump(dt_gs.best_estimator_, "outputs/models/decision_tree.pkl")

# Feature importance – Decision Tree
fi_dt = pd.Series(dt_gs.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 5))
fi_dt.plot(kind='barh', ax=ax, color='#457b9d', edgecolor='white')
ax.set_title('Decision Tree – Feature Importances', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/plots/fi_decision_tree.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 4c. Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-"*50)
print("  MODEL 3: Random Forest")
print("-"*50)
rf_params = {'n_estimators': [100, 200],
             'max_depth': [5, 10, None],
             'min_samples_split': [2, 5]}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42),
                     rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_model, rf_prob = evaluate("Random Forest", rf_gs,
                              X_train_sc, X_test_sc, y_train, y_test)
print(f"  Best params: {rf_gs.best_params_}")
joblib.dump(rf_gs.best_estimator_, "outputs/models/random_forest.pkl")

# Feature importance – Random Forest
fi_rf = pd.Series(rf_gs.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 5))
fi_rf.plot(kind='barh', ax=ax, color='#e63946', edgecolor='white')
ax.set_title('Random Forest – Feature Importances', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/plots/fi_random_forest.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 4d. Neural Network (Feedforward)
# ══════════════════════════════════════════════════════════════════════════════
nn_prob = None
if HAS_TF:
    print("\n" + "-"*50)
    print("  MODEL 4: Neural Network (Feedforward)")
    print("-"*50)

    nn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_sc.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer=Adam(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy'])
    nn.summary()

    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = nn.fit(X_train_sc, y_train,
                     epochs=150, batch_size=32,
                     validation_split=0.15,
                     callbacks=[es],
                     verbose=0)

    nn_prob = nn.predict(X_test_sc).flatten()
    y_pred_nn = (nn_prob >= 0.5).astype(int)
    nn_metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred_nn) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred_nn, zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred_nn, zero_division=0) * 100, 2),
        "F1-Score":  round(f1_score(y_test, y_pred_nn, zero_division=0) * 100, 2),
        "ROC-AUC":   round(roc_auc_score(y_test, nn_prob) * 100, 2),
    }
    results["Neural Network"] = nn_metrics
    print(f"\n[Neural Network] Results:")
    for k, v in nn_metrics.items():
        print(f"  {k:12s}: {v:.2f}%")
    print(classification_report(y_test, y_pred_nn, target_names=["No Disease", "Heart Disease"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_nn)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=["No Disease","Heart Disease"],
                yticklabels=["No Disease","Heart Disease"])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Neural Network – Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/plots/cm_Neural_Network.png", dpi=150)
    plt.close()

    # Training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train', color='#457b9d')
    ax1.plot(history.history['val_accuracy'], label='Validation', color='#e63946')
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Train', color='#457b9d')
    ax2.plot(history.history['val_loss'], label='Validation', color='#e63946')
    ax2.set_title('Model Loss', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/nn_training_history.png", dpi=150)
    plt.close()

    nn.save("outputs/models/neural_network.keras")
    print("[NN] Model saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARATIVE PLOTS
# ══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ['Model'] + list(results_df.columns[1:])
results_df.to_csv("outputs/results/model_comparison.csv", index=False)

print("\n[RESULTS] Model Comparison:")
print(results_df.to_string(index=False))

# ── Metric comparison bar chart ────────────────────────────────────────────────
metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
colors = ['#457b9d', '#2a9d8f', '#e9c46a', '#e63946'][:len(results_df)]
for i, metric in enumerate(metrics_list):
    bars = axes[i].bar(results_df['Model'], results_df[metric], color=colors, edgecolor='white')
    axes[i].set_title(metric, fontweight='bold', fontsize=11)
    axes[i].set_ylim(0, 110)
    axes[i].set_ylabel('%')
    axes[i].tick_params(axis='x', rotation=25)
    for bar, val in zip(bars, results_df[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("outputs/plots/06_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ── ROC Curves ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
roc_models = [
    ("Logistic Regression", lr_prob,  '#457b9d'),
    ("Decision Tree",       dt_prob,  '#2a9d8f'),
    ("Random Forest",       rf_prob,  '#e9c46a'),
]
if nn_prob is not None:
    roc_models.append(("Neural Network", nn_prob, '#e63946'))

for name, prob, color in roc_models:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, lw=2)

ax.plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves – All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/plots/07_roc_curves.png", dpi=150)
plt.close()

# ── Best model feature importance (Random Forest) ─────────────────────────────
fi_full = pd.Series(rf_gs.best_estimator_.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors_fi = ['#e63946' if v > fi_full.median() else '#457b9d' for v in fi_full]
fi_full.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_title('Feature Importance (Random Forest)\nRed = Above Median', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig("outputs/plots/08_feature_importance_rf.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 6. CROSS-VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[CV] 5-Fold Cross-Validation (Accuracy %)")
cv_models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, **lr_gs.best_params_, random_state=42), X_train_sc),
    "Decision Tree":       (DecisionTreeClassifier(**dt_gs.best_params_, random_state=42), X_train_sc),
    "Random Forest":       (RandomForestClassifier(**rf_gs.best_params_, random_state=42), X_train_sc),
}
cv_results = {}
for name, (model, Xd) in cv_models.items():
    scores = cross_val_score(model, Xd, y_train, cv=5, scoring='accuracy') * 100
    cv_results[name] = {'Mean': scores.mean().round(2), 'Std': scores.std().round(2)}
    print(f"  {name:22s}: {scores.mean():.2f}% ± {scores.std():.2f}%")

cv_df = pd.DataFrame(cv_results).T.reset_index()
cv_df.columns = ['Model', 'CV Mean Acc (%)', 'CV Std (%)']
cv_df.to_csv("outputs/results/cv_results.csv", index=False)

print("\n" + "="*60)
print("  PIPELINE COMPLETE!")
print("  Outputs in: outputs/plots/  &  outputs/models/  &  outputs/results/")
print("="*60)
