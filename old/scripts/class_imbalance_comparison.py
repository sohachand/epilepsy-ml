# ============================================================
# CLASS IMBALANCE TECHNIQUES — COMPARISON
# Add this after your existing PCA + train/test split cells
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier

# ── helper: evaluate any trained model ──────────────────────
def evaluate(name, y_true, y_pred):
    return {
        "Technique"  : name,
        "Precision"  : round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Recall"     : round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1-Score"   : round(f1_score(y_true, y_pred, zero_division=0), 3),
    }

results = []

# ────────────────────────────────────────────────────────────
# 1. BASELINE — Plain RF (no imbalance handling)
# ────────────────────────────────────────────────────────────
print("Running 1/6 — Baseline RF...")
m1 = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
m1.fit(X_train, y_train)
results.append(evaluate("1. Baseline RF", y_test, m1.predict(X_test)))

# ────────────────────────────────────────────────────────────
# 2. SMOTE + RF  (your previous approach)
# ────────────────────────────────────────────────────────────
print("Running 2/6 — SMOTE + RF...")
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
m2 = RandomForestClassifier(n_estimators=200, random_state=42)
m2.fit(X_smote, y_smote)
results.append(evaluate("2. SMOTE + RF", y_test, m2.predict(X_test)))

# ────────────────────────────────────────────────────────────
# 3. class_weight='balanced' RF
#    RF reweights samples internally — no synthetic data
# ────────────────────────────────────────────────────────────
print("Running 3/6 — class_weight='balanced' RF...")
m3 = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)
m3.fit(X_train, y_train)
results.append(evaluate("3. class_weight RF", y_test, m3.predict(X_test)))

# ────────────────────────────────────────────────────────────
# 4. BalancedRandomForestClassifier
#    Undersamples majority in each bootstrap tree — best for
#    heavy imbalance like EEG (83:1 ratio)
# ────────────────────────────────────────────────────────────
print("Running 4/6 — BalancedRandomForest...")
m4 = BalancedRandomForestClassifier(n_estimators=300, random_state=42)
m4.fit(X_train, y_train)
results.append(evaluate("4. BalancedRandomForest", y_test, m4.predict(X_test)))

# ────────────────────────────────────────────────────────────
# 5. SMOTETomek (hybrid)
#    SMOTE oversample minority + Tomek Links remove noisy
#    majority samples near boundary — cleaner than plain SMOTE
# ────────────────────────────────────────────────────────────
print("Running 5/6 — SMOTETomek + RF...")
X_smt, y_smt = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
m5 = RandomForestClassifier(n_estimators=200, random_state=42)
m5.fit(X_smt, y_smt)
results.append(evaluate("5. SMOTETomek + RF", y_test, m5.predict(X_test)))

# ────────────────────────────────────────────────────────────
# 6. BorderlineSMOTE + RF
#    Only synthesizes minority samples near the decision
#    boundary — more targeted, less precision damage
# ────────────────────────────────────────────────────────────
print("Running 6/6 — BorderlineSMOTE + RF...")
X_bs, y_bs = BorderlineSMOTE(random_state=42, kind='borderline-1').fit_resample(X_train, y_train)
m6 = RandomForestClassifier(n_estimators=200, random_state=42)
m6.fit(X_bs, y_bs)
results.append(evaluate("6. BorderlineSMOTE + RF", y_test, m6.predict(X_test)))

# ────────────────────────────────────────────────────────────
# RESULTS TABLE
# ────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
print("\n" + "="*62)
print("       SEIZURE CLASS — IMBALANCE TECHNIQUE COMPARISON")
print("="*62)
print(df.to_string(index=False))
print("="*62)

# ────────────────────────────────────────────────────────────
# THRESHOLD TUNING on best model (class_weight RF)
# Lowers the decision boundary to catch more seizures
# ────────────────────────────────────────────────────────────
print("\n--- Threshold Tuning on class_weight RF (model 3) ---")
y_proba = m3.predict_proba(X_test)[:, 1]
thresh_results = []
for thresh in [0.5, 0.4, 0.3, 0.2, 0.15]:
    y_thresh = (y_proba >= thresh).astype(int)
    thresh_results.append({
        "Threshold" : thresh,
        "Precision" : round(precision_score(y_test, y_thresh, zero_division=0), 3),
        "Recall"    : round(recall_score(y_test, y_thresh, zero_division=0), 3),
        "F1-Score"  : round(f1_score(y_test, y_thresh, zero_division=0), 3),
    })
df_thresh = pd.DataFrame(thresh_results)
print(df_thresh.to_string(index=False))

# ────────────────────────────────────────────────────────────
# VISUALIZATION — Bar chart comparing all techniques
# ────────────────────────────────────────────────────────────
labels     = [r["Technique"] for r in results]
precision  = [r["Precision"]  for r in results]
recall     = [r["Recall"]     for r in results]
f1         = [r["F1-Score"]   for r in results]

x      = np.arange(len(labels))
width  = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, precision, width, label="Precision", color="#4C72B0")
ax.bar(x,         recall,    width, label="Recall",    color="#DD8452")
ax.bar(x + width, f1,        width, label="F1-Score",  color="#55A868")

ax.set_ylabel("Score (Seizure Class)")
ax.set_title("Class Imbalance Techniques — Seizure Detection Performance")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_ylim(0, 1.05)
ax.legend()
ax.axhline(0.75, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='Target')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imbalance_comparison.png", dpi=150)
plt.show()
print("\nChart saved as imbalance_comparison.png")

# ────────────────────────────────────────────────────────────
# DETAILED REPORT for best F1 model
# ────────────────────────────────────────────────────────────
best_idx = df["F1-Score"].idxmax()
best_name = df.loc[best_idx, "Technique"]
print(f"\n{'='*62}")
print(f"  Best technique by F1: {best_name}")
print(f"{'='*62}")

# map back to the right model's predictions
best_preds = {
    "1. Baseline RF"         : m1.predict(X_test),
    "2. SMOTE + RF"          : m2.predict(X_test),
    "3. class_weight RF"     : m3.predict(X_test),
    "4. BalancedRandomForest": m4.predict(X_test),
    "5. SMOTETomek + RF"     : m5.predict(X_test),
    "6. BorderlineSMOTE + RF": m6.predict(X_test),
}
print(classification_report(y_test, best_preds[best_name]))
