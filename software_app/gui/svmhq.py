import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────── Load data ──
success = pd.read_csv(
    "C:/Users/sbeeram/Capstone 2025/Hurst/Success_hq_extracted.csv",
    low_memory=False
)
failure = pd.read_csv(
    "C:/Users/sbeeram/Capstone 2025/Hurst/Fail_hq_extracted.csv",
    low_memory=False
)

# Keep q == 2 only
success = success[success["q"] == 2]
failure = failure[failure["q"] == 2]

# Mean hq per trial × window
success_g = success.groupby(["trial", "window"])["hq"].mean().reset_index()
failure_g = failure.groupby(["trial", "window"])["hq"].mean().reset_index()
success_g["label"] = 1
failure_g["label"] = 0

df = pd.concat([success_g, failure_g], ignore_index=True)

# Pivot: one row per trial
dfp = df.pivot_table(index=["trial", "label"], columns="window", values="hq").reset_index()
dfp.columns.name = None
dfp = dfp.rename(columns={"W1": "Hurst_W1", "W2": "Hurst_W2"}).dropna(subset=["Hurst_W1", "Hurst_W2"])

# ─────────────────────────────────────────────────────────── Features ──
X = dfp[["Hurst_W1", "Hurst_W2"]]   # ← removed dHurst
y = dfp["label"].astype(int).values

# ─────────────────────────────────────────────────────────────── LOOCV ──
loo = LeaveOneOut()
y_true, y_prob, y_pred = [], [], []

for train_idx, test_idx in loo.split(X):
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True)
    )

    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1][0]
    y_prob.append(p)
    y_pred.append(int(p >= 0.5))
    y_true.append(int(yte[0]))

# ─────────────────────────────────────────────────────────── Evaluation ──
auc = roc_auc_score(y_true, y_prob)
acc = accuracy_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=3)

print(dfp.head())
print(f"\nSVM-RBF LOOCV results (n = {len(y_true)} trials)")
print("Accuracy:", round(acc, 3))
print("ROC-AUC :", round(auc, 3))
print("Confusion matrix [TN FP; FN TP]:\n", cm)
print("\nClassification report:\n", report)

# ───────────────────────────────────────────────────── Confusion-matrix plot ──
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix (SVM-RBF LOOCV)")
plt.colorbar()
classes = ["Fail (0)", "Success (1)"]
plt.xticks(np.arange(2), classes)
plt.yticks(np.arange(2), classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# annotate counts
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

plt.tight_layout()
plt.show()
