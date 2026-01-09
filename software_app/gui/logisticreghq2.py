import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ─────────────────────────── Load data ───────────────────────────
success = pd.read_csv("C:/Users/sbeeram/Capstone 2025/Hurst/Success_hq_extracted.csv", low_memory=False)
failure = pd.read_csv("C:/Users/sbeeram/Capstone 2025/Hurst/Fail_hq_extracted.csv", low_memory=False)

# Keep q == 2 only
success = success[success["q"] == 2]
failure = failure[failure["q"] == 2]

# Mean hq per trial × window
success_g = success.groupby(["trial", "window"])["hq"].mean().reset_index()
failure_g = failure.groupby(["trial", "window"])["hq"].mean().reset_index()
success_g["label"] = 1
failure_g["label"] = 0

df = pd.concat([success_g, failure_g], ignore_index=True)

# Pivot to wide format → each trial has W1 and W2
dfp = df.pivot_table(index=["trial", "label"], columns="window", values="hq").reset_index()
dfp.columns.name = None
dfp = dfp.rename(columns={"W1": "Hurst_W1", "W2": "Hurst_W2"}).dropna(subset=["Hurst_W1", "Hurst_W2"])

# ─────────────────────────── New features (variance & std) ───────────────────────────
dfp["Hurst_var"] = dfp[["Hurst_W1", "Hurst_W2"]].var(axis=1, ddof=0)
dfp["Hurst_std"] = dfp[["Hurst_W1", "Hurst_W2"]].std(axis=1, ddof=0)

# ─────────────────────────── Features / Target ───────────────────────────
X = dfp[["Hurst_W1", "Hurst_W2", "Hurst_var", "Hurst_std"]]
y = dfp["label"].astype(int).values

# ─────────────────────────── Leave-One-Out Cross Validation ───────────────────────────
loo = LeaveOneOut()
y_true, y_prob, y_pred = [], [], []

for train_idx, test_idx in loo.split(X):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    p = model.predict_proba(X.iloc[test_idx])[:, 1][0]
    y_prob.append(p)
    y_pred.append(int(p >= 0.5))
    y_true.append(int(y[test_idx][0]))

# ─────────────────────────── Evaluation ───────────────────────────
auc = roc_auc_score(y_true, y_prob)
acc = accuracy_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=3)

print(dfp.head())
print(f"\nLogistic Regression (LOOCV) results (n = {len(y_true)} trials)")
print("Accuracy:", round(acc, 3))
print("ROC-AUC :", round(auc, 3))
print("Confusion matrix [TN FP; FN TP]:\n", cm)
print("\nClassification report:\n", report)

# ─────────────────────────── Per-fold metrics for boxplots ───────────────────────────
acc_per_fold  = [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]
prec_per_fold = [precision_score([yt], [yp], zero_division=0) for yt, yp in zip(y_true, y_pred)]
rec_per_fold  = [recall_score([yt], [yp], zero_division=0)    for yt, yp in zip(y_true, y_pred)]

print(f"\nPer-fold means → "
      f"Accuracy: {np.mean(acc_per_fold):.3f}, "
      f"Precision: {np.mean(prec_per_fold):.3f}, "
      f"Recall: {np.mean(rec_per_fold):.3f}")

# ─────────────────────────── Combined Boxplot (all metrics together) ───────────────────────────
plt.figure(figsize=(7,6))
plt.boxplot(
    [acc_per_fold, prec_per_fold, rec_per_fold],
    vert=True,
    tick_labels=["Accuracy", "Precision", "Recall"]
)
plt.title("Model Performance Metrics (LOOCV per-fold)")
plt.ylabel("Score")
plt.ylim(-0.05, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ─────────────────────────── Confusion Matrix (cleaner plot) ───────────────────────────
plt.figure(figsize=(5,4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail (0)", "Success (1)"])
disp.plot(values_format="d", cmap="Greens", colorbar=True)
plt.title("Confusion Matrix (Logistic Regression LOOCV)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import LeaveOneOut
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import (
#     classification_report, confusion_matrix,
#     roc_auc_score, accuracy_score
# )
# import matplotlib.pyplot as plt

# # ─────────────────────────── Load data ───────────────────────────
# success = pd.read_csv("C:/Users/sbeeram/Capstone 2025/Hurst/Success_hq_extracted.csv", low_memory=False)
# failure = pd.read_csv("C:/Users/sbeeram/Capstone 2025/Hurst/Fail_hq_extracted.csv", low_memory=False)

# # Keep q == 2 only
# success = success[success["q"] == 2]
# failure = failure[failure["q"] == 2]

# # Mean hq per trial × window
# success_g = success.groupby(["trial", "window"])["hq"].mean().reset_index()
# failure_g = failure.groupby(["trial", "window"])["hq"].mean().reset_index()
# success_g["label"] = 1
# failure_g["label"] = 0

# df = pd.concat([success_g, failure_g], ignore_index=True)

# # Pivot to wide format
# dfp = df.pivot_table(index=["trial", "label"], columns="window", values="hq").reset_index()
# dfp.columns.name = None
# dfp = dfp.rename(columns={"W1": "Hurst_W1", "W2": "Hurst_W2"}).dropna(subset=["Hurst_W1", "Hurst_W2"])

# # ─────────────────────────── Features / Target ───────────────────────────
# X = dfp[["Hurst_W1", "Hurst_W2"]]  # ← removed dHurst
# y = dfp["label"].astype(int).values

# # ─────────────────────────── Leave-One-Out Cross Validation ───────────────────────────
# loo = LeaveOneOut()
# y_true, y_prob, y_pred = [], [], []

# for train_idx, test_idx in loo.split(X):
#     model = make_pipeline(
#         StandardScaler(),
#         LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
#     )
#     model.fit(X.iloc[train_idx], y[train_idx])
#     p = model.predict_proba(X.iloc[test_idx])[:, 1][0]
#     y_prob.append(p)
#     y_pred.append(int(p >= 0.5))
#     y_true.append(int(y[test_idx][0]))

# # ─────────────────────────── Evaluation ───────────────────────────
# auc = roc_auc_score(y_true, y_prob)
# acc = accuracy_score(y_true, y_pred)
# cm  = confusion_matrix(y_true, y_pred)
# report = classification_report(y_true, y_pred, digits=3)

# print(dfp.head())
# print(f"\nLogistic Regression (LOOCV) results (n = {len(y_true)} trials)")
# print("Accuracy:", round(acc, 3))
# print("ROC-AUC :", round(auc, 3))
# print("Confusion matrix [TN FP; FN TP]:\n", cm)
# print("\nClassification report:\n", report)

# # ─────────────────────────── Confusion Matrix Plot ───────────────────────────
# plt.figure(figsize=(5, 4))
# plt.imshow(cm, interpolation="nearest", cmap="Greens")
# plt.title("Confusion Matrix (Logistic Regression LOOCV)")
# plt.colorbar()
# classes = ["Fail (0)", "Success (1)"]
# plt.xticks(np.arange(2), classes)
# plt.yticks(np.arange(2), classes)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")

# # annotate counts
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

# plt.tight_layout()
# plt.show()




# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# # Load both datasets
# success = pd.read_csv("C:/Users/sbeeram/Capstone 2025/Hurst/Success_hq_extracted.csv")
# failure = pd.read_csv("C:/Users/sbeeram/Capstone 2025/Hurst/Fail_hq_extracted.csv")

# # Filter only q == 2
# success = success[success["q"] == 2]
# failure = failure[failure["q"] == 2]

# # Compute mean hq per trial
# success_grouped = success.groupby(["trial", "window"])["hq"].mean().reset_index()
# failure_grouped = failure.groupby(["trial", "window"])["hq"].mean().reset_index()

# # Add labels
# success_grouped["label"] = 1
# failure_grouped["label"] = 0

# # Combine
# df = pd.concat([success_grouped, failure_grouped], ignore_index=True)

# # Pivot so each trial has W1 and W2 as features
# df_pivot = df.pivot_table(index=["trial", "label"], columns="window", values="hq").reset_index()

# # Rename columns
# df_pivot.columns.name = None
# df_pivot.rename(columns={"W1": "Hurst_W1", "W2": "Hurst_W2"}, inplace=True)

# # Drop missing W1/W2 values if any
# df_pivot = df_pivot.dropna(subset=["Hurst_W1", "Hurst_W2"])

# print(df_pivot.head())

# # Train-test split
# X = df_pivot[["Hurst_W1", "Hurst_W2"]]
# y = df_pivot["label"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# # Logistic Regression pipeline
# model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]

# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# # Cross-validation for robustness
# cv_score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
# print("Cross-validated ROC-AUC:", cv_score)
