# gui/model_training_window.py
import csv, io, os
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QTextEdit, QMessageBox
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt


class ModelTrainingWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Training (Logistic Regression)")
        self.resize(700, 600)
        self._build_ui()

        self.df: pd.DataFrame | None = None  # loaded data
        self.pipeline = None                 # trained model

    # ────────────────────────────────────────────────────────── UI BUILD ──
    def _build_ui(self):
        main = QVBoxLayout(self)

        # File picker
        pick = QHBoxLayout()
        main.addLayout(pick)
        pick.addWidget(QLabel("Features CSV:"))
        self.le_path = QLineEdit(); self.le_path.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._choose_csv)
        pick.addWidget(self.le_path); pick.addWidget(btn_browse)

        # Settings row
        row = QHBoxLayout(); main.addLayout(row)
        row.addWidget(QLabel("Label column:"))
        self.cb_label = QComboBox()
        row.addWidget(self.cb_label)

        row.addWidget(QLabel("Test size (%):"))
        self.sb_split = QSpinBox(); self.sb_split.setRange(5, 50); self.sb_split.setValue(20)
        row.addWidget(self.sb_split)

        self.cb_scale = QCheckBox("Standardize features"); self.cb_scale.setChecked(True)
        row.addWidget(self.cb_scale)

        self.cb_dropna = QCheckBox("Drop rows with NaNs"); self.cb_dropna.setChecked(True)
        row.addWidget(self.cb_dropna)

        row.addStretch(1)
        self.btn_train = QPushButton("Train model"); self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self._train)
        row.addWidget(self.btn_train)

        # Text log
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        main.addWidget(self.txt_log, stretch=1)

    # ─────────────────────────────────────────────────────────── HELPERS ──
    def _choose_csv(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select master_features.csv", filter="CSV (*.csv)")
        if not f: return
        self.le_path.setText(f)
        try:
            self.df = pd.read_csv(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}")
            self.btn_train.setEnabled(False)
            return

        # populate label combo
        self.cb_label.clear()
        self.cb_label.addItems(self.df.columns.tolist())
        # try to select "Label"
        idx = self.cb_label.findText("Label")
        if idx != -1:
            self.cb_label.setCurrentIndex(idx)
        self.btn_train.setEnabled(True)
        self.txt_log.append(f"Loaded {len(self.df)} rows × {len(self.df.columns)} columns.")

    def _train(self):
        if self.df is None: return
        y_col = self.cb_label.currentText()
        if y_col == "":
            QMessageBox.warning(self, "No label", "Please choose a label column."); return

        df = self.df.copy()

        # drop rows where y is NaN or non‑numeric
        df = df[pd.to_numeric(df[y_col], errors="coerce").notnull()]
        df[y_col] = df[y_col].astype(float)

        # split
        test_pct = self.sb_split.value() / 100.0
        df = df.sample(frac=1, random_state=42)  # shuffle
        split = int(len(df) * (1 - test_pct))
        train_df, test_df = df.iloc[:split], df.iloc[split:]
        X_train, y_train = train_df.drop(columns=y_col), train_df[y_col]
        X_test,  y_test  = test_df.drop(columns=y_col),  test_df[y_col]

        # numeric-only (drop any non-numeric cols)
        X_train = X_train.select_dtypes(include=[np.number])
        X_test  = X_test [X_train.columns]  # ensure same columns

        # handle NaNs
        if self.cb_dropna.isChecked():
            mask = X_train.notna().all(axis=1)
            X_train, y_train = X_train[mask], y_train[mask]
            mask = X_test.notna().all(axis=1)
            X_test, y_test = X_test[mask], y_test[mask]

        # pipeline
        steps = [("imputer", SimpleImputer(strategy="mean"))]
        if self.cb_scale.isChecked():
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", LogisticRegression(max_iter=1000)))
        self.pipeline = make_pipeline(*[s[1] for s in steps])
        self.pipeline.fit(X_train, y_train)

        # metrics
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(y_test, y_pred)

        # log
        buf = io.StringIO()
        print("=== Train/Test split:", f"{len(train_df)}/{len(test_df)}", file=buf)
        print(f"Accuracy: {acc:.3f}", file=buf)
        print(f"ROC‑AUC:  {auc:.3f}", file=buf)
        print("Confusion matrix [TN FP; FN TP]:", file=buf)
        print(cm, file=buf)

        # top coefficients
        coef = self.pipeline[-1].coef_.flatten()
        feat_names = X_train.columns
        top_idx = np.argsort(np.abs(coef))[::-1][:10]
        print("\nTop 10 features by |coef|:", file=buf)
        for idx in top_idx:
            print(f"{feat_names[idx]:30s}  {coef[idx]: .4f}", file=buf)

        self.txt_log.append(buf.getvalue())

        # plot ROC curve
        plt.figure()
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC curve")
        plt.tight_layout()
        plt.show()

        # plot confusion matrix heat‑map
        plt.figure()
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title("Confusion matrix")
        plt.colorbar()
        plt.xticks([0,1], ["Pred 0","Pred 1"])
        plt.yticks([0,1], ["True 0","True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i,j], ha="center", va="center")
        plt.tight_layout()
        plt.show()


# quick manual test
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = ModelTrainingWindow()
    w.show()
    sys.exit(app.exec_())
