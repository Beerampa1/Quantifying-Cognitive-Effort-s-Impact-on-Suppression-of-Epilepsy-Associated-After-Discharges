# gui/model_training_window.py
import io
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QTextEdit, QMessageBox
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

import matplotlib.pyplot as plt


class ModelTrainingWindow(QDialog):
    """
    GUI for training a simple Logistic Regression classifier on a CSV of features.

    Improvements vs prior version:
      - Robust label handling (binary labels only; clear error messages)
      - Numeric-only features with consistent column alignment
      - Optional row-drop for NaNs OR mean-impute (imputer always present)
      - Stratified train/test split for class balance stability
      - Safer AUC computation (only if both classes present)
      - Coefficient reporting guarded for feature alignment
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Training (Logistic Regression)")
        self.resize(700, 600)
        self._build_ui()

        self.df: pd.DataFrame | None = None  # loaded data
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []  # keep the feature columns used to train

    # ────────────────────────────────────────────────────────── UI BUILD ──
    def _build_ui(self):
        main = QVBoxLayout(self)

        # File picker
        pick = QHBoxLayout()
        main.addLayout(pick)
        pick.addWidget(QLabel("Features CSV:"))
        self.le_path = QLineEdit()
        self.le_path.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._choose_csv)
        pick.addWidget(self.le_path)
        pick.addWidget(btn_browse)

        # Settings row
        row = QHBoxLayout()
        main.addLayout(row)

        row.addWidget(QLabel("Label column:"))
        self.cb_label = QComboBox()
        row.addWidget(self.cb_label)

        row.addWidget(QLabel("Test size (%):"))
        self.sb_split = QSpinBox()
        self.sb_split.setRange(5, 50)
        self.sb_split.setValue(20)
        row.addWidget(self.sb_split)

        self.cb_scale = QCheckBox("Standardize features")
        self.cb_scale.setChecked(True)
        row.addWidget(self.cb_scale)

        self.cb_dropna = QCheckBox("Drop rows with NaNs")
        self.cb_dropna.setChecked(True)
        row.addWidget(self.cb_dropna)

        row.addStretch(1)

        self.btn_train = QPushButton("Train model")
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self._train)
        row.addWidget(self.btn_train)

        # Text log
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        main.addWidget(self.txt_log, stretch=1)

    # ─────────────────────────────────────────────────────────── HELPERS ──
    def _choose_csv(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select master_features.csv", filter="CSV (*.csv)"
        )
        if not f:
            return

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
        self.txt_log.append("Choose label column and click Train.\n")

    def _train(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No data", "Load a CSV first.")
            return

        y_col = self.cb_label.currentText().strip()
        if not y_col or y_col not in self.df.columns:
            QMessageBox.warning(self, "No label", "Please choose a valid label column.")
            return

        df = self.df.copy()

        # ---- label parsing: require binary label (0/1) after coercion ----
        y = pd.to_numeric(df[y_col], errors="coerce")
        df = df[y.notna()].copy()
        y = y.loc[df.index].astype(int)

        unique = sorted(pd.unique(y))
        if len(unique) < 2:
            QMessageBox.warning(self, "Bad label", "Label column has only one class after cleaning.")
            return
        if len(unique) > 2:
            QMessageBox.warning(
                self, "Bad label",
                f"Label must be binary. Found classes: {unique}"
            )
            return

        # Map to {0,1} if needed (e.g., {1,2} or {-1,1})
        if unique != [0, 1]:
            mapping = {unique[0]: 0, unique[1]: 1}
            y = y.map(mapping).astype(int)

        # ---- features: numeric-only and aligned ----
        X = df.drop(columns=[y_col])
        X = X.select_dtypes(include=[np.number]).copy()

        if X.shape[1] == 0:
            QMessageBox.warning(self, "No features", "No numeric feature columns found.")
            return

        # ---- dropna option (pre-imputer) ----
        if self.cb_dropna.isChecked():
            mask = X.notna().all(axis=1)
            X = X.loc[mask]
            y = y.loc[mask]

        if len(X) < 10:
            QMessageBox.warning(self, "Too little data", "Not enough rows after cleaning.")
            return

        test_size = self.sb_split.value() / 100.0

        # Stratified split for stability
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                shuffle=True,
                stratify=y
            )
        except ValueError as e:
            QMessageBox.warning(self, "Split error", f"Could not split data:\n{e}")
            return

        # ---- build pipeline ----
        steps = [("imputer", SimpleImputer(strategy="mean"))]
        if self.cb_scale.isChecked():
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", LogisticRegression(max_iter=2000, solver="lbfgs")))

        self.pipeline = Pipeline(steps)
        self.feature_columns = list(X_train.columns)

        # ---- train ----
        try:
            self.pipeline.fit(X_train, y_train)
        except Exception as e:
            QMessageBox.critical(self, "Training failed", f"Model training failed:\n{e}")
            return

        # ---- evaluate ----
        y_pred = self.pipeline.predict(X_test)

        # probabilities (only for binary clf)
        try:
            y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)

        # AUC requires both classes in y_test and probability scores
        if y_prob is not None and len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = float("nan")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        # ---- log ----
        buf = io.StringIO()
        print("=== Train/Test split:", f"{len(X_train)}/{len(X_test)}", file=buf)
        print("Features used:", X_train.shape[1], file=buf)
        print(f"Accuracy: {acc:.3f}", file=buf)
        print(f"ROC-AUC:  {auc:.3f}", file=buf)
        print("Confusion matrix [TN FP; FN TP]:", file=buf)
        print(cm, file=buf)

        # top coefficients (only if linear model has coef_)
        try:
            clf = self.pipeline.named_steps["clf"]
            coef = np.asarray(clf.coef_).reshape(-1)
            top_idx = np.argsort(np.abs(coef))[::-1][:10]
            print("\nTop 10 features by |coef|:", file=buf)
            for idx in top_idx:
                print(f"{self.feature_columns[idx]:30s}  {coef[idx]: .4f}", file=buf)
        except Exception:
            print("\n(Feature coefficient report unavailable)", file=buf)

        self.txt_log.append(buf.getvalue())

        # ---- plots ----
        # ROC curve
        if y_prob is not None and len(np.unique(y_test)) == 2:
            plt.figure()
            RocCurveDisplay.from_predictions(y_test, y_prob)
            plt.title("ROC curve")
            plt.tight_layout()
            plt.show()

        # Confusion matrix
        plt.figure()
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title("Confusion matrix")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center")
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
