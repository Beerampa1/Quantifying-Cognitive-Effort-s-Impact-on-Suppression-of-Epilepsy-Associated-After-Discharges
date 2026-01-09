# file: gui/route_selector_window.py

import sys
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt


def _safe_import(path: str, name: str):
    """
    Import helper: returns attribute `name` from module `path`,
    or None if anything fails.
    """
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name, None)
    except Exception:
        return None



AppGUI = _safe_import("gui.main_gui", "AppGUI")
FeatureExtractionWindow = _safe_import("gui.feature_extraction_window", "FeatureExtractionWindow")
FODNPostWindow = _safe_import("gui.fodn_post_window", "FODNPostWindow")
DFAPostWindow = _safe_import("gui.dfa_post_window", "DFAPostWindow")
MFDFAOverlapWindow = _safe_import("gui.mfdfa_overlap_window", "MFDFAOverlapWindow")
ModelTrainingWindow = _safe_import("gui.model_training_window", "ModelTrainingWindow")


class RouteSelectorWindow(QDialog):
    """
    Simple launcher dialog. Improvements vs prior version:
      - Centralized safe imports (less repeated try/except)
      - Buttons disable themselves if the route isn't available
      - Keeps references to opened windows to prevent GC closing them
      - Consistent "show + close" flow
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Processing Route")
        self.setMinimumSize(420, 320)

        # Keep refs to opened windows so they stay alive
        self._opened_windows = []

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel("Please choose a processing route:")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        self.btn_current = QPushButton("Process Route")
        self.btn_feature = QPushButton("Feature Extraction Route")
        self.btn_fodn_plots = QPushButton("FODN Plots / Post-proc")
        self.btn_dfa_plots = QPushButton("DFA / MF-DFA Plots")
        self.btn_mfdfa_overlap = QPushButton("MF-DFA Overlapping Analysis")
        self.btn_model_training = QPushButton("Model Training Route")

        for b in (
            self.btn_current, self.btn_feature, self.btn_fodn_plots,
            self.btn_dfa_plots, self.btn_mfdfa_overlap, self.btn_model_training
        ):
            b.setFixedHeight(40)
            layout.addWidget(b)

        # Wire clicks
        self.btn_current.clicked.connect(self.open_current_route)
        self.btn_feature.clicked.connect(self.open_feature_route)
        self.btn_fodn_plots.clicked.connect(self.open_fodn_plots)
        self.btn_dfa_plots.clicked.connect(self.open_dfa_plots)
        self.btn_mfdfa_overlap.clicked.connect(self.open_mfdfa_overlap)
        self.btn_model_training.clicked.connect(self.open_model_training)

        # Enable/disable based on availability
        self._set_available(self.btn_current, AppGUI, "Main processing window (AppGUI) not available.")
        self._set_available(self.btn_feature, FeatureExtractionWindow, "FeatureExtractionWindow not available.")
        self._set_available(self.btn_fodn_plots, FODNPostWindow, "FODNPostWindow not available.")
        self._set_available(self.btn_dfa_plots, DFAPostWindow, "DFAPostWindow not available.")
        # Overlap window usually requires data; keep enabled if class exists
        self._set_available(self.btn_mfdfa_overlap, MFDFAOverlapWindow, "MFDFAOverlapWindow not available.")
        self._set_available(self.btn_model_training, ModelTrainingWindow, "ModelTrainingWindow not available.")

    def _set_available(self, button: QPushButton, cls, missing_tip: str):
        if cls is None:
            button.setEnabled(False)
            button.setToolTip(missing_tip)
        else:
            button.setEnabled(True)
            button.setToolTip("")

    def _open_window(self, win):
        """Show window and keep reference to prevent premature close."""
        self._opened_windows.append(win)
        win.show()
        self.close()

    # ---------- routes ----------
    def open_current_route(self):
        if AppGUI is None:
            QMessageBox.warning(self, "Error", "Process Route window not available.")
            return
        self._open_window(AppGUI())

    def open_feature_route(self):
        if FeatureExtractionWindow is None:
            QMessageBox.warning(self, "Error", "Feature Extraction route window not available.")
            return
        self._open_window(FeatureExtractionWindow())

    def open_fodn_plots(self):
        if FODNPostWindow is None:
            QMessageBox.warning(self, "Error", "FODN plots window not available.")
            return
        self._open_window(FODNPostWindow())

    def open_dfa_plots(self):
        if DFAPostWindow is None:
            QMessageBox.warning(self, "Error", "DFA/MF-DFA plots window not available.")
            return
        self._open_window(DFAPostWindow())

    def open_mfdfa_overlap(self):
        if MFDFAOverlapWindow is None:
            QMessageBox.warning(self, "Error", "Overlapping MF-DFA window not available.")
            return

        # This window typically needs signals/time; guide user to the right place.
        QMessageBox.information(
            self, "How to run",
            "MF-DFA Overlap needs loaded iEEG data.\n\n"
            "Recommended path:\n"
            "1) Open Process Route\n"
            "2) Select a trial and load the H5\n"
            "3) Open Channel Plot Window\n"
            "4) Click 'Open MF-DFA (Overlapping)'"
        )

        # Optionally open main route for them
        if AppGUI is not None:
            self._open_window(AppGUI())

    def open_model_training(self):
        if ModelTrainingWindow is None:
            QMessageBox.warning(self, "Error", "Model training route window not available.")
            return
        self._open_window(ModelTrainingWindow())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RouteSelectorWindow()
    window.show()
    sys.exit(app.exec_())
