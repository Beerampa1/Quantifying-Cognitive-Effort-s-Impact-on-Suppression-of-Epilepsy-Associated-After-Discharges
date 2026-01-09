# file: gui/route_selector_window.py

import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import Qt

try:
    from gui.main_gui import AppGUI  #existing main window
except ImportError:
    AppGUI = None

try:
    from gui.feature_extraction_window import FeatureExtractionWindow  # New window for feature extraction
except ImportError:
    FeatureExtractionWindow = None

try:
    from gui.fodn_post_window import FODNPostWindow  # New window for FODN plots
except ImportError:
    FODNPostWindow = None

try:
    from gui.dfa_post_window import DFAPostWindow  # New window for DFA plots
except ImportError:
    DFAPostWindow = None

try:
    from gui.mfdfa_overlap_window import MFDFAOverlapWindow
except ImportError:
    MFDFAOverlapWindow = None



try:
    from gui.model_training_window import ModelTrainingWindow  # New window for model training
except ImportError:
    ModelTrainingWindow = None

class RouteSelectorWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Processing Route")
        self.setMinimumSize(400, 300)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        info_label = QLabel("Please choose a processing route:")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        self.btn_current = QPushButton("Process Route")
        self.btn_feature = QPushButton("Feature Extraction Route")
        self.btn_fodn_plots     = QPushButton("FODN Plots / Post‑proc")
        self.btn_dfa_plots      = QPushButton("DFA / MFDFA Plots")
        self.btn_mfdfa_overlap = QPushButton("MF DFA Overlapping Analysis")  # NEW
        self.btn_model_training = QPushButton("Model Training Route")
        
        self.btn_current.setFixedHeight(40)
        self.btn_feature.setFixedHeight(40)
        self.btn_fodn_plots.setFixedHeight(40)
        self.btn_dfa_plots.setFixedHeight(40)
        self.btn_mfdfa_overlap.setFixedHeight(40)  
        self.btn_model_training.setFixedHeight(40)
        
        layout.addWidget(self.btn_current)
        layout.addWidget(self.btn_feature)
        layout.addWidget(self.btn_fodn_plots)
        layout.addWidget(self.btn_dfa_plots)
        layout.addWidget(self.btn_mfdfa_overlap)
        layout.addWidget(self.btn_model_training)
        
        self.btn_current.clicked.connect(self.open_current_route)
        self.btn_feature.clicked.connect(self.open_feature_route)
        self.btn_fodn_plots.clicked.connect(self.open_fodn_plots)
        self.btn_dfa_plots.clicked.connect(self.open_dfa_plots)
        self.btn_mfdfa_overlap.clicked.connect(self.open_mfdfa_overlap)
        self.btn_model_training.clicked.connect(self.open_model_training)
    
    def open_current_route(self):
        if AppGUI is not None:
            self.current_route = AppGUI()
            self.current_route.show()
            self.close()
        else:
            QMessageBox.warning(self, "Error", "Current route window not available.")
    
    def open_feature_route(self):
        if FeatureExtractionWindow is not None:
            self.batch_route = FeatureExtractionWindow()
            self.batch_route.show()
            self.close()
        else:
            QMessageBox.warning(self, "Error", "Feature Extraction route window not available.")
    
    def open_fodn_plots(self):
        self.fodn_win = FODNPostWindow(); self.fodn_win.show(); self.close()

    def open_dfa_plots(self):
        self.dfa_win = DFAPostWindow(); self.dfa_win.show(); self.close()
    
    def open_mfdfa_overlap(self):
      if MFDFAOverlapWindow is not None:
        # The overlap window needs data; typically you open it from Channel Plot.
        QMessageBox.information(
            self, "How to run",
            "Open the Process Route → select a trial → Open Channel Plot Window → "
            "click 'Open MF-DFA (Overlapping)'."
        )
        # Optionally open the main route to get them there quicker:
        if AppGUI is not None:
            gui = AppGUI()
            gui.show()
            self.close()
        return
      QMessageBox.warning(self, "Error", "Overlapping MF-DFA window not available.")


    def open_model_training(self):
        if ModelTrainingWindow is not None:
            self.model_training_route = ModelTrainingWindow()
            self.model_training_route.show()
            self.close()
        else:
            QMessageBox.warning(self, "Error", "Model training route window not available.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RouteSelectorWindow()
    window.show()
    sys.exit(app.exec_())
