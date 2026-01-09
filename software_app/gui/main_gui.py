# file: gui/main_gui.py
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox, 
    QPlainTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt
from utils.file_utils import load_excel_file, load_h5_file
from gui.channel_plot_window import ChannelPlotWindow
import pandas as pd

class AppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seizure Suppression Analysis App")
        self.setWindowState(Qt.WindowMaximized)
        
        self.excel_filepath = None
        self.h5_filepath = None
        self.df_events = None
        self.selected_trial = None
        self.signals = None
        self.time_array = None
        self.channel_names = None
        
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        grid = QGridLayout(central_widget)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(10)
        
        # Row 0: Excel file label + button
        lbl_excel = QLabel("Excel File:")
        self.excel_label = QLabel("No Excel file selected")
        self.excel_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        btn_excel = QPushButton("Browse Excel File")
        btn_excel.clicked.connect(self.choose_excel_file)
        
        grid.addWidget(lbl_excel, 0, 0)
        grid.addWidget(self.excel_label, 0, 1)
        grid.addWidget(btn_excel, 0, 2)
        
        lbl_h5 = QLabel("H5 File:")
        self.h5_label = QLabel("No H5 file selected")
        self.h5_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        btn_h5 = QPushButton("Browse H5 File")
        btn_h5.clicked.connect(self.choose_h5_file)
        
        grid.addWidget(lbl_h5, 1, 0)
        grid.addWidget(self.h5_label, 1, 1)
        grid.addWidget(btn_h5, 1, 2)
        
        lbl_trial = QLabel("Select a Patient Trial:")
        grid.addWidget(lbl_trial, 2, 0, 1, 3)
        
        self.trial_list = QListWidget()
        self.trial_list.itemSelectionChanged.connect(self.trial_selected)
        self.trial_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.trial_list, 3, 0, 1, 3)
        
        btn_show_info = QPushButton("Show Selected Trial Info")
        btn_show_info.clicked.connect(self.show_trial_info)
        btn_change_trial = QPushButton("Choose Different Trial")
        btn_change_trial.clicked.connect(self.reset_trial_selection)
        btn_plotwin = QPushButton("Open Channel Plot Window")
        btn_plotwin.clicked.connect(self.open_plot_window)
        
        hbox_buttons = QHBoxLayout()
        hbox_buttons.addWidget(btn_show_info)
        hbox_buttons.addWidget(btn_change_trial)
        hbox_buttons.addWidget(btn_plotwin)
        
        grid.addLayout(hbox_buttons, 4, 0, 1, 3)
        
        lbl_info = QLabel("Trial Info / Status:")
        grid.addWidget(lbl_info, 5, 0, 1, 3)
        
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.info_text, 6, 0, 1, 3)
    
    def choose_excel_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)"
        )
        if filepath:
            self.excel_filepath = filepath
            self.excel_label.setText(filepath)
            self.df_events = load_excel_file(filepath)
            if self.df_events is not None:
                self.populate_trial_list()
                self.selected_trial = None
                self.h5_filepath = None
                self.h5_label.setText("No H5 file selected")
                self.signals = None
                self.time_array = None
                self.channel_names = None
                self.info_text.clear()
    
    def populate_trial_list(self):
        self.trial_list.clear()
        if self.df_events is None or len(self.df_events) == 0:
            return
        for idx, row in self.df_events.iterrows():
            trial_str = row.get("Patient Session Trial", f"Trial {idx+1}")
            self.trial_list.addItem(str(trial_str))
    
    def trial_selected(self):
        items = self.trial_list.selectedItems()
        if items:
            index = self.trial_list.currentRow()
            self.selected_trial = self.df_events.iloc[index]
            self.info_text.setPlainText(
                f"Selected trial index: {index}\n" +
                f"Selected trial: {str(self.selected_trial.get('Patient Session Trial','Unknown'))}"
            )
    
    def choose_h5_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select H5 File", "", "H5 Files (*.h5)"
        )
        if filepath:
            self.h5_filepath = filepath
            self.h5_label.setText(filepath)
            self.signals, self.time_array, self.channel_names = load_h5_file(filepath)
            if self.signals is None:
                QMessageBox.warning(self, "Error", "Failed to load H5 file.")
    def show_trial_info(self):
        if self.selected_trial is None:
            QMessageBox.warning(self, "No Trial Selected", "Please select a trial from the list.")
            return

        import pandas as pd
        from utils.file_utils import parse_excel_display_to_seconds, fmt_mmss_exact

        row = self.selected_trial
        info_lines = []
        time_cols = ["LS_Start","LS_End","AD_Start","Qes_Start","Qes_End","Ans_Start","Ans_End","AD_End"]

        for key, raw in row.to_dict().items():
            if not isinstance(key, str) or key.startswith("Unnamed"):
                continue

            sraw = "" if raw is None else str(raw).strip()
            if sraw == "" or sraw.upper() in {"NA","N/A","-","--"}:
                info_lines.append(f"{key}: NA")
                continue

            if key in time_cols:
                secs, dp = parse_excel_display_to_seconds(sraw, default_dp=1, within_hour=True)
                if secs is None:
                    info_lines.append(f"{key}: (unparsable)")
                else:
                    pretty = fmt_mmss_exact(secs, dp)   # e.g. '13:15.0'
                    info_lines.append(f"{key}: {pretty}  ({secs:.{dp}f}s)")
            else:
                info_lines.append(f"{key}: {raw}")

        self.info_text.setPlainText("\n".join(info_lines))





    # def show_trial_info(self):
    #     if self.selected_trial is None:
    #         QMessageBox.warning(self, "No Trial Selected", "Please select a trial from the list.")
    #         return

    #     import pandas as pd
    #     from decimal import Decimal, ROUND_HALF_UP
    #     from utils.file_utils import parse_tod  # must return float seconds or None

    #     row = self.selected_trial
    #     info_lines = []

    #     # Columns that contain times
    #     time_cols = ["LS_Start","LS_End","AD_Start","Qes_Start","Qes_End","Ans_Start","Ans_End","AD_End"]

    #     def fmt_mmss(secs, *, seconds_decimals: int = 1) -> str:
    #         """
    #         Format seconds (float) as 'MM:SS.s' (default 1 decimal) with Excel-style ROUND_HALF_UP.
    #         Handles 59.95 -> 60.0 rollover.
    #         Accepts float or numeric string; returns 'NA' for None/NaN.
    #         """
    #         if secs is None or (isinstance(secs, float) and pd.isna(secs)):
    #             return "NA"

    #         # Accept numeric strings defensively
    #         if isinstance(secs, str):
    #             try:
    #                 secs = float(secs)
    #             except ValueError:
    #                 return "NA"

    #         # minutes / seconds split
    #         mm = int(secs // 60)
    #         ss = secs - mm * 60

    #         # Excel-style rounding on seconds field
    #         q = Decimal(10) ** (-seconds_decimals)
    #         ss_dec = Decimal(ss).quantize(q, rounding=ROUND_HALF_UP)

    #         # Rollover (e.g., 59.95 at 1dp -> 60.0)
    #         if ss_dec >= Decimal(60).quantize(q):
    #             mm += 1
    #             ss_dec = Decimal(0).quantize(q)

    #         # Build seconds string with fixed decimals and zero-pad to at least 2 before '.'
    #         if seconds_decimals > 0:
    #             ss_str = f"{ss_dec:.{seconds_decimals}f}"
    #             if ss_dec < Decimal(10):
    #                 ss_str = "0" + ss_str  # e.g., '04.4'
    #         else:
    #             ss_str = f"{int(ss_dec):02d}"

    #         return f"{mm:02d}:{ss_str}"

    #     # Iterate through row
    #     for key, raw in row.to_dict().items():
    #         if not isinstance(key, str) or key.startswith("Unnamed"):
    #             continue
    #         if pd.isna(raw) or raw == "":
    #             info_lines.append(f"{key}: NA")
    #             continue

    #         if key in time_cols:
    #             secs = parse_tod(raw)  # expected: float seconds or None
    #             if secs is not None and not pd.isna(secs):
    #                 # UI string with 1 decimal on seconds (e.g., 20:24.4) + numeric seconds display
    #                 pretty = fmt_mmss(secs, seconds_decimals=1)
    #                 info_lines.append(f"{key}: {pretty}  ({secs:.1f}s)")
    #             else:
    #                 info_lines.append(f"{key}: (unparsable)")
    #         else:
    #             info_lines.append(f"{key}: {raw}")

    #     self.info_text.setPlainText("\n".join(info_lines))

    # def show_trial_info(self):
    #     if self.selected_trial is None:
    #         QMessageBox.warning(self, "No Trial Selected", "Please select a trial from the list.")
    #         return

    #     from utils.file_utils import parse_tod

    #     info_lines = []
    #     row = self.selected_trial
    #     time_cols = ["LS_Start","LS_End","AD_Start","Qes_Start","Qes_End","Ans_Start","Ans_End","AD_End"]

    #     def fmt_mmss(secs: float) -> str:
    #         mm = int(secs // 60)
    #         ss = secs - 60*mm
    #         return f"{mm}:{ss:05.2f}"  # e.g., 34:10.60

    #     for key, raw in row.to_dict().items():
    #         if not isinstance(key, str) or key.startswith("Unnamed") or raw in (None, "", float("nan")):
    #             continue

    #         if key in time_cols:
    #             secs = parse_tod(raw)
    #             if secs is not None:
    #                 info_lines.append(f"{key}: {fmt_mmss(secs)}  ({secs:.2f}s)")
    #             else:
    #                 info_lines.append(f"{key}: (unparsable)")
    #         else:
    #             info_lines.append(f"{key}: {raw}")

    #     self.info_text.setPlainText("\n".join(info_lines))

    # def show_trial_info(self):
    #     """Display a more readable info about the selected trial in the text area."""
    #     if self.selected_trial is None:
    #         QMessageBox.warning(self, "No Trial Selected", "Please select a trial from the list.")
    #         return
        
    #     info_lines = []
    #     row_dict = self.selected_trial.to_dict()
        
    #     for key, val in row_dict.items():
    #         if not isinstance(key, str):
    #             continue
    #         if key.startswith("Unnamed"):
    #             continue
    #         if pd.isna(val):
    #             continue
    #         info_lines.append(f"{key}: {val}")
        
        # self.info_text.setPlainText("\n".join(info_lines))
    
    def reset_trial_selection(self):
        self.trial_list.clearSelection()
        self.selected_trial = None
        self.info_text.clear()
        self.info_text.setPlainText("Trial selection cleared.\nPlease choose a new trial.")
        self.h5_filepath = None
        self.h5_label.setText("No H5 file selected")
        self.signals = None
        self.time_array = None
        self.channel_names = None
    
    def open_plot_window(self):
        if self.signals is None or self.time_array is None:
            QMessageBox.warning(self, "No H5 File", "Please load an iEEG H5 file first.")
            return
        if self.selected_trial is None:
            QMessageBox.warning(self, "No Trial Selected", "Please select a trial first.")
            return
        
        win = ChannelPlotWindow(
            self, 
            self.signals, 
            self.time_array, 
            self.selected_trial, 
            self.channel_names
        )
        win.setWindowState(Qt.WindowMaximized)
        win.exec_()
