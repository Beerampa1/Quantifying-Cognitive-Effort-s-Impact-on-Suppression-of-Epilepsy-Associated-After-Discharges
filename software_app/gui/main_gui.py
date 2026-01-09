# file: gui/main_gui.py
#
# Purpose:
#   Main application window for the Seizure Suppression Analysis App.
#   This GUI lets the user:
#     1) Load an Excel file containing trial/event metadata (times, labels, etc.)
#     2) Choose a specific trial row from the Excel table
#     3) Load an H5 file containing iEEG signals + time array + channel names
#     4) Inspect the selected trial’s metadata in a readable form
#     5) Launch the ChannelPlotWindow to plot selected channels in a chosen time window
#
# Key data objects:
#   - self.df_events: pandas DataFrame loaded from Excel (one row per trial)
#   - self.selected_trial: pandas Series for the currently-selected row from df_events
#   - self.signals: numpy array loaded from H5, shape (n_channels, n_samples)
#   - self.time_array: numpy array loaded from H5, shape (n_samples,)
#   - self.channel_names: list[str] loaded from H5, length n_channels

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
    """
    Main GUI window (QMainWindow).
    Hosts:
      - File selectors (Excel + H5)
      - Trial selector list
      - Trial info/status text area
      - Buttons to open the plot window
    """
    def __init__(self):
        super().__init__()

        # Window configuration
        self.setWindowTitle("Seizure Suppression Analysis App")
        self.setWindowState(Qt.WindowMaximized)

        # App state (populated by UI actions)
        self.excel_filepath = None     # path to Excel metadata file
        self.h5_filepath = None        # path to H5 signal file
        self.df_events = None          # DataFrame from Excel
        self.selected_trial = None     # Series (one row) from df_events
        self.signals = None            # signal matrix from H5
        self.time_array = None         # time stamps from H5
        self.channel_names = None      # list of channel names from H5

        self.initUI()

    def initUI(self):
        """
        Build the main window layout.

        Layout is a grid:
          Row 0: Excel file label + filepath + browse button
          Row 1: H5 file label + filepath + browse button
          Row 2: "Select trial" label
          Row 3: trial list widget
          Row 4: action buttons (show info, reset trial, open plot window)
          Row 5: "Trial Info / Status" label
          Row 6: multi-line text area for info/status
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        grid = QGridLayout(central_widget)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(10)

        # ---------------------------
        # Row 0: Excel file selector
        # ---------------------------
        lbl_excel = QLabel("Excel File:")
        self.excel_label = QLabel("No Excel file selected")
        self.excel_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        btn_excel = QPushButton("Browse Excel File")
        btn_excel.clicked.connect(self.choose_excel_file)

        grid.addWidget(lbl_excel, 0, 0)
        grid.addWidget(self.excel_label, 0, 1)
        grid.addWidget(btn_excel, 0, 2)

        # ------------------------
        # Row 1: H5 file selector
        # ------------------------
        lbl_h5 = QLabel("H5 File:")
        self.h5_label = QLabel("No H5 file selected")
        self.h5_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        btn_h5 = QPushButton("Browse H5 File")
        btn_h5.clicked.connect(self.choose_h5_file)

        grid.addWidget(lbl_h5, 1, 0)
        grid.addWidget(self.h5_label, 1, 1)
        grid.addWidget(btn_h5, 1, 2)

        # -----------------------
        # Row 2: Trial list label
        # -----------------------
        lbl_trial = QLabel("Select a Patient Trial:")
        grid.addWidget(lbl_trial, 2, 0, 1, 3)

        # -----------------------
        # Row 3: Trial list widget
        # -----------------------
        self.trial_list = QListWidget()
        self.trial_list.itemSelectionChanged.connect(self.trial_selected)
        self.trial_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.trial_list, 3, 0, 1, 3)

        # ------------------------
        # Row 4: Action buttons
        # ------------------------
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

        # ------------------------
        # Row 5-6: Trial info area
        # ------------------------
        lbl_info = QLabel("Trial Info / Status:")
        grid.addWidget(lbl_info, 5, 0, 1, 3)

        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.info_text, 6, 0, 1, 3)

    # -------------------------------------------------------------------------
    # File loading
    # -------------------------------------------------------------------------

    def choose_excel_file(self):
        """
        Prompt the user to choose an Excel file, then load it via utils.file_utils.load_excel_file.

        Side effects on success:
          - sets self.excel_filepath
          - updates label in UI
          - loads DataFrame into self.df_events
          - populates the trial list
          - resets any previously-selected trial and any previously-loaded H5 data
        """
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)"
        )
        if not filepath:
            return

        self.excel_filepath = filepath
        self.excel_label.setText(filepath)

        # Load event/trial table from Excel
        self.df_events = load_excel_file(filepath)

        # If Excel loaded correctly, populate the list and clear downstream state
        if self.df_events is not None:
            self.populate_trial_list()

            # Reset downstream selections because trials changed
            self.selected_trial = None
            self.h5_filepath = None
            self.h5_label.setText("No H5 file selected")
            self.signals = None
            self.time_array = None
            self.channel_names = None
            self.info_text.clear()

    def choose_h5_file(self):
        """
        Prompt the user to choose an H5 file, then load it via utils.file_utils.load_h5_file.

        Expected return from load_h5_file:
          signals, time_array, channel_names

        Side effects on success:
          - sets self.h5_filepath
          - updates label in UI
          - fills self.signals / self.time_array / self.channel_names
        """
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select H5 File", "", "H5 Files (*.h5)"
        )
        if not filepath:
            return

        self.h5_filepath = filepath
        self.h5_label.setText(filepath)

        # Load iEEG data
        self.signals, self.time_array, self.channel_names = load_h5_file(filepath)

        # Simple guard if the file load failed
        if self.signals is None:
            QMessageBox.warning(self, "Error", "Failed to load H5 file.")

    # -------------------------------------------------------------------------
    # Trial list + selection
    # -------------------------------------------------------------------------

    def populate_trial_list(self):
        """
        Populate the QListWidget using the Excel DataFrame.

        Each list entry uses the "Patient Session Trial" column if available,
        otherwise falls back to a generic label ("Trial N").
        """
        self.trial_list.clear()

        if self.df_events is None or len(self.df_events) == 0:
            return

        for idx, row in self.df_events.iterrows():
            trial_str = row.get("Patient Session Trial", f"Trial {idx+1}")
            self.trial_list.addItem(str(trial_str))

    def trial_selected(self):
        """
        Slot called when the selection changes in the QListWidget.

        Sets:
          - self.selected_trial: the DataFrame row (as a Series) for the selected index

        Also updates the status area with a short “selected index/trial id” message.
        """
        items = self.trial_list.selectedItems()
        if not items:
            return

        index = self.trial_list.currentRow()
        self.selected_trial = self.df_events.iloc[index]

        self.info_text.setPlainText(
            f"Selected trial index: {index}\n"
            f"Selected trial: {str(self.selected_trial.get('Patient Session Trial','Unknown'))}"
        )

    # -------------------------------------------------------------------------
    # Display trial metadata
    # -------------------------------------------------------------------------

    def show_trial_info(self):
        """
        Render the selected trial's metadata into the info text area.

        Special handling:
          - For known time columns (LS_Start, LS_End, ...), attempt to parse
            Excel-style time strings using parse_excel_display_to_seconds and
            display both:
              * human-friendly MM:SS.s format
              * numeric seconds value with matching decimal precision
          - For missing/NA-style values, display "NA"
        """
        if self.selected_trial is None:
            QMessageBox.warning(self, "No Trial Selected", "Please select a trial from the list.")
            return

        from utils.file_utils import parse_excel_display_to_seconds, fmt_mmss_exact

        row = self.selected_trial
        info_lines = []

        # Columns expected to contain event times in Excel formats
        time_cols = ["LS_Start", "LS_End", "AD_Start", "Qes_Start", "Qes_End", "Ans_Start", "Ans_End", "AD_End"]

        # Iterate through all columns in the row
        for key, raw in row.to_dict().items():
            # Skip non-string columns and Excel "Unnamed" artifacts
            if not isinstance(key, str) or key.startswith("Unnamed"):
                continue

            # Normalize raw string for missing checks
            sraw = "" if raw is None else str(raw).strip()
            if sraw == "" or sraw.upper() in {"NA", "N/A", "-", "--"}:
                info_lines.append(f"{key}: NA")
                continue

            # If this is a time column, parse and pretty-print it
            if key in time_cols:
                secs, dp = parse_excel_display_to_seconds(sraw, default_dp=1, within_hour=True)
                if secs is None:
                    info_lines.append(f"{key}: (unparsable)")
                else:
                    pretty = fmt_mmss_exact(secs, dp)  # e.g., "13:15.0"
                    info_lines.append(f"{key}: {pretty}  ({secs:.{dp}f}s)")
            else:
                # Non-time fields displayed as-is
                info_lines.append(f"{key}: {raw}")

        self.info_text.setPlainText("\n".join(info_lines))

    # -------------------------------------------------------------------------
    # Reset + open plot window
    # -------------------------------------------------------------------------

    def reset_trial_selection(self):
        """
        Clear the currently selected trial and also reset any loaded H5 state.

        This is useful when the user wants to pick a different trial and ensure
        downstream windows aren't opened with stale selections/data.
        """
        self.trial_list.clearSelection()
        self.selected_trial = None

        self.info_text.clear()
        self.info_text.setPlainText("Trial selection cleared.\nPlease choose a new trial.")

        # Reset signal file state as well (forces user to reload if needed)
        self.h5_filepath = None
        self.h5_label.setText("No H5 file selected")
        self.signals = None
        self.time_array = None
        self.channel_names = None

    def open_plot_window(self):
        """
        Open the ChannelPlotWindow for the currently loaded H5 data and selected trial.

        Guards:
          - H5 must be loaded (signals/time_array exist)
          - A trial must be selected

        ChannelPlotWindow receives:
          - signals (n_channels x n_samples)
          - time_array (n_samples)
          - selected_trial (pandas Series for event times/metadata)
          - channel_names (list[str])
        """
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
