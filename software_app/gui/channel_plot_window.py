from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QSizePolicy, QVBoxLayout, QScrollArea, QWidget,
    QCheckBox, QFrame, QHBoxLayout, QComboBox, QLabel, QLineEdit, QPushButton,
    QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from gui.flow_layout import FlowLayout
from utils.file_utils import parse_excel_display_to_seconds, fmt_mmss_exact




class ChannelPlotWindow(QDialog):
    def __init__(self, parent, signals, time_array, trial_info, channel_names=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Plotting (Flow Layout for Channels)")
        self.setWindowState(Qt.WindowMaximized)
        self.resize(900, 700)

        self.signals = signals                       # shape: (n_channels, n_samples)
        self.time_array = time_array                 # shape: (n_samples,)
        self.fs = 1000                               
        self.trial_info = trial_info

        # channel names + map (prevents silent fallback to channel 0)
        if channel_names:
            self.channel_names = channel_names
        else:
            num_channels = signals.shape[0]
            self.channel_names = [f"Channel {i}" for i in range(num_channels)]
        self.name_to_index = {name: i for i, name in enumerate(self.channel_names)}

        self.time_columns = ["LS_Start","LS_End","AD_Start","Qes_Start","Qes_End","Ans_Start","Ans_End","AD_End"]
        self.all_times = {}   # label -> (secs, dp)

        for col in self.time_columns:
            if col in trial_info.index:
                raw = trial_info[col]
                secs, dp = parse_excel_display_to_seconds(raw, default_dp=1, within_hour=True)
                if secs is not None:
                    self.all_times[col] = (float(secs), int(dp))

        sorted_items = sorted(self.all_times.items(), key=lambda kv: kv[1][0])
        self.full_start_options = [(c, v[0], v[1]) for c, v in sorted_items if c != "AD_End"]
        self.full_end_options   = [(c, v[0], v[1]) for c, v in sorted_items if c != "LS_Start"]

        self.start_options = self.full_start_options.copy()
        self.end_options   = self.full_end_options.copy()

        self.last_valid_start = self.start_options[0][1] if self.start_options else 0.0
        self.last_valid_end   = self.end_options[-1][1]  if self.end_options   else 0.0
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # ====== Channel selection group ======
        channel_group = QGroupBox("Select Channels")
        channel_group.setFixedHeight(140)
        channel_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ch_vbox = QVBoxLayout(channel_group)

        self.channel_scroll = QScrollArea()
        self.channel_scroll.setWidgetResizable(True)
        self.channel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.channel_container = QWidget()
        self.flow_layout = FlowLayout(self.channel_container)
        self.checkboxes = []

        exclude_channels = ["EKG1","EKG2","X1 DC1","X1 DC2","X1 DC3","X1 DC4"]
        for ch_name in self.channel_names:
            cb = QCheckBox(ch_name)
            should_exclude = any(pat.lower() in ch_name.lower() for pat in exclude_channels)
            cb.setChecked(not should_exclude)
            self.flow_layout.addWidget(cb)
            self.checkboxes.append(cb)

        self.channel_container.setMinimumHeight(300)
        self.channel_scroll.setWidget(self.channel_container)
        ch_vbox.addWidget(self.channel_scroll)
        main_layout.addWidget(channel_group)

        # ====== Time selectors ======
        time_frame = QFrame()
        time_hbox = QHBoxLayout(time_frame)

        self.start_combo = QComboBox(); self.end_combo = QComboBox()
        for c, secs, dp in self.start_options:
            self.start_combo.addItem(f"{c} ({fmt_mmss_exact(secs, dp)})", (secs, dp))
        for c, secs, dp in self.end_options:
            self.end_combo.addItem(f"{c} ({fmt_mmss_exact(secs, dp)})", (secs, dp))
        self.start_combo.currentIndexChanged.connect(self.on_start_select)
        self.end_combo.currentIndexChanged.connect(self.on_end_select)




        time_hbox.addWidget(QLabel("Start Time:"))
        time_hbox.addWidget(self.start_combo)
        time_hbox.addWidget(QLabel("End Time:"))
        time_hbox.addWidget(self.end_combo)

        self.custom_start = QLineEdit()
        self.custom_end   = QLineEdit()
        time_hbox.addWidget(QLabel("Custom Start (s):"))
        time_hbox.addWidget(self.custom_start)
        time_hbox.addWidget(QLabel("Custom End (s):"))
        time_hbox.addWidget(self.custom_end)

        self.extra_before = QLineEdit("0")
        self.extra_after  = QLineEdit("0")
        time_hbox.addWidget(QLabel("Extra Before:"))
        time_hbox.addWidget(self.extra_before)
        time_hbox.addWidget(QLabel("Extra After:"))
        time_hbox.addWidget(self.extra_after)

        main_layout.addWidget(time_frame)

        # Optional: live replot on edits (comment out if you prefer button-only)
        self.custom_start.editingFinished.connect(self.plot_data)
        self.custom_end.editingFinished.connect(self.plot_data)
        self.extra_before.editingFinished.connect(self.plot_data)
        self.extra_after.editingFinished.connect(self.plot_data)

        # ====== Buttons ======
        btn_hbox = QHBoxLayout()
        self.plot_btn = QPushButton("Plot Channels")
        self.plot_btn.clicked.connect(self.plot_data)
        self.analysis_btn = QPushButton("Open Analysis Window")
        self.analysis_btn.clicked.connect(self.open_analysis_window)
        self.fodn_btn = QPushButton("Open FODN Analysis")
        self.fodn_btn.clicked.connect(self.open_fodn_analysis_window)
        self.mfdfa_btn = QPushButton("Open MF-DFA Analysis")
        self.mfdfa_btn.clicked.connect(self.open_mfdfa_analysis_window)
        self.mfdfa_overlap_btn = QPushButton("Open MF-DFA (Overlapping)")
        self.mfdfa_overlap_btn.clicked.connect(self.open_mfdfa_overlap_window)
        self.model_training_btn = QPushButton("Open Model Training")
        self.model_training_btn.clicked.connect(self.open_model_training_window)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)

        for b in (self.plot_btn, self.analysis_btn, self.fodn_btn,
                  self.mfdfa_btn, self.mfdfa_overlap_btn, self.model_training_btn, self.close_btn):
            btn_hbox.addWidget(b)
        main_layout.addLayout(btn_hbox)

        # ====== Plot + legend ======
        plot_legend_hbox = QHBoxLayout()
        self.figure = plt.Figure(figsize=(7,4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_legend_hbox.addWidget(self.canvas, stretch=5)

        self.legend_scroll = QScrollArea()
        self.legend_scroll.setWidgetResizable(True)
        self.legend_widget = QWidget()
        self.legend_vlayout = QVBoxLayout(self.legend_widget)
        self.legend_scroll.setWidget(self.legend_widget)
        self.legend_scroll.setFixedWidth(80)
        plot_legend_hbox.addWidget(self.legend_scroll, stretch=1)
        main_layout.addLayout(plot_legend_hbox)

        self.sync_custom_times()

    # ---- helpers ----
    def sync_custom_times(self):
        try:
            start_secs, start_dp = self.start_combo.currentData()
            self.last_valid_start = start_secs
            self.custom_start.setText(f"{start_secs:.{start_dp}f}")
        except Exception:
            self.custom_start.setText(f"{self.last_valid_start:.2f}")

        try:
            end_secs, end_dp = self.end_combo.currentData()
            self.last_valid_end = end_secs
            self.custom_end.setText(f"{end_secs:.{end_dp}f}")
        except Exception:
            self.custom_end.setText(f"{self.last_valid_end:.2f}")



    def on_start_select(self):
        self.start_combo.blockSignals(True)
        self.end_combo.blockSignals(True)

        self.sync_custom_times()
        try:
            start_secs, _ = self.start_combo.currentData()
        except Exception:
            start_secs = self.last_valid_start

        # self.full_end_options is [(c, s, dp)]
        new_end = [(c, s, dp) for (c, s, dp) in self.full_end_options if s > start_secs]

        if new_end:
            self.end_options = new_end
            self.end_combo.clear()
            for c, s, dp in self.end_options:
                # show MM:SS with the same decimals Excel showed
                self.end_combo.addItem(f"{c} ({fmt_mmss_exact(s, dp)})", (s, dp))
            self.end_combo.setCurrentIndex(0)
        else:
            QMessageBox.warning(
                self, "Invalid Selection",
                "No valid end times available for the chosen start time.\n"
                "Please choose a different start time or use custom entry."
            )

        self.sync_custom_times()
        self.start_combo.blockSignals(False)
        self.end_combo.blockSignals(False)


    def on_end_select(self):
       self.sync_custom_times()


    # ---- plotting ----
    def plot_data(self):
        checked_channels = [cb.text() for cb in self.channel_container.findChildren(QCheckBox) if cb.isChecked()]
        if not checked_channels:
            QMessageBox.warning(self, "Error", "Please select at least one channel.")
            return

        try:
            start_time = float(self.custom_start.text())
            end_time   = float(self.custom_end.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid start/end time.")
            return
        if start_time >= end_time:
            QMessageBox.critical(self, "Error", "Start time must be less than end time.")
            return

        try:
            extra_b = float(self.extra_before.text())
            extra_a = float(self.extra_after.text())
        except ValueError:
            extra_b = extra_a = 0.0

        start_time_adj = max(0.0, start_time - extra_b)
        end_time_adj   = end_time + extra_a

        import numpy as np
        n_samples = len(self.time_array)
        start_idx = int(np.rint(start_time_adj * self.fs))
        end_idx   = int(np.rint(end_time_adj   * self.fs))
        start_idx = max(0, min(start_idx, n_samples - 1))
        end_idx   = max(0, min(end_idx,   n_samples))
        if start_idx >= end_idx:
            QMessageBox.critical(self, "Error", "Invalid time interval after adjustment.")
            return

        self.ax.clear()
        legend_entries = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, ch_name in enumerate(checked_channels):
            try:
                ch_index = self.name_to_index[ch_name]
            except KeyError:
                QMessageBox.warning(self, "Channel name mismatch",
                                    f"Could not find data for '{ch_name}'.")
                continue

            data_snip = self.signals[ch_index, start_idx:end_idx]
            t_snip    = self.time_array[start_idx:end_idx]
            color = colors[i % len(colors)]
            self.ax.plot(t_snip, data_snip, color=color, label=ch_name)
            legend_entries.append((ch_name, color))

        # lines at adjusted bounds
        self.ax.axvline(x=start_time_adj, color='green', linestyle='--', label="Start")
        self.ax.axvline(x=end_time_adj,   color='red',   linestyle='--', label="End")

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        # show exactly what the text fields say for clarity
        self.ax.set_title(f"Multi-Channel Plot [{self.custom_start.text()}s → {self.custom_end.text()}s]")

        self.ax.relim(); self.ax.autoscale_view()
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.update_legend_widget(legend_entries)



    def update_legend_widget(self, legend_entries):
        # clear
        for i in reversed(range(self.legend_vlayout.count())):
            w = self.legend_vlayout.itemAt(i).widget()
            if w:
                w.setParent(None)
        # add
        for name, color in legend_entries:
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.legend_vlayout.addWidget(lbl)

    # ---- other windows (unchanged) ----
    def open_analysis_window(self):
        checked_channels = [cb.text() for cb in self.channel_container.findChildren(QCheckBox) if cb.isChecked()]
        if not checked_channels:
            QMessageBox.warning(self, "Error", "Please select at least one channel.")
            return
        channel_indices = []
        for ch_name in checked_channels:
            try:
                idx = self.name_to_index[ch_name]
            except KeyError:
                idx = 0
            channel_indices.append(idx)
        try:
            start_time = float(self.custom_start.text())
            end_time   = float(self.custom_end.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid time selection.")
            return
        if start_time >= end_time:
            QMessageBox.warning(self, "Error", "Start time must be less than end time.")
            return
        from gui.analysis_window import AnalysisWindow
        AnalysisWindow(self, self.signals, self.time_array, channel_indices, start_time, end_time, channel_names=self.channel_names).exec_()

    def open_fodn_analysis_window(self):
        try:
            start_time = float(self.custom_start.text())
            end_time   = float(self.custom_end.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid time selection.")
            return
        if start_time >= end_time:
            QMessageBox.warning(self, "Error", "Start time must be less than end time.")
            return
        labeled_times = [
        (label, secs)
        for label, (secs, _dp) in self.all_times.items()
        if start_time <= secs <= end_time
        ]
        labeled_times.sort(key=lambda x: x[1])
        if not labeled_times or labeled_times[0][1] > start_time:
            labeled_times.insert(0, ("Custom_Start", start_time))
        if labeled_times[-1][1] < end_time:
            labeled_times.append(("Custom_End", end_time))
        from gui.fodn_analysis_window import FODNAnalysisWindow
        win = FODNAnalysisWindow(self, self.signals, self.time_array, start_time, end_time, self.channel_names, labeled_times=labeled_times)
        win.setWindowState(Qt.WindowMaximized)
        win.exec_()

    def open_mfdfa_analysis_window(self):
        try:
            start_time = float(self.custom_start.text())
            end_time   = float(self.custom_end.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid time selection.")
            return
        if start_time >= end_time:
            QMessageBox.warning(self, "Error", "Start time must be less than end time.")
            return
        labeled_times = [
        (label, secs)
        for label, (secs, _dp) in self.all_times.items()
        if start_time <= secs <= end_time
        ]
        labeled_times.sort(key=lambda x: x[1])
        if not labeled_times or labeled_times[0][1] > start_time:
            labeled_times.insert(0, ("Custom_Start", start_time))
        if labeled_times[-1][1] < end_time:
            labeled_times.append(("Custom_End", end_time))

        from gui.mfdfa_analysis_window import MFDFAAnalysisWindow
        win = MFDFAAnalysisWindow(self, self.signals, self.time_array, start_time, end_time, self.channel_names, labeled_times=labeled_times)
        win.setWindowState(Qt.WindowMaximized)
        win.exec_()

    def open_mfdfa_overlap_window(self):
        try:
            start_time = float(self.custom_start.text())
            end_time   = float(self.custom_end.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid time selection.")
            return
        if start_time >= end_time:
            QMessageBox.warning(self, "Error", "Start time must be less than end time.")
            return
        # labeled_times = [
        # (label, secs)
        # for label, (secs, _dp) in self.all_times.items()
        # if start_time <= secs <= end_time
        # ]
        # labeled_times.sort(key=lambda x: x[1])
        # if not labeled_times or labeled_times[0][1] > start_time:
        #     labeled_times.insert(0, ("Custom_Start", start_time))
        # if labeled_times[-1][1] < end_time:
        #     labeled_times.append(("Custom_End", end_time))
        labeled_times = [
        ("Custom_Start", start_time),
        ("Custom_End",   end_time),
    ]
        from gui.mfdfa_overlap_window import MFDFAOverlapWindow
        win = MFDFAOverlapWindow(self, self.signals, self.time_array, start_time, end_time, self.channel_names, labeled_times=labeled_times)
        win.setWindowState(Qt.WindowMaximized)
        win.exec_()

    def open_model_training_window(self):
        from gui.model_training_window import ModelTrainingWindow
        win = ModelTrainingWindow(self)
        win.setWindowState(Qt.WindowMaximized)
        win.exec_()






















































# # file: gui/channel_plot_window.py
# from PyQt5.QtWidgets import (
#     QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout,
#     QLabel, QComboBox, QLineEdit, QPushButton, QMessageBox, 
#     QScrollArea, QWidget, QCheckBox, QSizePolicy, QFrame
# )
# from PyQt5.QtCore import Qt
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from utils.file_utils import parse_tod
# from gui.flow_layout import FlowLayout
# from gui.analysis_window import AnalysisWindow


# try:
#     from gui.mfdfa_overlap_window import MFDFAOverlapWindow
# except Exception:
#     MFDFAOverlapWindow = None



# class ChannelPlotWindow(QDialog):
#     def __init__(self, parent, signals, time_array, trial_info, channel_names=None):
#         super().__init__(parent)
#         self.setWindowTitle("Channel Plotting (Flow Layout for Channels)")
#         self.setWindowState(Qt.WindowMaximized)
#         self.resize(900, 700)
#         self.signals = signals
#         self.time_array = time_array
#         self.fs = 1000  
#         self.trial_info = trial_info
        
#         if channel_names:
#             self.channel_names = channel_names
#         else:
#             num_channels = signals.shape[0]
#             self.channel_names = [f"Channel {i}" for i in range(num_channels)]
        
#         self.time_columns = ["LS_Start", "LS_End", "AD_Start", "Qes_Start", "Qes_End", "Ans_Start", "Ans_End", "AD_End"]
#         self.all_times = {}
#         for col in self.time_columns:
#             if col in trial_info.index:
#                 val = parse_tod(trial_info[col])
#                 if val is not None:
#                     self.all_times[col] = val
#         sorted_times = sorted(self.all_times.items(), key=lambda x: x[1])
        
#         self.full_start_options = [(c, v) for c, v in sorted_times if c != "AD_End"]
#         self.full_end_options = [(c, v) for c, v in sorted_times if c != "LS_Start"]
        
#         self.start_options = self.full_start_options.copy()
#         self.end_options = self.full_end_options.copy()
        
#         self.last_valid_start = self.start_options[0][1] if self.start_options else 0
#         self.last_valid_end = self.end_options[-1][1] if self.end_options else 0
        
#         self.initUI()
    
#     def initUI(self):
#         main_layout = QVBoxLayout(self)
        
#         channel_group = QGroupBox("Select Channels")
#         channel_group.setFixedHeight(140)
#         channel_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
#         ch_vbox = QVBoxLayout(channel_group)
        
#         self.channel_scroll = QScrollArea()
#         self.channel_scroll.setWidgetResizable(True)
#         self.channel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
#         self.channel_container = QWidget()
#         self.flow_layout = FlowLayout(self.channel_container)
#         self.checkboxes = []

#         exclude_channels = ["EKG1","EKG2","X1 DC1","X1 DC2","X1 DC3","X1 DC4"]
#         for ch_name in self.channel_names:
#             cb = QCheckBox(ch_name)
#             should_exclude = any(pattern.lower() in ch_name.lower() for pattern in exclude_channels)
#             cb.setChecked(not should_exclude)
#             self.flow_layout.addWidget(cb)
#             self.checkboxes.append(cb)

#         self.channel_container.setMinimumHeight(300)
        
#         self.channel_scroll.setWidget(self.channel_container)
#         ch_vbox.addWidget(self.channel_scroll)
#         main_layout.addWidget(channel_group)
        
#         time_frame = QFrame()
#         time_hbox = QHBoxLayout(time_frame)
#         self.start_combo = QComboBox()
#         for c, v in self.start_options:
#             self.start_combo.addItem(f"{c} ({v:.2f}s)")
#         self.end_combo = QComboBox()
#         for c, v in self.end_options:
#             self.end_combo.addItem(f"{c} ({v:.2f}s)")
#         self.start_combo.currentIndexChanged.connect(self.on_start_select)
#         self.end_combo.currentIndexChanged.connect(self.on_end_select)
#         time_hbox.addWidget(QLabel("Start Time:"))
#         time_hbox.addWidget(self.start_combo)
#         time_hbox.addWidget(QLabel("End Time:"))
#         time_hbox.addWidget(self.end_combo)
#         self.custom_start = QLineEdit()
#         self.custom_end = QLineEdit()
#         time_hbox.addWidget(QLabel("Custom Start (s):"))
#         time_hbox.addWidget(self.custom_start)
#         time_hbox.addWidget(QLabel("Custom End (s):"))
#         time_hbox.addWidget(self.custom_end)
#         self.extra_before = QLineEdit("0")
#         self.extra_after = QLineEdit("0")
#         time_hbox.addWidget(QLabel("Extra Before:"))
#         time_hbox.addWidget(self.extra_before)
#         time_hbox.addWidget(QLabel("Extra After:"))
#         time_hbox.addWidget(self.extra_after)
#         main_layout.addWidget(time_frame)
        
#         btn_hbox = QHBoxLayout()
#         self.plot_btn = QPushButton("Plot Channels")
#         self.plot_btn.clicked.connect(self.plot_data)
#         self.analysis_btn = QPushButton("Open Analysis Window")
#         self.analysis_btn.clicked.connect(self.open_analysis_window)
#         self.fodn_btn = QPushButton("Open FODN Analysis")
#         self.fodn_btn.clicked.connect(self.open_fodn_analysis_window)
#         self.mfdfa_btn = QPushButton("Open MF-DFA Analysis")
#         self.mfdfa_btn.clicked.connect(self.open_mfdfa_analysis_window)
#         self.mfdfa_overlap_btn = QPushButton("Open MF-DFA (Overlapping)")
#         self.mfdfa_overlap_btn.clicked.connect(self.open_mfdfa_overlap_window)
#         self.model_training_btn = QPushButton("Open Model Training")
#         self.model_training_btn.clicked.connect(self.open_model_training_window)
#         self.close_btn = QPushButton("Close")
#         self.close_btn.clicked.connect(self.accept)
#         btn_hbox.addWidget(self.plot_btn)
#         btn_hbox.addWidget(self.analysis_btn)
#         btn_hbox.addWidget(self.fodn_btn)
#         btn_hbox.addWidget(self.mfdfa_btn)
#         btn_hbox.addWidget(self.mfdfa_overlap_btn)
#         btn_hbox.addWidget(self.model_training_btn)
#         btn_hbox.addWidget(self.close_btn)
#         main_layout.addLayout(btn_hbox)
        
#         plot_legend_hbox = QHBoxLayout()
#         self.figure = plt.Figure(figsize=(7,4), dpi=100)
#         self.ax = self.figure.add_subplot(111)
#         self.canvas = FigureCanvas(self.figure)
#         self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         plot_legend_hbox.addWidget(self.canvas, stretch=5)
        
#         self.legend_scroll = QScrollArea()
#         self.legend_scroll.setWidgetResizable(True)
#         self.legend_widget = QWidget()
#         self.legend_vlayout = QVBoxLayout(self.legend_widget)
#         self.legend_scroll.setWidget(self.legend_widget)
#         self.legend_scroll.setFixedWidth(80)
#         plot_legend_hbox.addWidget(self.legend_scroll, stretch=1)
#         main_layout.addLayout(plot_legend_hbox)
        
#         self.sync_custom_times()
    
#     def sync_custom_times(self):
#         try:
#             start_text = self.start_combo.currentText()
#             start_val = float(start_text.split("(")[1].split("s")[0])
#             self.last_valid_start = start_val
#         except Exception:
#             start_val = self.last_valid_start
#         try:
#             end_text = self.end_combo.currentText()
#             end_val = float(end_text.split("(")[1].split("s")[0])
#             self.last_valid_end = end_val
#         except Exception:
#             end_val = self.last_valid_end
#         self.custom_start.setText(f"{start_val:.2f}")
#         self.custom_end.setText(f"{end_val:.2f}")
    
#     def on_start_select(self):
#         self.start_combo.blockSignals(True)
#         self.end_combo.blockSignals(True)
#         self.sync_custom_times()
#         try:
#             start_text = self.start_combo.currentText()
#             start_val = float(start_text.split("(")[1].split("s")[0])
#         except Exception:
#             start_val = self.last_valid_start
#         new_end = [(c, v) for c, v in self.full_end_options if v > start_val]
#         if new_end:
#             self.end_options = new_end
#             self.end_combo.clear()
#             for c, v in self.end_options:
#                 self.end_combo.addItem(f"{c} ({v:.2f}s)")
#             self.end_combo.setCurrentIndex(0)
#         else:
#             QMessageBox.warning(self, "Invalid Selection", 
#                 "No valid end times available for the chosen start time.\nPlease choose a different start time or use custom entry.")
#         self.sync_custom_times()
#         self.start_combo.blockSignals(False)
#         self.end_combo.blockSignals(False)
    
#     def on_end_select(self):
#         self.sync_custom_times()
    
#     def plot_data(self):
#         checked_channels = []
#         for cb in self.channel_container.findChildren(QCheckBox):
#             if cb.isChecked():
#                 checked_channels.append(cb.text())
#         if not checked_channels:
#             QMessageBox.warning(self, "Error", "Please select at least one channel.")
#             return
        
#         try:
#             start_time = float(self.custom_start.text())
#             end_time = float(self.custom_end.text())
#         except ValueError:
#             QMessageBox.critical(self, "Error", "Invalid start/end time.")
#             return
#         if start_time >= end_time:
#             QMessageBox.critical(self, "Error", "Start time must be less than end time.")
#             return
        
#         try:
#             extra_b = float(self.extra_before.text())
#             extra_a = float(self.extra_after.text())
#         except ValueError:
#             extra_b = extra_a = 0
        
#         start_time_adj = max(0, start_time - extra_b)
#         end_time_adj = end_time + extra_a
        
#         n_samples = len(self.time_array)
#         start_idx = int(start_time_adj * self.fs)
#         end_idx = int(end_time_adj * self.fs)
#         start_idx = max(0, min(start_idx, n_samples-1))
#         end_idx = max(0, min(end_idx, n_samples))
#         if start_idx >= end_idx:
#             QMessageBox.critical(self, "Error", "Invalid time interval after adjustment.")
#             return
        
#         self.ax.clear()
#         legend_entries = []
#         colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#         i = 0
#         for ch_name in checked_channels:
#             try:
#                 ch_index = self.channel_names.index(ch_name)
#             except Exception:
#                 ch_index = 0
#             data_snip = self.signals[ch_index, start_idx:end_idx]
#             t_snip = self.time_array[start_idx:end_idx]
#             color = colors[i % len(colors)]
#             self.ax.plot(t_snip, data_snip, color=color, label=ch_name)
#             legend_entries.append((ch_name, color))
#             i += 1
        
#         self.ax.axvline(x=start_time, color='green', linestyle='--', label="Start Time")
#         self.ax.axvline(x=end_time, color='red', linestyle='--', label="End Time")
#         self.ax.set_xlabel("Time (s)")
#         self.ax.set_ylabel("Amplitude")
#         self.ax.set_title("Multi-Channel Plot")
#         self.canvas.draw()
#         self.update_legend_widget(legend_entries)
    
#     def update_legend_widget(self, legend_entries):
#         for i in reversed(range(self.legend_vlayout.count())):
#             widget = self.legend_vlayout.itemAt(i).widget()
#             if widget:
#                 widget.setParent(None)
#         for name, color in legend_entries:
#             lbl = QLabel(name)
#             lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
#             self.legend_vlayout.addWidget(lbl)
    
#     def open_analysis_window(self):
#         checked_channels = []
#         for cb in self.channel_container.findChildren(QCheckBox):
#             if cb.isChecked():
#                 checked_channels.append(cb.text())
#         if not checked_channels:
#             QMessageBox.warning(self, "Error", "Please select at least one channel.")
#             return
#         channel_indices = []
#         for ch_name in checked_channels:
#             try:
#                 idx = self.channel_names.index(ch_name)
#             except Exception:
#                 idx = 0
#             channel_indices.append(idx)
#         try:
#             start_time = float(self.custom_start.text())
#             end_time = float(self.custom_end.text())
#         except:
#             QMessageBox.warning(self, "Error", "Invalid time selection.")
#             return
#         if start_time >= end_time:
#             QMessageBox.warning(self, "Error", "Start time must be less than end time.")
#             return
#         from gui.analysis_window import AnalysisWindow
#         analysisWin = AnalysisWindow(self, self.signals, self.time_array, channel_indices, start_time, end_time, channel_names=self.channel_names)
#         analysisWin.exec_()

#     def open_fodn_analysis_window(self):
#         try:
#             start_time = float(self.custom_start.text())
#             end_time = float(self.custom_end.text())
#         except ValueError:
#             QMessageBox.warning(self, "Error", "Invalid time selection.")
#             return
#         if start_time >= end_time:
#             QMessageBox.warning(self, "Error", "Start time must be less than end time.")
#             return
#         # Retrieve labeled times from self.all_times (populated when trial info was loaded)
#         # self.all_times is a dictionary {label: time}
#         labeled_times = [(label, t) for label, t in self.all_times.items() if t >= start_time and t <= end_time]
#         # Sort labeled times in ascending order:
#         labeled_times = sorted(labeled_times, key=lambda x: x[1])
#         # Ensure that the custom start and end boundaries are included:
#         if not labeled_times or labeled_times[0][1] > start_time:
#             labeled_times.insert(0, ("Custom_Start", start_time))
#         if labeled_times[-1][1] < end_time:
#             labeled_times.append(("Custom_End", end_time))
#         from gui.fodn_analysis_window import FODNAnalysisWindow
#         fodn_win = FODNAnalysisWindow(self, self.signals, self.time_array, start_time, end_time, self.channel_names, labeled_times=labeled_times)
#         fodn_win.setWindowState(Qt.WindowMaximized)
#         fodn_win.exec_()

#     def open_mfdfa_analysis_window(self):
#         try:
#             start_time = float(self.custom_start.text())
#             end_time   = float(self.custom_end.text())
#         except ValueError:
#             QMessageBox.warning(self, "Error", "Invalid time selection.")
#             return
#         if start_time >= end_time:
#             QMessageBox.warning(self, "Error", "Start time must be less than end time.")
#             return

#         # 1. Collect all labeled times in the interval
#         labeled_times = [
#             (label, t)
#             for label, t in self.all_times.items()
#             if t >= start_time and t <= end_time
#         ]
#         labeled_times.sort(key=lambda x: x[1])
#         # 2. Ensure custom bounds are included
#         if not labeled_times or labeled_times[0][1] > start_time:
#             labeled_times.insert(0, ("Custom_Start", start_time))
#         if labeled_times[-1][1] < end_time:
#             labeled_times.append(("Custom_End", end_time))

#         from gui.mfdfa_analysis_window import MFDFAAnalysisWindow
#         # 3. Pass labeled_times into the constructor
#         mfdfa_win = MFDFAAnalysisWindow(
#             self,
#             self.signals,
#             self.time_array,
#             start_time,
#             end_time,
#             self.channel_names,
#             labeled_times=labeled_times
#         )
#         mfdfa_win.setWindowState(Qt.WindowMaximized)
#         mfdfa_win.exec_()

#     def open_mfdfa_overlap_window(self):
#      if MFDFAOverlapWindow is None:
#         QMessageBox.warning(self, "Unavailable",
#                             "Overlapping MF-DFA window not found (mfdfa_overlap_window.py).")
#         return
#      try:
#         start_time = float(self.custom_start.text())
#         end_time   = float(self.custom_end.text())
#      except ValueError:
#         QMessageBox.warning(self, "Error", "Invalid time selection.")
#         return
#      if start_time >= end_time:
#         QMessageBox.warning(self, "Error", "Start time must be less than end time.")
#         return

#      labeled_times = [
#         (label, t) for label, t in self.all_times.items()
#         if t >= start_time and t <= end_time
#      ]
#      labeled_times.sort(key=lambda x: x[1])
#      if not labeled_times or labeled_times[0][1] > start_time:
#         labeled_times.insert(0, ("Custom_Start", start_time))
#      if labeled_times[-1][1] < end_time:
#         labeled_times.append(("Custom_End", end_time))

#      win = MFDFAOverlapWindow(self, self.signals, self.time_array,
#                              start_time, end_time, self.channel_names,
#                              labeled_times=labeled_times)
#      win.setWindowState(Qt.WindowMaximized)
#      win.exec_()



#     def open_model_training_window(self):
#         from gui.model_training_window import ModelTrainingWindow
#         model_win = ModelTrainingWindow(self)
#         model_win.setWindowState(Qt.WindowMaximized)
#         model_win.exec_()
