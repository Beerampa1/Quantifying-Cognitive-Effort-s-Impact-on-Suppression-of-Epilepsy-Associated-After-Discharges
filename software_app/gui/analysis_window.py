# file: gui/analysis_window.py

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox,
    QScrollArea, QWidget, QFrame, QSizePolicy, QFileDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def rolling_statistic(data, window_size, func):
    """
    Compute rolling statistic (mean, variance, etc.)
    using a sliding window over the signal
    """
    if len(data) < window_size:
        return np.array([])
    
    # Apply the function to each window segment
    return np.array([func(data[i:i+window_size])
                     for i in range(len(data) - window_size + 1)])


class ExpandedGraphDialog(QDialog):
    """
    Dialog window used to display an expanded version of a plot
    when the user double-clicks on a graph
    """
    def __init__(self, title, x, y, parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.resize(1000, 700)

        layout = QVBoxLayout(self)

        # Create matplotlib figure
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)

        # Plot data
        self.ax.plot(x, y)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.ax.set_title(title)
        self.ax.legend()
        self.figure.tight_layout()

        # Embed plot in Qt widget
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class AnalysisWindow(QDialog):
    """
    Main analysis window for visualizing original signals
    and computing rolling statistics
    """
    def __init__(self, parent, signals, time_array,
                 channel_indices, start_time, end_time,
                 channel_names=None):

        super().__init__(parent)

        self.setWindowTitle("Analysis Window")
        self.setWindowState(Qt.WindowMaximized)

        # Store input data
        self.signals = signals
        self.time_array = time_array
        self.fs = 1000  # Sampling frequency (Hz)
        self.channel_indices = channel_indices
        self.start_time = start_time
        self.end_time = end_time

        # Channel labels
        if channel_names is not None:
            self.channel_names = channel_names
        else:
            self.channel_names = [f"Channel {ch}" for ch in channel_indices]

        # Convert time window to sample indices
        start_idx = int(start_time * self.fs)
        end_idx = int(end_time * self.fs)

        # Extract time segment
        self.t_snippet = self.time_array[start_idx:end_idx]

        # Extract signal snippets for selected channels
        self.snippets = {}
        for ch in channel_indices:
            self.snippets[ch] = self.signals[ch, start_idx:end_idx]

        # Rolling window length (samples)
        self.rolling_window = 100

        # Store feature plot figure for saving
        self.feature_plot_figure = None

        self.initUI()

    def initUI(self):
        """Initialize GUI layout"""
        main_layout = QVBoxLayout(self)

        # ---------------- Original signal plot ----------------
        orig_container = QHBoxLayout()

        self.orig_canvas = self.create_plot_canvas(
            self.t_snippet, self.snippets, "Original Data"
        )
        self.orig_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        orig_container.addWidget(self.orig_canvas, stretch=1)

        # Scrollable legend
        self.orig_legend_scroll = QScrollArea()
        self.orig_legend_scroll.setWidgetResizable(True)
        self.orig_legend_widget = QWidget()
        self.orig_legend_layout = QVBoxLayout(self.orig_legend_widget)
        self.orig_legend_scroll.setWidget(self.orig_legend_widget)
        self.orig_legend_scroll.setFixedWidth(80)

        orig_container.addWidget(self.orig_legend_scroll, stretch=0)
        main_layout.addLayout(orig_container)

        # Update legend labels
        self.update_legend_widget(self.get_legend_entries(self.snippets))

        # ---------------- Feature buttons ----------------
        btn_layout = QHBoxLayout()

        btn_mean = QPushButton("Plot Rolling Mean")
        btn_mean.clicked.connect(self.plot_mean)

        btn_var = QPushButton("Plot Rolling Variance")
        btn_var.clicked.connect(self.plot_variance)

        btn_std = QPushButton("Plot Rolling Std Dev")
        btn_std.clicked.connect(self.plot_std)

        btn_median = QPushButton("Plot Rolling Median")
        btn_median.clicked.connect(self.plot_median)

        btn_rms = QPushButton("Plot Rolling RMS")
        btn_rms.clicked.connect(self.plot_rms)

        btn_layout.addWidget(btn_mean)
        btn_layout.addWidget(btn_var)
        btn_layout.addWidget(btn_std)
        btn_layout.addWidget(btn_median)
        btn_layout.addWidget(btn_rms)

        main_layout.addLayout(btn_layout)

        # ---------------- Feature plot area ----------------
        feature_container = QHBoxLayout()

        self.feature_plot_widget = QWidget()
        self.feature_plot_layout = QHBoxLayout(self.feature_plot_widget)
        self.feature_plot_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        feature_container.addWidget(self.feature_plot_widget, stretch=1)

        # Spacer for alignment with legend
        right_spacer = QFrame()
        right_spacer.setFixedWidth(80)
        feature_container.addWidget(right_spacer, stretch=0)

        main_layout.addLayout(feature_container)

        # ---------------- Save buttons ----------------
        save_layout = QHBoxLayout()

        btn_save_orig = QPushButton("Save Original Plot")
        btn_save_orig.clicked.connect(self.save_orig_plot)

        btn_save_feat = QPushButton("Save Feature Plot")
        btn_save_feat.clicked.connect(self.save_feature_plot)

        save_layout.addWidget(btn_save_orig)
        save_layout.addWidget(btn_save_feat)

        main_layout.addLayout(save_layout)

    def create_plot_canvas(self, t, snippets_dict, title):
        """Create a matplotlib canvas for plotting signals"""
        fig = plt.Figure(figsize=(7, 3), dpi=100)
        ax = fig.add_subplot(111)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Plot each channel
        for i, (ch, data) in enumerate(snippets_dict.items()):
            ch_name = (
                self.channel_names[ch]
                if ch < len(self.channel_names)
                else f"Channel {ch}"
            )
            ax.plot(t, data, label=ch_name,
                    color=colors[i % len(colors)])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Double-click to expand plot
        canvas.mpl_connect(
            "button_press_event",
            lambda event: self.expand_graph(title, t, snippets_dict)
            if event.dblclick else None
        )

        return canvas

    def expand_graph(self, title, t, snippets_dict):
        """Open expanded plot window"""
        dlg = ExpandedGraphDialog(
            title, t, self.aggregate_snippets(snippets_dict), self
        )
        dlg.exec_()

    def aggregate_snippets(self, snippets_dict):
        """Return data as-is (placeholder for future aggregation)"""
        return snippets_dict

    def get_legend_entries(self, snippets_dict):
        """Generate legend labels and colors"""
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        entries = []

        for i, ch in enumerate(snippets_dict.keys()):
            ch_name = (
                self.channel_names[ch]
                if ch < len(self.channel_names)
                else f"Channel {ch}"
            )
            entries.append((ch_name, colors[i % len(colors)]))

        return entries

    def update_legend_widget(self, legend_entries):
        """Update legend panel"""
        for i in reversed(range(self.orig_legend_layout.count())):
            widget = self.orig_legend_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for name, color in legend_entries:
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.orig_legend_layout.addWidget(lbl)

    def add_feature_plot(self, title, t, feature_data):
        """Plot computed rolling feature"""
        for i in reversed(range(self.feature_plot_layout.count())):
            widget = self.feature_plot_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        fig = plt.Figure(figsize=(7, 3), dpi=100)
        ax = fig.add_subplot(111)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Plot feature per channel
        for i, (ch, data) in enumerate(feature_data.items()):
            ch_name = (
                self.channel_names[ch]
                if ch < len(self.channel_names)
                else f"Channel {ch}"
            )
            ax.plot(t, data, label=ch_name,
                    color=colors[i % len(colors)])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(title)
        ax.set_title(title)
        fig.tight_layout()

        self.feature_plot_figure = fig

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        canvas.mpl_connect(
            "button_press_event",
            lambda event: self.expand_graph(title, t, feature_data)
            if event.dblclick else None
        )

        self.feature_plot_layout.addWidget(canvas, 1)

    def plot_feature(self, func, feature_title):
        """Compute and plot rolling feature"""
        feature_data = {}

        # Compute rolling statistic per channel
        for ch in self.channel_indices:
            data = self.snippets[ch]
            feature_data[ch] = rolling_statistic(
                data, self.rolling_window, func
            )

        if not feature_data:
            QMessageBox.warning(
                self, "Error",
                "No data available for feature calculation."
            )
            return

        sample_len = len(next(iter(feature_data.values())))
        if sample_len == 0:
            QMessageBox.warning(
                self, "Insufficient Data",
                "Not enough data to compute rolling statistics."
            )
            return

        # Align time axis with feature length
        t_feat = self.t_snippet[:sample_len]

        self.add_feature_plot(feature_title, t_feat, feature_data)

    # ---------- Feature buttons ----------
    def plot_mean(self):
        """Compute and plot rolling mean"""
        self.plot_feature(np.mean, "Rolling Mean")

    def plot_variance(self):
        """Compute and plot rolling variance"""
        self.plot_feature(np.var, "Rolling Variance")

    def plot_std(self):
        """Compute and plot rolling standard deviation"""
        self.plot_feature(np.std, "Rolling Std Dev")

    def plot_median(self):
        """Compute and plot rolling median"""
        self.plot_feature(np.median, "Rolling Median")

    def plot_rms(self):
        """Compute and plot rolling RMS"""
        def rms_func(x):
            return np.sqrt(np.mean(x ** 2))
        self.plot_feature(rms_func, "Rolling RMS")

    # ---------- Save plots ----------
    def save_orig_plot(self):
        """Save original signal plot"""
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Original Plot",
            "original_plot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        if fileName:
            self.orig_canvas.figure.savefig(fileName)

    def save_feature_plot(self):
        """Save computed feature plot"""
        if self.feature_plot_figure is None:
            QMessageBox.warning(
                self, "Error",
                "No feature plot available to save."
            )
            return

        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Feature Plot",
            "feature_plot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        if fileName:
            self.feature_plot_figure.savefig(fileName)
