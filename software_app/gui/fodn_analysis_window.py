# file: gui/fodn_analysis_window.py
# GUI window for running and visualizing FODN analysis on a selected time interval.
# Features:
#   - Collapsible parameter panel (chunk size, numFract, niter, lambdaUse)
#   - Collapsible channel selection (uses FlowLayout to wrap checkboxes)
#   - Runs FODN analysis per labeled segment and stores chunk-wise results
#   - Chunk navigation (Prev/Next) with live plot refresh
#   - Plot widgets (up to 4) with Save PNG / Save CSV / Close
#   - "Save All Chunks" export: writes per-segment/per-chunk folders with data + plots
#   - Extra global plots across many folders (alpha boxplot + eigenvector heatmap)

import sys
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QProgressDialog,
    QScrollArea, QWidget, QSizePolicy, QGroupBox, QLineEdit, QCheckBox, QToolButton,
    QFileDialog, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Runs the actual FODN computation on a data segment
from utils.fodn_utils import run_fodn_analysis


# --------------------------- Collapsible Panel widget ---------------------------

class CollapsiblePanel(QGroupBox):
    """
    Simple collapsible UI panel:
    - Shows a header button with an arrow
    - Expands/collapses the content widget
    Used for parameters and channel selection to keep UI compact.
    """
    def __init__(self, title="Parameters", parent=None):
        super().__init__(parent)
        self.setTitle("")  # hide groupbox title, we use the toolbutton instead

        # Toggle button that controls expansion
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; padding: 2px; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_toggle)

        # Content area shown/hidden by toggle
        self.content = QWidget()
        self.content.setVisible(False)
        self.content.setContentsMargins(0, 0, 0, 0)

        # Layout contains the header button + the content widget
        lay = QHBoxLayout()
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        self.setLayout(lay)

        # When collapsed, limit height to only the header row
        self.setMaximumHeight(self.toggle_button.sizeHint().height())

    def on_toggle(self):
        """Expand/collapse panel and update arrow direction"""
        checked = self.toggle_button.isChecked()
        self.content.setVisible(checked)
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        # Allow panel to expand when opened, clamp height when closed
        if checked:
            self.setMaximumHeight(16777215)
        else:
            self.setMaximumHeight(self.toggle_button.sizeHint().height())


# --------------------------- Plot cell widget ---------------------------

class PlotCellWidget(QWidget):
    """
    A container for one plot + control buttons.
    Each cell holds:
      - Matplotlib canvas
      - Save PNG / Save CSV / Close buttons
      - plot_type label stored to allow "refresh" when chunk changes
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.canvas = None
        self.plot_type = None  # remembers which plot to re-create when chunk changes

        # Small control buttons under each plot
        self.btn_save_png = QPushButton("Save PNG")
        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_close = QPushButton("Close")

        for btn in [self.btn_save_png, self.btn_save_csv, self.btn_close]:
            btn.setMaximumHeight(25)
            btn.setFont(QFont("", 8))

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(2, 2, 2, 2)
        btn_layout.addWidget(self.btn_save_png)
        btn_layout.addWidget(self.btn_save_csv)
        btn_layout.addWidget(self.btn_close)

        self.layout.addLayout(btn_layout)

    def setCanvas(self, canvas):
        """Replace existing canvas with a new one (used during refresh)"""
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.setParent(None)

        self.canvas = canvas
        self.layout.insertWidget(0, self.canvas)


# --------------------------- Main window ---------------------------

class FODNAnalysisWindow(QDialog):
    """
    Main FODN analysis UI.

    Inputs:
      signals: numpy array (channels x samples)
      time_array: numpy array (samples,)
      start_time, end_time: selected window from channel plot window
      channel_names: list of channel names (from .h5)
      labeled_times: optional segment boundaries inside the selected interval
    """
    def __init__(self, parent, signals, time_array, start_time, end_time, channel_names, labeled_times=None):
        super().__init__(parent)

        self.setWindowTitle("FODN Analysis Window")
        self.setWindowState(Qt.WindowMaximized)

        # Store input data
        self.signals = signals
        self.time_array = time_array
        self.fs = 1000  # sampling frequency (Hz) used to convert seconds -> indices
        self.start_time = start_time
        self.end_time = end_time
        self.all_channel_names = channel_names

        # Default channel selection (exclude known unwanted channels)
        self.default_channel_states = {}
        unwanted = ["EKG1", "EKG2", "X1 DC1", "X1 DC2", "X1 DC3", "X1 DC4"]
        for name in self.all_channel_names:
            self.default_channel_states[name] = (False if name in unwanted else True)

        # ----- Analysis parameters (editable via UI) -----
        self.chunk_size = 0.5
        self.numFract = 165  # chosen to avoid numerical errors (largest safe J array length)
        self.niter = 3
        self.lambdaUse = 0.5

        # Storage for chunk-wise results
        self.fodn_results = []           # list of dicts (one dict per chunk)
        self.current_chunk_index = 0     # index into fodn_results for navigation

        self.selected_channel_indices = None  # actual channel indices used in analysis

        # ----- Process labeled segment boundaries -----
        if labeled_times is None or len(labeled_times) == 0:
            # Default: analyze the whole selected window as one segment
            self.labeled_times = [("Custom_Start", start_time), ("Custom_End", end_time)]
        else:
            # Sort and clamp to [start_time, end_time]
            sorted_times = sorted(labeled_times, key=lambda x: x[1])
            self.labeled_times = [(label, t) for label, t in sorted_times if start_time <= t <= end_time]

            # Ensure boundaries exist
            if not self.labeled_times or self.labeled_times[0][1] > start_time:
                self.labeled_times.insert(0, ("Custom_Start", start_time))
            if self.labeled_times[-1][1] < end_time:
                self.labeled_times.append(("Custom_End", end_time))

        self.initUI()

    def initUI(self):
        """Build the full UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # ---------------- Top row: parameters + channel selection ----------------
        top_row = QHBoxLayout()
        top_row.setSpacing(5)

        # Parameter panel (collapsible)
        self.param_panel = CollapsiblePanel("FODN Parameters")
        param_layout = QHBoxLayout(self.param_panel.content)
        param_layout.setContentsMargins(0, 0, 0, 0)

        # Parameter fields
        lbl_chunk = QLabel("Chunk Size (sec):")
        self.edit_chunk = QLineEdit(str(self.chunk_size))
        self.edit_chunk.setMaximumWidth(60)

        lbl_numfract = QLabel("numFract:")
        self.edit_numfract = QLineEdit(str(self.numFract))
        self.edit_numfract.setMaximumWidth(60)

        lbl_niter = QLabel("niter:")
        self.edit_niter = QLineEdit(str(self.niter))
        self.edit_niter.setMaximumWidth(60)

        lbl_lambda = QLabel("lambdaUse:")
        self.edit_lambda = QLineEdit(str(self.lambdaUse))
        self.edit_lambda.setMaximumWidth(60)

        for widget in [lbl_chunk, self.edit_chunk, lbl_numfract, self.edit_numfract,
                       lbl_niter, self.edit_niter, lbl_lambda, self.edit_lambda]:
            widget.setFont(QFont("", 8))

        param_layout.addWidget(lbl_chunk)
        param_layout.addWidget(self.edit_chunk)
        param_layout.addWidget(lbl_numfract)
        param_layout.addWidget(self.edit_numfract)
        param_layout.addWidget(lbl_niter)
        param_layout.addWidget(self.edit_niter)
        param_layout.addWidget(lbl_lambda)
        param_layout.addWidget(self.edit_lambda)
        param_layout.addStretch()

        top_row.addWidget(self.param_panel)

        # Channel selection panel (collapsible)
        self.chan_panel = CollapsiblePanel("Select Channels")
        chan_layout = QHBoxLayout(self.chan_panel.content)
        chan_layout.setContentsMargins(0, 0, 0, 0)

        self.chan_scroll = QScrollArea()
        self.chan_scroll.setWidgetResizable(True)

        self.chan_container = QWidget()

        # Use FlowLayout so checkboxes wrap automatically
        from gui.flow_layout import FlowLayout
        self.chan_flow_layout = FlowLayout(self.chan_container)

        self.chan_checkboxes = {}
        for name in self.all_channel_names:
            cb = QCheckBox(name)
            cb.setChecked(self.default_channel_states[name])
            cb.setFont(QFont("", 8))
            self.chan_flow_layout.addWidget(cb)
            self.chan_checkboxes[name] = cb

        self.chan_scroll.setWidget(self.chan_container)
        chan_layout.addWidget(self.chan_scroll)

        top_row.addWidget(self.chan_panel)
        main_layout.addLayout(top_row)

        # ---------------- Chunk navigation row ----------------
        nav_layout = QHBoxLayout()
        self.btn_prev_chunk = QPushButton("Previous Chunk")
        self.btn_prev_chunk.setFixedHeight(30)
        self.btn_prev_chunk.clicked.connect(self.prev_chunk)

        self.btn_next_chunk = QPushButton("Next Chunk")
        self.btn_next_chunk.setFixedHeight(30)
        self.btn_next_chunk.clicked.connect(self.next_chunk)

        self.lbl_chunk_info = QLabel("No chunk selected")
        self.lbl_chunk_info.setFont(QFont("", 9))

        nav_layout.addWidget(self.btn_prev_chunk)
        nav_layout.addWidget(self.btn_next_chunk)
        nav_layout.addWidget(self.lbl_chunk_info)
        main_layout.addLayout(nav_layout)

        # ---------------- Save all chunks ----------------
        save_all_layout = QHBoxLayout()
        self.btn_save_all = QPushButton("Save All Chunks")
        self.btn_save_all.setFixedHeight(30)
        self.btn_save_all.clicked.connect(self.save_all_results)
        save_all_layout.addWidget(self.btn_save_all)
        main_layout.addLayout(save_all_layout)

        # ---------------- Run analysis ----------------
        self.btn_start_analysis = QPushButton("Start FODN Analysis")
        self.btn_start_analysis.setMaximumHeight(30)
        self.btn_start_analysis.setFont(QFont("", 9))
        self.btn_start_analysis.clicked.connect(self.run_analysis)
        main_layout.addWidget(self.btn_start_analysis)

        # ---------------- Plot grid (up to 4 plot cells) ----------------
        self.plot_grid = QGridLayout()
        self.plot_grid.setSpacing(5)
        main_layout.addLayout(self.plot_grid)

        # Buttons to add different plot types
        options_panel = QGroupBox("Plot Options")
        options_layout = QHBoxLayout(options_panel)
        options_layout.setSpacing(5)

        self.btn_alpha_scatter = QPushButton("Alpha Scatter")
        self.btn_alpha_scatter.setMaximumHeight(30)
        self.btn_alpha_scatter.clicked.connect(lambda: self.add_plot_cell("Alpha Scatter"))

        self.btn_alpha_heatmap = QPushButton("Alpha Heatmap")
        self.btn_alpha_heatmap.setMaximumHeight(30)
        self.btn_alpha_heatmap.clicked.connect(lambda: self.add_plot_cell("Alpha Heatmap"))

        self.btn_alpha_box = QPushButton("Alpha Box Plot")
        self.btn_alpha_box.setMaximumHeight(30)
        self.btn_alpha_box.clicked.connect(lambda: self.add_plot_cell("Alpha Box Plot"))

        self.btn_alpha_dist = QPushButton("Alpha Distribution")
        self.btn_alpha_dist.setMaximumHeight(30)
        self.btn_alpha_dist.clicked.connect(lambda: self.add_plot_cell("Alpha Distribution"))

        self.btn_coupling_heatmap = QPushButton("Coupling Heatmap")
        self.btn_coupling_heatmap.setMaximumHeight(30)
        self.btn_coupling_heatmap.clicked.connect(lambda: self.add_plot_cell("Coupling Heatmap"))

        self.btn_coupling_3d = QPushButton("Coupling 3D Plot")
        self.btn_coupling_3d.setMaximumHeight(30)
        self.btn_coupling_3d.clicked.connect(lambda: self.add_plot_cell("Coupling 3D Plot"))

        options_layout.addWidget(self.btn_alpha_scatter)
        options_layout.addWidget(self.btn_alpha_heatmap)
        options_layout.addWidget(self.btn_alpha_box)
        options_layout.addWidget(self.btn_alpha_dist)
        options_layout.addWidget(self.btn_coupling_heatmap)
        options_layout.addWidget(self.btn_coupling_3d)

        main_layout.addWidget(options_panel)

        # Status label at bottom
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("", 9))
        main_layout.addWidget(self.status_label)

        # ---------------- Global plotting tools across folders ----------------
        box_eig_layout = QHBoxLayout()

        self.btn_box_plot = QPushButton("Generate Box Plot (All Folders)")
        self.btn_box_plot.setFixedHeight(30)
        self.btn_box_plot.clicked.connect(self.handle_box_plot)

        self.btn_eigenvector_heatmap = QPushButton("Generate Eigenvector Heatmap (All Folders)")
        self.btn_eigenvector_heatmap.setFixedHeight(30)
        self.btn_eigenvector_heatmap.clicked.connect(self.handle_eigenvector_heatmap)

        box_eig_layout.addWidget(self.btn_box_plot)
        box_eig_layout.addWidget(self.btn_eigenvector_heatmap)

        self.layout().addLayout(box_eig_layout)

    # --------------------------- Analysis runner ---------------------------

    def run_analysis(self):
        """
        Run FODN analysis over each labeled segment:
          - convert segment start/end (seconds) into sample indices
          - run_fodn_analysis() produces a list of chunk results
          - store all chunks into self.fodn_results
        """
        # Read parameters from UI
        try:
            self.chunk_size = float(self.edit_chunk.text())
            self.numFract = int(self.edit_numfract.text())
            self.niter = int(self.edit_niter.text())
            self.lambdaUse = float(self.edit_lambda.text())
        except ValueError:
            QMessageBox.warning(self, "Parameter Error", "Please enter valid parameter values.")
            return

        # Determine which channels are selected
        selected_channel_indices = []
        for i, name in enumerate(self.all_channel_names):
            if self.chan_checkboxes[name].isChecked():
                selected_channel_indices.append(i)

        if not selected_channel_indices:
            QMessageBox.warning(self, "Channel Selection", "Please select at least one channel.")
            return

        self.selected_channel_indices = selected_channel_indices

        # Progress dialog for segment-level analysis
        total_segments = len(self.labeled_times) - 1
        progress_dialog = QProgressDialog("Running FODN Analysis...", "Cancel", 0, total_segments, self)
        progress_dialog.setWindowTitle("FODN Analysis Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()

        all_segment_results = []

        # Loop over each labeled segment boundary pair
        for i in range(total_segments):
            if progress_dialog.wasCanceled():
                self.status_label.setText("FODN Analysis canceled.")
                return

            seg_label = f"{self.labeled_times[i][0]}-{self.labeled_times[i+1][0]}"
            seg_start = self.labeled_times[i][1]
            seg_end = self.labeled_times[i+1][1]

            # Convert seconds -> sample indices
            start_idx = int(seg_start * self.fs)
            end_idx = int(seg_end * self.fs)
            if end_idx <= start_idx:
                continue

            # Extract signal segment (selected channels only)
            data_segment = self.signals[selected_channel_indices, start_idx:end_idx]

            try:
                # Run analysis (returns list of chunk dicts)
                segment_results = run_fodn_analysis(
                    data_segment,
                    chunk_size=self.chunk_size,
                    numFract=self.numFract,
                    niter=self.niter,
                    lambdaUse=self.lambdaUse,
                    base_time=seg_start  # used so chunk times are absolute in original recording
                )

                # Tag each chunk with segment label
                for res in segment_results:
                    res["segment_label"] = seg_label

                all_segment_results.append(segment_results)

            except Exception as e:
                QMessageBox.critical(self, "FODN Analysis Error", str(e))
                self.status_label.setText("FODN Analysis failed.")
                progress_dialog.close()
                return

            progress_dialog.setValue(i + 1)

        progress_dialog.close()

        # Flatten segment results into one list of chunks
        self.fodn_results = [res for seg in all_segment_results for res in seg]

        if len(self.fodn_results) == 0:
            QMessageBox.warning(self, "No Chunks", "No complete chunks were processed.")
            return

        # Start at first chunk and update UI
        self.current_chunk_index = 0
        self.update_chunk_info()
        self.status_label.setText("FODN Analysis completed successfully.")

    # --------------------------- Chunk navigation ---------------------------

    def update_chunk_info(self):
        """Update chunk label in the navigation row"""
        current = self.fodn_results[self.current_chunk_index]
        self.lbl_chunk_info.setText(
            f"Chunk {self.current_chunk_index+1}/{len(self.fodn_results)}: "
            f"{current['chunk_start']:.2f} - {current['chunk_end']:.2f} sec"
        )

    def next_chunk(self):
        """Move to next chunk and refresh plots"""
        if self.fodn_results is None or self.current_chunk_index >= len(self.fodn_results) - 1:
            return
        self.current_chunk_index += 1
        self.update_chunk_info()
        self.refresh_all_plots()

    def prev_chunk(self):
        """Move to previous chunk and refresh plots"""
        if self.fodn_results is None or self.current_chunk_index <= 0:
            return
        self.current_chunk_index -= 1
        self.update_chunk_info()
        self.refresh_all_plots()

    def refresh_all_plots(self):
        """
        Re-create each active plot cell using the new chunk index.
        This keeps the same plot types open but updates their data.
        """
        for i in reversed(range(self.plot_grid.count())):
            widget = self.plot_grid.itemAt(i).widget()
            if isinstance(widget, PlotCellWidget) and widget.plot_type:
                new_canvas = self.generate_plot_canvas(widget.plot_type)
                if new_canvas is not None:
                    widget.setCanvas(new_canvas)

    # --------------------------- Plot cell management ---------------------------

    def add_plot_cell(self, plot_type):
        """
        Add a plot cell to the grid (max 4 at once).
        Each cell remembers its plot_type so it can refresh on chunk change.
        """
        current_cells = self.plot_grid.count()
        if current_cells >= 4:
            QMessageBox.warning(self, "Too Many Plots", "Please close an existing plot cell before adding a new one.")
            return

        cell = PlotCellWidget()
        cell.plot_type = plot_type

        canvas = self.generate_plot_canvas(plot_type)
        if canvas is None:
            return

        cell.setCanvas(canvas)

        # Connect buttons for this cell
        cell.btn_save_png.clicked.connect(lambda: self.save_plot(canvas, plot_type, "png"))
        cell.btn_save_csv.clicked.connect(lambda: self.save_plot(canvas, plot_type, "csv", data=True))
        cell.btn_close.clicked.connect(lambda: self.close_plot_cell(cell))

        # Place in 2x2 grid
        row = current_cells // 2
        col = current_cells % 2
        self.plot_grid.addWidget(cell, row, col)

    def generate_plot_canvas(self, plot_type):
        """
        Create a Matplotlib canvas for the current chunk.
        Supported plot types:
          Alpha Scatter / Heatmap / Box / Distribution
          Coupling Heatmap / 3D surface
        """
        if self.fodn_results is None:
            QMessageBox.warning(self, "No Analysis Data", "Please run FODN Analysis first.")
            return None

        current = self.fodn_results[self.current_chunk_index]
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # ----- Alpha plots -----
        if plot_type.startswith("Alpha"):
            alpha_vals = current.get("alpha", None)
            if alpha_vals is None:
                QMessageBox.warning(self, "Missing Data", "Alpha values not available.")
                return None

            channels = np.arange(len(alpha_vals))

            # Use selected channel names if available (keeps labels aligned)
            if self.selected_channel_indices is not None:
                labels = [self.all_channel_names[i] for i in self.selected_channel_indices]
            else:
                labels = self.all_channel_names[:len(alpha_vals)]

            if plot_type == "Alpha Scatter":
                ax.scatter(channels, alpha_vals, color=colors[0])
                ax.set_xlabel("Channel")
                ax.set_ylabel("Alpha Value")
                ax.set_title("Alpha Scatter Plot")

            elif plot_type == "Alpha Heatmap":
                hm_data = np.reshape(alpha_vals, (1, -1))
                im = ax.imshow(hm_data, aspect='auto', cmap='viridis')
                ax.set_yticks([])
                ax.set_xticks(channels)
                ax.set_xticklabels(labels, rotation=90, fontsize=4)
                ax.set_title("Alpha Heatmap")
                fig.colorbar(im, ax=ax)

            elif plot_type == "Alpha Box Plot":
                ax.boxplot(alpha_vals, vert=True)
                ax.set_xticklabels(["Alpha"])
                ax.set_title("Alpha Box Plot")

            elif plot_type == "Alpha Distribution":
                ax.hist(alpha_vals, bins=20, color=colors[1])
                ax.set_xlabel("Alpha Value")
                ax.set_ylabel("Frequency")
                ax.set_title("Alpha Distribution")

            else:
                QMessageBox.warning(self, "Unknown Plot", f"Plot type {plot_type} not recognized for Alpha plots.")
                return None

        # ----- Coupling plots -----
        elif plot_type.startswith("Coupling"):
            coupling_matrix = current.get("coupling_matrix", None)
            if coupling_matrix is None:
                QMessageBox.warning(self, "Missing Data", "Coupling matrix not available.")
                return None

            if plot_type == "Coupling Heatmap":
                im = ax.imshow(coupling_matrix, cmap='hot', aspect='auto')
                ax.set_title("Coupling Matrix Heatmap")
                fig.colorbar(im, ax=ax)

            elif plot_type == "Coupling 3D Plot":
                # Replace axes with 3D projection
                ax = fig.add_subplot(111, projection='3d')
                X, Y = np.meshgrid(
                    np.arange(coupling_matrix.shape[0]),
                    np.arange(coupling_matrix.shape[1])
                )
                ax.plot_surface(X, Y, coupling_matrix, cmap='viridis')
                ax.set_title("Coupling Matrix 3D Plot")

            else:
                QMessageBox.warning(self, "Unknown Plot", f"Plot type {plot_type} not recognized for Coupling plots.")
                return None

        else:
            QMessageBox.warning(self, "Unknown Plot", f"Plot type {plot_type} not recognized.")
            return None

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return canvas

    def save_plot(self, canvas, plot_type, fmt, data=False):
        """
        Save either:
          - the plot image (PNG/JPG) OR
          - the raw numeric data (CSV)
        """
        import time
        suggested_name = f"{plot_type}_{int(time.time())}.{fmt}"
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", suggested_name,
            f"{fmt.upper()} Files (*.{fmt});;All Files (*)"
        )
        if fileName:
            if data:
                # Save raw data for current chunk
                if plot_type.startswith("Alpha"):
                    np.savetxt(fileName, self.fodn_results[self.current_chunk_index].get("alpha", np.array([])), delimiter=",")
                elif plot_type.startswith("Coupling"):
                    np.savetxt(fileName, self.fodn_results[self.current_chunk_index].get("coupling_matrix", np.array([])), delimiter=",")
            else:
                # Save figure image
                canvas.figure.savefig(fileName)

    def close_plot_cell(self, cell):
        """Remove a plot cell from the grid"""
        cell.setParent(None)
        cell.deleteLater()

    # --------------------------- Save all results export ---------------------------

    def save_all_results(self):
        """
        Export all chunk outputs into a folder tree:
          Main folder: FODN_<start>-<end>_c<chunk>s
            Segment folders: FODN_<segStart>-<segEnd>_c<chunk>s
              Chunk folders: chunkN_<t0>-<t1>/
                Alpha_Data.csv
                Coupling_Data.csv
                *.png plots
            plus overall_parameters_and_channels.txt
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save All Results", "")
        if not folder:
            return

        # Main output folder name uses full interval + chunk size
        default_folder = f"FODN_{int(self.start_time)}-{int(self.end_time)}_c{self.chunk_size}s"
        main_folder = os.path.join(folder, default_folder)
        os.makedirs(main_folder, exist_ok=True)

        # Determine channels used/not used for documentation
        if self.selected_channel_indices is not None:
            channels_used = [self.all_channel_names[i] for i in self.selected_channel_indices]
            channels_not_selected = [
                name for i, name in enumerate(self.all_channel_names)
                if i not in self.selected_channel_indices
            ]
        else:
            channels_used = list(self.all_channel_names)
            channels_not_selected = []

        # Write overall parameters summary
        overall_param_text = (
            "Overall FODN Parameters:\n"
            f"Chunk Size: {self.edit_chunk.text().strip()} sec\n"
            f"numFract: {self.edit_numfract.text().strip()}\n"
            f"niter: {self.edit_niter.text().strip()}\n"
            f"lambdaUse: {self.edit_lambda.text().strip()}\n\n"
            "Channels used:\n" +
            ("\n".join(channels_used) if channels_used else "None") + "\n\n"
            "Channels not selected:\n" +
            ("\n".join(channels_not_selected) if channels_not_selected else "None") + "\n"
        )
        with open(os.path.join(main_folder, "overall_parameters_and_channels.txt"), "w") as f:
            f.write(overall_param_text)

        # Group chunks by segment_label (assigned in run_analysis)
        segments = {}
        for chunk in self.fodn_results:
            seg = chunk.get("segment_label", "Unknown")
            segments.setdefault(seg, []).append(chunk)

        # Create folders for each segment
        for seg_label, chunk_list in segments.items():
            chunk_list.sort(key=lambda c: c["chunk_start"])

            seg_start = chunk_list[0]["chunk_start"]
            seg_end = chunk_list[-1]["chunk_end"]

            segment_folder_name = f"FODN_{seg_start:.2f}-{seg_end:.2f}_c{self.chunk_size}s"
            segment_folder = os.path.join(main_folder, segment_folder_name)
            os.makedirs(segment_folder, exist_ok=True)

            # Human-readable segment label text file
            seg_parts = seg_label.split("-")
            name1 = seg_parts[0].strip() if len(seg_parts) > 0 else ""
            name2 = seg_parts[1].strip() if len(seg_parts) > 1 else ""
            if name1.lower().startswith("custom"):
                name1 = "custom"
            if name2.lower().startswith("custom"):
                name2 = "custom"
            segment_names_str = f"{name1} - {name2}"

            param_text = (
                "FODN Parameters:\n"
                f"Chunk Size: {self.edit_chunk.text().strip()} sec\n"
                f"numFract: {self.edit_numfract.text().strip()}\n"
                f"niter: {self.edit_niter.text().strip()}\n"
                f"lambdaUse: {self.edit_lambda.text().strip()}\n\n"
                "Channels not selected:\n" + "\n".join(channels_not_selected) + "\n\n"
                "Segment Names:\n" + segment_names_str + "\n\n"
                "Segment Time Interval:\n"
                f"{seg_start:.2f} - {seg_end:.2f} seconds\n"
            )
            with open(os.path.join(segment_folder, "parameters_and_channels.txt"), "w") as f:
                f.write(param_text)

            # Plot types to export for each chunk
            plot_types = [
                "Alpha Scatter", "Alpha Heatmap", "Alpha Box Plot", "Alpha Distribution",
                "Coupling Heatmap", "Coupling 3D Plot"
            ]

            # Save each chunk’s data + plots
            for idx, chunk in enumerate(chunk_list):
                chunk_folder = os.path.join(
                    segment_folder,
                    f"chunk{idx+1}_{chunk['chunk_start']:.2f}-{chunk['chunk_end']:.2f}"
                )
                os.makedirs(chunk_folder, exist_ok=True)

                # Save raw arrays
                np.savetxt(os.path.join(chunk_folder, "Alpha_Data.csv"),
                           chunk.get("alpha", np.array([])), delimiter=",")
                np.savetxt(os.path.join(chunk_folder, "Coupling_Data.csv"),
                           chunk.get("coupling_matrix", np.array([])), delimiter=",")

                # Save plots (note: generate_plot_canvas uses current chunk index;
                # here we reuse plotting logic but you could also generate directly from `chunk`)
                for pt in plot_types:
                    try:
                        canvas = self.generate_plot_canvas(pt)
                        if canvas is None:
                            continue
                        png_path = os.path.join(chunk_folder, f"{pt.replace(' ', '_')}.png")
                        canvas.figure.savefig(png_path)
                        plt.close(canvas.figure)
                    except Exception as e:
                        print(f"Error saving plot for chunk {idx+1}, plot type {pt}: {e}")

        self.status_label.setText(f"All results saved to: {main_folder}")

    # --------------------------- Global (across folders) utilities ---------------------------

    def handle_box_plot(self):
        """
        Build a global alpha boxplot across many saved FODN folders:
          - user selects a folder
          - we find all FODN_*_c*s subfolders recursively
          - load Alpha_Data.csv from each chunk in chronological order
          - display one big boxplot (x-axis = window index)
        """
        dirs = QFileDialog.getExistingDirectory(self, "Select Folder with FODN subfolders", "")
        if not dirs:
            return

        fodn_folders = self.find_fodn_folders(dirs)
        if not fodn_folders:
            QMessageBox.warning(self, "No FODN Folders", "No FODN_*_c*s folders found.")
            return

        # Parse folder names to sort by time
        fodn_folders_info = []
        pattern = r"FODN_(\d+(\.\d+)?)-(\d+(\.\d+)?)_c(\d+(\.\d+)?)s"
        for f in fodn_folders:
            name = os.path.basename(f)
            match = re.match(pattern, name)
            if match:
                start_str, _, end_str, _, chunk_str, _ = match.groups()
                fodn_folders_info.append((float(start_str), float(end_str), float(chunk_str), f))

        if not fodn_folders_info:
            QMessageBox.warning(self, "No FODN Folders", "No matching subfolders found.")
            return

        fodn_folders_info.sort(key=lambda x: (x[0], x[1]))

        alpha_data = []

        # Load alpha arrays from each chunk folder
        for (_, _, _, folder_path) in fodn_folders_info:
            chunk_subfolders = [
                os.path.join(folder_path, d) for d in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("chunk")
            ]

            chunk_subfolders_info = []
            chunk_pattern = r"chunk(\d+)_(\d+(\.\d+)?)-(\d+(\.\d+)?)"
            for csub in chunk_subfolders:
                cname = os.path.basename(csub)
                m = re.match(chunk_pattern, cname)
                if m:
                    _, cstart_str, _, cend_str, _ = m.groups()
                    chunk_subfolders_info.append((float(cstart_str), float(cend_str), csub))

            chunk_subfolders_info.sort(key=lambda x: (x[0], x[1]))

            for (_, _, chunk_path) in chunk_subfolders_info:
                alpha_file = os.path.join(chunk_path, "Alpha_Data.csv")
                if os.path.exists(alpha_file):
                    arr = pd.read_csv(alpha_file, header=None).values.flatten()
                    alpha_data.append(arr)

        if not alpha_data:
            QMessageBox.warning(self, "No Alpha Data", "No Alpha_Data.csv files found.")
            return

        # Display in a new plot cell
        cell = PlotCellWidget()
        cell.plot_type = "Global Box Plot"

        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.boxplot(alpha_data, vert=True, patch_artist=True)

        x_labels = np.arange(1, len(alpha_data) + 1)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels, rotation=0)

        ax.set_xlabel("Window Index")
        ax.set_ylabel("Alpha Values")
        ax.set_title("Alpha Box Plot (All Folders)")
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cell.setCanvas(canvas)

        # Save handlers for global plot
        cell.btn_save_png.clicked.connect(lambda: self.save_custom_plot(canvas, "global_box_plot", "png"))
        cell.btn_save_csv.clicked.connect(lambda: self.save_custom_plot(canvas, "global_box_plot", "csv", data=alpha_data))
        cell.btn_close.clicked.connect(lambda: self.close_plot_cell(cell))

        row = self.plot_grid.rowCount()
        self.plot_grid.addWidget(cell, row, 0, 1, 2)  # span 2 columns

    def handle_eigenvector_heatmap(self):
        """
        Build a global heatmap of dominant eigenvectors across saved chunks:
          - loads Coupling_Data.csv per chunk
          - computes dominant eigenvector for each matrix
          - stacks vectors into a matrix and visualizes it as a heatmap
        """
        dirs = QFileDialog.getExistingDirectory(self, "Select Folder with FODN subfolders", "")
        if not dirs:
            return

        fodn_folders = self.find_fodn_folders(dirs)
        if not fodn_folders:
            QMessageBox.warning(self, "No FODN Folders", "No FODN_*_c*s folders found.")
            return

        # Parse folder names to sort
        pattern = r"FODN_(\d+(\.\d+)?)-(\d+(\.\d+)?)_c(\d+(\.\d+)?)s"
        fodn_folders_info = []
        for f in fodn_folders:
            name = os.path.basename(f)
            match = re.match(pattern, name)
            if match:
                start_str, _, end_str, _, chunk_str, _ = match.groups()
                fodn_folders_info.append((float(start_str), float(end_str), float(chunk_str), f))
        fodn_folders_info.sort(key=lambda x: (x[0], x[1]))

        all_eigenvectors = []

        # Walk chunks and compute eigenvectors
        for (_, _, _, folder_path) in fodn_folders_info:
            chunk_subfolders = [
                os.path.join(folder_path, d) for d in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("chunk")
            ]

            chunk_subfolders_info = []
            chunk_pattern = r"chunk(\d+)_(\d+(\.\d+)?)-(\d+(\.\d+)?)"
            for csub in chunk_subfolders:
                cname = os.path.basename(csub)
                m = re.match(chunk_pattern, cname)
                if m:
                    _, cstart_str, _, cend_str, _ = m.groups()
                    chunk_subfolders_info.append((float(cstart_str), float(cend_str), csub))
            chunk_subfolders_info.sort(key=lambda x: (x[0], x[1]))

            for (_, _, chunk_path) in chunk_subfolders_info:
                coupling_file = os.path.join(chunk_path, "Coupling_Data.csv")
                if os.path.exists(coupling_file):
                    matrix = pd.read_csv(coupling_file, header=None).values
                    dominant_vec = self.compute_dominant_eigenvector(matrix)
                    all_eigenvectors.append(dominant_vec)

        if not all_eigenvectors:
            QMessageBox.warning(self, "No Coupling Data", "No Coupling_Data.csv found.")
            return

        # Stack eigenvectors into a matrix: rows=components, cols=windows
        ev_matrix = np.column_stack(all_eigenvectors)

        cell = PlotCellWidget()
        cell.plot_type = "Global Eigen Heatmap"

        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(ev_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        fig.colorbar(im, ax=ax)

        ax.set_title("Dominant Eigenvector Heatmap (All Folders)")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Eigenvector Component Index")

        x_labels = np.arange(1, ev_matrix.shape[1] + 1)
        ax.set_xticks(x_labels - 1)
        ax.set_xticklabels(x_labels, rotation=0, fontsize=7)

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cell.setCanvas(canvas)

        # Save handlers
        cell.btn_save_png.clicked.connect(lambda: self.save_custom_plot(canvas, "global_eigenvector_heatmap", "png"))
        cell.btn_save_csv.clicked.connect(lambda: self.save_custom_plot(canvas, "global_eigenvector_heatmap", "csv", data=ev_matrix))
        cell.btn_close.clicked.connect(lambda: self.close_plot_cell(cell))

        row = self.plot_grid.rowCount()
        self.plot_grid.addWidget(cell, row, 0, 1, 2)

    def save_custom_plot(self, canvas, base_name, fmt, data=None):
        """
        Save helper for global plots:
          - png: saves figure
          - csv: saves numeric matrix/array
        """
        import time
        suggested_name = f"{base_name}_{int(time.time())}.{fmt}"
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", suggested_name,
            f"{fmt.upper()} Files (*.{fmt});;All Files (*)"
        )
        if fileName:
            if fmt == "png":
                canvas.figure.savefig(fileName)
            elif fmt == "csv" and data is not None:
                np.savetxt(fileName, data, delimiter=",")

    def compute_dominant_eigenvector(self, matrix):
        """Compute eigenvector corresponding to the largest eigenvalue (dominant mode)"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        idx_max = np.argmax(eigenvalues)
        return eigenvectors[:, idx_max].real

    def find_fodn_folders(self, start_folder):
        """
        Recursively search for folders matching:
            FODN_<start>-<end>_c<chunk>s
        Returns list of matching folder paths.
        """
        results = []
        pattern = r"^FODN_\d+(\.\d+)?-\d+(\.\d+)?_c\d+(\.\d+)?s$"
        for root, dirs, files in os.walk(start_folder):
            for d in dirs:
                if re.match(pattern, d):
                    results.append(os.path.join(root, d))
        return results


# --------------------------- Test harness ---------------------------

if __name__ == "__main__":
    # Standalone test with fake data
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    fake_signals = np.random.randn(64, 60 * 1000)  # 64 channels, 60 seconds @ 1000Hz
    fake_time = np.linspace(0, 60, 60 * 1000)

    fake_channel_names = [f"Channel {i}" for i in range(64)]
    for i, name in enumerate(fake_channel_names):
        if i in [1, 2, 3, 4, 5, 6]:
            fake_channel_names[i] = "EKG1" if i == 1 else ("EKG2" if i == 2 else f"X1 DC{i-2}")

    win = FODNAnalysisWindow(None, fake_signals, fake_time, 10, 20, fake_channel_names)

    # Extra save button for quick testing
    btn_save_all = QPushButton("Save All Results", win)
    btn_save_all.clicked.connect(win.save_all_results)
    win.layout().addWidget(btn_save_all)

    win.show()
    sys.exit(app.exec_())
