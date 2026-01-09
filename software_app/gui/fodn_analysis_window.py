# file: gui/fodn_analysis_window.py

import sys
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QProgressDialog,
    QScrollArea, QWidget, QSizePolicy, QGroupBox, QLineEdit, QCheckBox, QToolButton, QFileDialog, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utils.fodn_utils import run_fodn_analysis

# Collapsible Panel widget
class CollapsiblePanel(QGroupBox):
    def __init__(self, title="Parameters", parent=None):
        super().__init__(parent)
        self.setTitle("")
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; padding: 2px; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content = QWidget()
        self.content.setVisible(False)
        self.content.setContentsMargins(0, 0, 0, 0)

        lay = QHBoxLayout()
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        self.setLayout(lay)
        self.setMaximumHeight(self.toggle_button.sizeHint().height())

    def on_toggle(self):
        checked = self.toggle_button.isChecked()
        self.content.setVisible(checked)
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        if checked:
            self.setMaximumHeight(16777215)
        else:
            self.setMaximumHeight(self.toggle_button.sizeHint().height())

# PlotCellWidget stores the canvas and its plot type.
class PlotCellWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.canvas = None
        self.plot_type = None  # Store the plot type for refresh
        
        # Create Save, Save CSV, and Close buttons.
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
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
        self.canvas = canvas
        self.layout.insertWidget(0, self.canvas)

# FODNAnalysisWindow now supports chunk navigation and "Save All" functionality.
class FODNAnalysisWindow(QDialog):
    def __init__(self, parent, signals, time_array, start_time, end_time, channel_names, labeled_times=None):
        """
        signals: 2D numpy array (channels, timepoints)
        time_array: 1D numpy array of time stamps (seconds)
        start_time, end_time: floats defining the selected time window from the previous window
        channel_names: list of channel names from the H5 file
        labeled_times: Optional list of tuples (label, time) for internal segment boundaries.
                       If provided, these should be in ascending order and within [start_time, end_time].
                       If not provided, defaults to [('Custom_Start', start_time), ('Custom_End', end_time)]
        """
        super().__init__(parent)
        self.setWindowTitle("FODN Analysis Window")
        self.setWindowState(Qt.WindowMaximized)
        self.signals = signals
        self.time_array = time_array
        self.fs = 1000
        self.start_time = start_time
        self.end_time = end_time
        self.all_channel_names = channel_names

        self.default_channel_states = {}
        unwanted = ["EKG1", "EKG2", "X1 DC1", "X1 DC2", "X1 DC3", "X1 DC4"]
        for name in self.all_channel_names:
            self.default_channel_states[name] = (False if name in unwanted else True)

        # Analysis parameters:
        self.chunk_size = 0.5
        self.numFract = 165  # 165 is largest J array length without numerical error issues
        self.niter = 3
        self.lambdaUse = 0.5

        # List to store chunk analysis results.
        self.fodn_results = []
        self.current_chunk_index = 0

        self.selected_channel_indices = None

        # Process labeled_times
        if labeled_times is None or len(labeled_times) == 0:
            self.labeled_times = [("Custom_Start", start_time), ("Custom_End", end_time)]
        else:
            # Sort the labeled times by time and filter to the user-defined interval.
            sorted_times = sorted(labeled_times, key=lambda x: x[1])
            self.labeled_times = [ (label, t) for label, t in sorted_times if t >= start_time and t <= end_time ]
            # Ensure boundaries are included.
            if not self.labeled_times or self.labeled_times[0][1] > start_time:
                self.labeled_times.insert(0, ("Custom_Start", start_time))
            if self.labeled_times[-1][1] < end_time:
                self.labeled_times.append(("Custom_End", end_time))

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Top row: Parameter panel and channel selection.
        top_row = QHBoxLayout()
        top_row.setSpacing(5)
        self.param_panel = CollapsiblePanel("FODN Parameters")
        param_layout = QHBoxLayout(self.param_panel.content)
        param_layout.setContentsMargins(0, 0, 0, 0)
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
        for widget in [lbl_chunk, self.edit_chunk, lbl_numfract, self.edit_numfract, lbl_niter, self.edit_niter, lbl_lambda, self.edit_lambda]:
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

        self.chan_panel = CollapsiblePanel("Select Channels")
        chan_layout = QHBoxLayout(self.chan_panel.content)
        chan_layout.setContentsMargins(0, 0, 0, 0)
        self.chan_scroll = QScrollArea()
        self.chan_scroll.setWidgetResizable(True)
        # self.chan_scroll.setFixedHeight(200)
        self.chan_container = QWidget()
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

        # Navigation row: Previous/Next chunk buttons and chunk info.
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

        # New: Save All button row.
        save_all_layout = QHBoxLayout()
        self.btn_save_all = QPushButton("Save All Chunks")
        self.btn_save_all.setFixedHeight(30)
        self.btn_save_all.clicked.connect(self.save_all_results)
        save_all_layout.addWidget(self.btn_save_all)
        main_layout.addLayout(save_all_layout)

        # Start analysis button.
        self.btn_start_analysis = QPushButton("Start FODN Analysis")
        self.btn_start_analysis.setMaximumHeight(30)
        self.btn_start_analysis.setFont(QFont("", 9))
        self.btn_start_analysis.clicked.connect(self.run_analysis)
        main_layout.addWidget(self.btn_start_analysis)

        # Plot grid for individual plot cells.
        self.plot_grid = QGridLayout()
        self.plot_grid.setSpacing(5)
        main_layout.addLayout(self.plot_grid)

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

        self.status_label = QLabel("")
        self.status_label.setFont(QFont("", 9))
        main_layout.addWidget(self.status_label)
#-------------------------------
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
#-------------------------------
    def run_analysis(self):
        try:
            self.chunk_size = float(self.edit_chunk.text())
            self.numFract = int(self.edit_numfract.text())
            self.niter = int(self.edit_niter.text())
            self.lambdaUse = float(self.edit_lambda.text())
        except ValueError:
            QMessageBox.warning(self, "Parameter Error", "Please enter valid parameter values.")
            return

        selected_channel_indices = []
        for i, name in enumerate(self.all_channel_names):
            if self.chan_checkboxes[name].isChecked():
                selected_channel_indices.append(i)
        if not selected_channel_indices:
            QMessageBox.warning(self, "Channel Selection", "Please select at least one channel.")
            return
        self.selected_channel_indices = selected_channel_indices

        # Prepare a progress dialog
        total_segments = len(self.labeled_times) - 1
        progress_dialog = QProgressDialog("Running FODN Analysis...", "Cancel", 0, total_segments, self)
        progress_dialog.setWindowTitle("FODN Analysis Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()

        all_segment_results = []
        for i in range(total_segments):
            if progress_dialog.wasCanceled():
                self.status_label.setText("FODN Analysis canceled.")
                return
            seg_label = f"{self.labeled_times[i][0]}-{self.labeled_times[i+1][0]}"
            seg_start = self.labeled_times[i][1]
            seg_end = self.labeled_times[i+1][1]
            start_idx = int(seg_start * self.fs)
            end_idx = int(seg_end * self.fs)
            if end_idx <= start_idx:
                continue
            data_segment = self.signals[selected_channel_indices, start_idx:end_idx]
            try:
                segment_results = run_fodn_analysis(
                    data_segment,
                    chunk_size=self.chunk_size,
                    numFract=self.numFract,
                    niter=self.niter,
                    lambdaUse=self.lambdaUse,
                    base_time=seg_start
                )
                for res in segment_results:
                    res["segment_label"] = seg_label
                all_segment_results.append(segment_results)
            except Exception as e:
                QMessageBox.critical(self, "FODN Analysis Error", str(e))
                self.status_label.setText("FODN Analysis failed.")
                progress_dialog.close()
                return
            progress_dialog.setValue(i+1)

        progress_dialog.close()

        self.fodn_results = [res for seg in all_segment_results for res in seg]
        if len(self.fodn_results) == 0:
            QMessageBox.warning(self, "No Chunks", "No complete chunks were processed.")
            return
        self.current_chunk_index = 0
        self.update_chunk_info()
        self.status_label.setText("FODN Analysis completed successfully.")


    def update_chunk_info(self):
        current = self.fodn_results[self.current_chunk_index]
        self.lbl_chunk_info.setText(f"Chunk {self.current_chunk_index+1}/{len(self.fodn_results)}: {current['chunk_start']:.2f} - {current['chunk_end']:.2f} sec")

    def next_chunk(self):
        if self.fodn_results is None or self.current_chunk_index >= len(self.fodn_results) - 1:
            return
        self.current_chunk_index += 1
        self.update_chunk_info()
        self.refresh_all_plots()

    def prev_chunk(self):
        if self.fodn_results is None or self.current_chunk_index <= 0:
            return
        self.current_chunk_index -= 1
        self.update_chunk_info()
        self.refresh_all_plots()

    def refresh_all_plots(self):
        for i in reversed(range(self.plot_grid.count())):
            widget = self.plot_grid.itemAt(i).widget()
            if isinstance(widget, PlotCellWidget) and widget.plot_type:
                new_canvas = self.generate_plot_canvas(widget.plot_type)
                if new_canvas is not None:
                    widget.setCanvas(new_canvas)

    def add_plot_cell(self, plot_type):
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
        cell.btn_save_png.clicked.connect(lambda: self.save_plot(canvas, plot_type, "png"))
        cell.btn_save_csv.clicked.connect(lambda: self.save_plot(canvas, plot_type, "csv", data=True))
        cell.btn_close.clicked.connect(lambda: self.close_plot_cell(cell))
        
        row = current_cells // 2
        col = current_cells % 2
        self.plot_grid.addWidget(cell, row, col)

    def generate_plot_canvas(self, plot_type):
        if self.fodn_results is None:
            QMessageBox.warning(self, "No Analysis Data", "Please run FODN Analysis first.")
            return None
        current = self.fodn_results[self.current_chunk_index]
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if plot_type.startswith("Alpha"):
            alpha_vals = current.get("alpha", None)
            if alpha_vals is None:
                QMessageBox.warning(self, "Missing Data", "Alpha values not available.")
                return None
            channels = np.arange(len(alpha_vals))
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
                from mpl_toolkits.mplot3d import Axes3D
                ax = fig.add_subplot(111, projection='3d')
                X, Y = np.meshgrid(np.arange(coupling_matrix.shape[0]), np.arange(coupling_matrix.shape[1]))
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
        import time
        suggested_name = f"{plot_type}_{int(time.time())}.{fmt}"
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Plot", suggested_name,
                                                  f"{fmt.upper()} Files (*.{fmt});;All Files (*)")
        if fileName:
            if data:
                if plot_type.startswith("Alpha"):
                    np.savetxt(fileName, self.fodn_results[self.current_chunk_index].get("alpha", np.array([])), delimiter=",")
                elif plot_type.startswith("Coupling"):
                    np.savetxt(fileName, self.fodn_results[self.current_chunk_index].get("coupling_matrix", np.array([])), delimiter=",")
            else:
                canvas.figure.savefig(fileName)

    def close_plot_cell(self, cell):
        cell.setParent(None)
        cell.deleteLater()

    def next_chunk(self):
        if self.fodn_results is None or self.current_chunk_index >= len(self.fodn_results) - 1:
            return
        self.current_chunk_index += 1
        self.update_chunk_info()
        self.refresh_all_plots()

    def prev_chunk(self):
        if self.fodn_results is None or self.current_chunk_index <= 0:
            return
        self.current_chunk_index -= 1
        self.update_chunk_info()
        self.refresh_all_plots()

    def update_chunk_info(self):
        current = self.fodn_results[self.current_chunk_index]
        self.lbl_chunk_info.setText(f"Chunk {self.current_chunk_index+1}/{len(self.fodn_results)}: {current['chunk_start']:.2f} - {current['chunk_end']:.2f} sec")

    def refresh_all_plots(self):
        for i in reversed(range(self.plot_grid.count())):
            widget = self.plot_grid.itemAt(i).widget()
            if isinstance(widget, PlotCellWidget) and widget.plot_type:
                new_canvas = self.generate_plot_canvas(widget.plot_type)
                if new_canvas is not None:
                    widget.setCanvas(new_canvas)

    def save_all_results(self):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save All Results", "")
        if not folder:
            return

        # Create the main folder using overall start/end times and chunk size.
        default_folder = f"FODN_{int(self.start_time)}-{int(self.end_time)}_c{self.chunk_size}s"
        main_folder = os.path.join(folder, default_folder)
        os.makedirs(main_folder, exist_ok=True)

        # ---- NEW: overall parameters + channels file in main_folder ----
        # Determine channels used and not used based on current selection.
        if self.selected_channel_indices is not None:
            channels_used = [self.all_channel_names[i] for i in self.selected_channel_indices]
            channels_not_selected = [
                name for i, name in enumerate(self.all_channel_names)
                if i not in self.selected_channel_indices
            ]
        else:
            # Fallback: if for some reason selection is None, assume all used.
            channels_used = list(self.all_channel_names)
            channels_not_selected = []

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
        # ---- END NEW BLOCK ------------------------------------------------

        # Group chunks by segments using the "segment_label" set during analysis.
        segments = {}
        for chunk in self.fodn_results:
            seg = chunk.get("segment_label", "Unknown")
            if seg not in segments:
                segments[seg] = []
            segments[seg].append(chunk)

        # We already computed channels_not_selected above; reuse it here.
        not_selected = channels_not_selected

        # For each segment, create a segment folder using only the time values.
        for seg_label, chunk_list in segments.items():
            # Sort the chunks within this segment by their start time.
            chunk_list.sort(key=lambda c: c["chunk_start"])
            # Determine overall segment start and end times.
            seg_start = chunk_list[0]["chunk_start"]
            seg_end = chunk_list[-1]["chunk_end"]

            # Create the segment folder name using time only.
            segment_folder_name = f"FODN_{seg_start:.2f}-{seg_end:.2f}_c{self.chunk_size}s"
            segment_folder = os.path.join(main_folder, segment_folder_name)
            os.makedirs(segment_folder, exist_ok=True)

            # Process the segment label into segment names.
            seg_parts = seg_label.split("-")
            name1 = seg_parts[0].strip() if len(seg_parts) > 0 else ""
            name2 = seg_parts[1].strip() if len(seg_parts) > 1 else ""
            if name1.lower().startswith("custom"):
                name1 = "custom"
            if name2.lower().startswith("custom"):
                name2 = "custom"
            segment_names_str = f"{name1} - {name2}"

            # Prepare the parameters and channels text file for this segment.
            param_text = (
                "FODN Parameters:\n"
                f"Chunk Size: {self.edit_chunk.text().strip()} sec\n"
                f"numFract: {self.edit_numfract.text().strip()}\n"
                f"niter: {self.edit_niter.text().strip()}\n"
                f"lambdaUse: {self.edit_lambda.text().strip()}\n\n"
                "Channels not selected:\n" + "\n".join(not_selected) + "\n\n"
                "Segment Names:\n" + segment_names_str + "\n\n"
                "Segment Time Interval:\n"
                f"{seg_start:.2f} - {seg_end:.2f} seconds\n"
            )
            with open(os.path.join(segment_folder, "parameters_and_channels.txt"), "w") as f:
                f.write(param_text)

            # Define plot types to save.
            plot_types = ["Alpha Scatter", "Alpha Heatmap", "Alpha Box Plot", "Alpha Distribution",
                          "Coupling Heatmap", "Coupling 3D Plot"]

            # For each chunk in the segment, create a subfolder and save its files.
            for idx, chunk in enumerate(chunk_list):
                chunk_folder = os.path.join(
                    segment_folder,
                    f"chunk{idx+1}_{chunk['chunk_start']:.2f}-{chunk['chunk_end']:.2f}"
                )
                os.makedirs(chunk_folder, exist_ok=True)

                # Save Alpha data.
                alpha_csv_path = os.path.join(chunk_folder, "Alpha_Data.csv")
                np.savetxt(alpha_csv_path, chunk.get("alpha", np.array([])), delimiter=",")

                # Save Coupling Matrix.
                coupling_csv_path = os.path.join(chunk_folder, "Coupling_Data.csv")
                np.savetxt(coupling_csv_path, chunk.get("coupling_matrix", np.array([])), delimiter=",")

                # Save plots for each plot type.
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


    def save_plot(self, canvas, plot_type, fmt, data=False):
        import time
        suggested_name = f"{plot_type}_{int(time.time())}.{fmt}"
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Plot", suggested_name,
                                                  f"{fmt.upper()} Files (*.{fmt});;All Files (*)")
        if fileName:
            if data:
                if plot_type.startswith("Alpha"):
                    np.savetxt(fileName, self.fodn_results[self.current_chunk_index].get("alpha", np.array([])), delimiter=",")
                elif plot_type.startswith("Coupling"):
                    np.savetxt(fileName, self.fodn_results[self.current_chunk_index].get("coupling_matrix", np.array([])), delimiter=",")
            else:
                canvas.figure.savefig(fileName)

    def close_plot_cell(self, cell):
        cell.setParent(None)
        cell.deleteLater()

    def handle_box_plot(self):
        """
        1) Prompt user for one or multiple folders (currently just 1).
        2) Recursively search them for FODN_*_c*s subfolders.
        3) Gather alpha data from each chunk in chronological order.
        4) Display a box plot in the GUI with x-axis labeled 1..N.
        """
        dirs = QFileDialog.getExistingDirectory(self, "Select Folder with FODN subfolders", "")
        if not dirs:
            return
        selected_dirs = [dirs]
        fodn_folders = []
        for folder in selected_dirs:
            fodn_folders.extend(self.find_fodn_folders(folder))

        if not fodn_folders:
            QMessageBox.warning(self, "No FODN Folders", "No FODN_*_c*s folders found.")
            return

        # parse each folder name: FODN_start-end_cchunks
        fodn_folders_info = []
        pattern = r"FODN_(\d+(\.\d+)?)-(\d+(\.\d+)?)_c(\d+(\.\d+)?)s"
        for f in fodn_folders:
            name = os.path.basename(f)
            match = re.match(pattern, name)
            if match:
                start_str, _, end_str, _, chunk_str, _ = match.groups()
                start_val = float(start_str)
                end_val = float(end_str)
                chunk_val = float(chunk_str)
                fodn_folders_info.append((start_val, end_val, chunk_val, f))

        if not fodn_folders_info:
            QMessageBox.warning(self, "No FODN Folders", "No matching subfolders found.")
            return

        fodn_folders_info.sort(key=lambda x: (x[0], x[1]))

        alpha_data = []

        for (start_val, end_val, chunk_val, folder_path) in fodn_folders_info:
            chunk_subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
                                if os.path.isdir(os.path.join(folder_path, d))
                                and d.startswith("chunk")]
            chunk_subfolders_info = []
            chunk_pattern = r"chunk(\d+)_(\d+(\.\d+)?)-(\d+(\.\d+)?)"
            for csub in chunk_subfolders:
                cname = os.path.basename(csub)
                m = re.match(chunk_pattern, cname)
                if m:
                    _, cstart_str, _, cend_str, _ = m.groups()
                    cstart_val = float(cstart_str)
                    cend_val = float(cend_str)
                    chunk_subfolders_info.append((cstart_val, cend_val, csub))
            chunk_subfolders_info.sort(key=lambda x: (x[0], x[1]))

            for (_, _, chunk_path) in chunk_subfolders_info:
                alpha_file = os.path.join(chunk_path, "Alpha_Data.csv")
                if os.path.exists(alpha_file):
                    arr = pd.read_csv(alpha_file, header=None).values.flatten()
                    alpha_data.append(arr)

        if not alpha_data:
            QMessageBox.warning(self, "No Alpha Data", "No Alpha_Data.csv files found.")
            return

        cell = PlotCellWidget()
        cell.plot_type = "Global Box Plot"

        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.boxplot(alpha_data, vert=True, patch_artist=True)
        x_labels = np.arange(1, len(alpha_data)+1)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Alpha Values")
        ax.set_title("Alpha Box Plot (All Folders)")
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cell.setCanvas(canvas)

        cell.btn_save_png.clicked.connect(lambda: self.save_custom_plot(canvas, "global_box_plot", "png"))
        cell.btn_save_csv.clicked.connect(lambda: self.save_custom_plot(canvas, "global_box_plot", "csv", data=alpha_data))
        cell.btn_close.clicked.connect(lambda: self.close_plot_cell(cell))

        row = self.plot_grid.rowCount()
        col = 0
        self.plot_grid.addWidget(cell, row, col, 1, 2)

    def handle_eigenvector_heatmap(self):
        """
        Same idea but for loading Coupling_Data.csv from each chunk,
        computing the dominant eigenvector, stacking them, and producing a heatmap
        with a simple x-axis labeling (1..N).
        """
        dirs = QFileDialog.getExistingDirectory(self, "Select Folder with FODN subfolders", "")
        if not dirs:
            return
        selected_dirs = [dirs]
        fodn_folders = []
        for folder in selected_dirs:
            fodn_folders.extend(self.find_fodn_folders(folder))

        if not fodn_folders:
            QMessageBox.warning(self, "No FODN Folders", "No FODN_*_c*s folders found.")
            return

        pattern = r"FODN_(\d+(\.\d+)?)-(\d+(\.\d+)?)_c(\d+(\.\d+)?)s"
        fodn_folders_info = []
        for f in fodn_folders:
            name = os.path.basename(f)
            match = re.match(pattern, name)
            if match:
                start_str, _, end_str, _, chunk_str, _ = match.groups()
                start_val = float(start_str)
                end_val = float(end_str)
                chunk_val = float(chunk_str)
                fodn_folders_info.append((start_val, end_val, chunk_val, f))
        fodn_folders_info.sort(key=lambda x: (x[0], x[1]))

        all_eigenvectors = []
        for (start_val, end_val, chunk_val, folder_path) in fodn_folders_info:
            chunk_subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
                                if os.path.isdir(os.path.join(folder_path, d))
                                and d.startswith("chunk")]
            chunk_subfolders_info = []
            chunk_pattern = r"chunk(\d+)_(\d+(\.\d+)?)-(\d+(\.\d+)?)"
            for csub in chunk_subfolders:
                cname = os.path.basename(csub)
                m = re.match(chunk_pattern, cname)
                if m:
                    _, cstart_str, _, cend_str, _ = m.groups()
                    cstart_val = float(cstart_str)
                    cend_val = float(cend_str)
                    chunk_subfolders_info.append((cstart_val, cend_val, csub))
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
        ax.set_xticks(x_labels - 1)  # because index starts at 0
        ax.set_xticklabels(x_labels, rotation=0, fontsize=7)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cell.setCanvas(canvas)

        # Connect save buttons
        cell.btn_save_png.clicked.connect(lambda: self.save_custom_plot(canvas, "global_eigenvector_heatmap", "png"))
        cell.btn_save_csv.clicked.connect(lambda: self.save_custom_plot(canvas, "global_eigenvector_heatmap", "csv", data=ev_matrix))
        cell.btn_close.clicked.connect(lambda: self.close_plot_cell(cell))

        row = self.plot_grid.rowCount()
        col = 0
        self.plot_grid.addWidget(cell, row, col, 1, 2)  # span 2 columns

    def save_custom_plot(self, canvas, base_name, fmt, data=None):
        """
        Save the current figure or the data to the selected filename.
        base_name is a short label for the plot (e.g., 'global_box_plot').
        fmt is 'png' or 'csv'.
        data is optional; if present and fmt=='csv', we save it as a numeric array.
        """
        import time
        suggested_name = f"{base_name}_{int(time.time())}.{fmt}"
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Plot", suggested_name,
                                                f"{fmt.upper()} Files (*.{fmt});;All Files (*)")
        if fileName:
            if fmt == "png":
                canvas.figure.savefig(fileName)
            elif fmt == "csv" and data is not None:
                np.savetxt(fileName, data, delimiter=",")

    def compute_dominant_eigenvector(self, matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        idx_max = np.argmax(eigenvalues)
        return eigenvectors[:, idx_max].real

    def find_fodn_folders(self, start_folder):
        """
        Recursively search 'start_folder' for subfolders that match 'FODN_*_c*s'.
        Return a list of all matching folder paths.
        """
        results = []
        pattern = r"^FODN_\d+(\.\d+)?-\d+(\.\d+)?_c\d+(\.\d+)?s$"
        for root, dirs, files in os.walk(start_folder):
            for d in dirs:
                if re.match(pattern, d):
                    results.append(os.path.join(root, d))
        return results

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QSizePolicy
    app = QApplication(sys.argv)
    # For testing, create some fake data.
    fake_signals = np.random.randn(64, 60 * 1000)
    fake_time = np.linspace(0, 60, 60 * 1000)
    fake_channel_names = [f"Channel {i}" for i in range(64)]
    for i, name in enumerate(fake_channel_names):
        if i in [1, 2, 3, 4, 5, 6]:
            fake_channel_names[i] = "EKG1" if i == 1 else ("EKG2" if i == 2 else f"X1 DC{i-2}")
    win = FODNAnalysisWindow(None, fake_signals, fake_time, 10, 20, fake_channel_names)
    # For testing, add a "Save All" button to the window.
    btn_save_all = QPushButton("Save All Results", win)
    btn_save_all.clicked.connect(win.save_all_results)
    layout = win.layout()
    layout.addWidget(btn_save_all)
    win.show()
    sys.exit(app.exec_())
