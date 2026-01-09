"""
main.py

Application entry point for the Seizure Suppression Analysis toolkit.

This script initializes the Qt application and launches the
RouteSelectorWindow, which allows the user to choose between
data processing, feature extraction, DFA/MFDFA analysis,
overlapping MF-DFA, and model training routes.
"""

import sys
from PyQt5.QtWidgets import QApplication

# Top-level routing window (main menu)
from gui.route_selector_window import RouteSelectorWindow


if __name__ == "__main__":
    # Create the Qt application (must be created exactly once)
    app = QApplication(sys.argv)

    # Launch the route-selection window
    window = RouteSelectorWindow()
    window.show()

    # Start the Qt event loop and exit cleanly
    sys.exit(app.exec_())
