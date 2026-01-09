# Entry point for launching the GUI application

from gui.main_gui import AppGUI as MainAppGUI


class AppGUI(MainAppGUI):
    """
    Main application class.
    Inherits all functionality from MainAppGUI.
    """
    pass


if __name__ == '__main__':
    # Required imports for Qt application
    import sys
    from PyQt5.QtWidgets import QApplication

    # Create the Qt application instance
    app = QApplication(sys.argv)

    # Initialize main application window
    window = AppGUI()

    # Display the GUI
    window.show()

    # Start the Qt event loop
    sys.exit(app.exec_())
