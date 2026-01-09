from gui.main_gui import AppGUI as MainAppGUI

class AppGUI(MainAppGUI):
    pass

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = AppGUI()
    window.show()
    sys.exit(app.exec_())
