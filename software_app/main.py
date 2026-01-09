import sys
from PyQt5.QtWidgets import QApplication
from gui.route_selector_window import RouteSelectorWindow



if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    window = RouteSelectorWindow()
    window.show()
    sys.exit(app.exec_())
    

