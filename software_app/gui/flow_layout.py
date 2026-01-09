# file: gui/flow_layout.py
from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtWidgets import QLayout, QSizePolicy, QWidget

class FlowLayout(QLayout):
    """A custom flow layout that arranges child widgets horizontally, then wraps to a new line as needed."""
    
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self.itemList = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing if spacing >= 0 else self.spacing())
    
    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)
    
    def addItem(self, item):
        self.itemList.append(item)
    
    def count(self):
        return len(self.itemList)
    
    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None
    
    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None
    
    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))
    
    def sizeHint(self):
        return self.minimumSize()
    
    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2*self.spacing(), 2*self.spacing())
        return size
    
    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect)
    
    def doLayout(self, rect, testOnly=False):
        x = 0
        y = 0
        lineHeight = 0
        
        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing()
            spaceY = self.spacing()
            nextX = x + wid.sizeHint().width() + spaceX
            if nextX > rect.width() and x > 0:
                x = 0
                y = y + lineHeight + spaceY
                nextX = wid.sizeHint().width() + spaceX
                lineHeight = 0
            
            if not testOnly:
                item.setGeometry(QRect(rect.x() + x, 
                                       rect.y() + y,
                                       wid.sizeHint().width(),
                                       wid.sizeHint().height()))
            
            x = nextX
            lineHeight = max(lineHeight, wid.sizeHint().height())
        
        return y + lineHeight

