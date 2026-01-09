# file: gui/flow_layout.py
# Custom Qt layout that places widgets left-to-right like "flow" text,
# and automatically wraps to the next line when there is no space.

from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """
    A custom flow layout:
    - Arranges child widgets horizontally
    - When a widget would go beyond the container width, it wraps to next line
    Useful for lots of checkboxes/buttons (e.g., channel selection).
    """

    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)

        # List that stores all layout items
        self.itemList = []

        # Outer padding around the layout
        self.setContentsMargins(margin, margin, margin, margin)

        # Spacing between widgets (use default spacing if spacing < 0)
        self.setSpacing(spacing if spacing >= 0 else self.spacing())

    def __del__(self):
        """Cleanup: remove all items from the layout"""
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        """Qt calls this when a widget is added to the layout"""
        self.itemList.append(item)

    def count(self):
        """Return how many items are currently in the layout"""
        return len(self.itemList)

    def itemAt(self, index):
        """Return the item at a given index (without removing it)"""
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        """Remove and return the item at a given index"""
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        """
        Tell Qt that this layout does NOT force expansion in any direction.
        (It will size based on its contents.)
        """
        return Qt.Orientations(Qt.Orientation(0))

    def sizeHint(self):
        """Suggested size for this layout (Qt uses this for initial sizing)"""
        return self.minimumSize()

    def minimumSize(self):
        """
        Minimum size needed to show content.
        Uses the largest minimumSize among the child widgets.
        """
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())

        # Add padding based on spacing so widgets don’t collide
        size += QSize(2 * self.spacing(), 2 * self.spacing())
        return size

    def setGeometry(self, rect):
        """
        Called when Qt assigns a rectangle to this layout.
        We respond by laying out children inside this rectangle.
        """
        super().setGeometry(rect)
        self.doLayout(rect)

    def doLayout(self, rect, testOnly=False):
        """
        Core layout logic:
        - Place widgets in a row
        - If the next widget exceeds rect.width(), wrap to next line
        - Update positions based on each widget's sizeHint()

        If testOnly=True, only calculates size without placing widgets.
        """
        x = 0
        y = 0
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()

            # Horizontal and vertical spacing between items
            spaceX = self.spacing()
            spaceY = self.spacing()

            # Proposed next x-position if we place widget on current row
            nextX = x + wid.sizeHint().width() + spaceX

            # Wrap to next line if widget would overflow the available width
            if nextX > rect.width() and x > 0:
                x = 0
                y = y + lineHeight + spaceY
                nextX = wid.sizeHint().width() + spaceX
                lineHeight = 0

            # Actually set widget geometry (unless only testing)
            if not testOnly:
                item.setGeometry(
                    QRect(
                        rect.x() + x,
                        rect.y() + y,
                        wid.sizeHint().width(),
                        wid.sizeHint().height()
                    )
                )

            # Move x forward for next widget
            x = nextX

            # Track tallest widget in the current row
            lineHeight = max(lineHeight, wid.sizeHint().height())

        # Total height used by the layout
        return y + lineHeight
