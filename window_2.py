from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QToolButton, QStyle


class CustomTitleBar(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(20)

        self.setAutoFillBackground(True)
        self.initial_pos = None

        title_bar_layout = QtWidgets.QHBoxLayout(self)
        # self.setLayout(title_bar_layout)
        title_bar_layout.setContentsMargins(1, 1, 1, 1)
        # title_bar_layout.setSpacing(2)

        # Placeholder for a title
        self.title = QLabel(f"{self.__class__.__name__}", self)
        # self.title = QLabel("")
        self.title.setStyleSheet(
            """font-weight: bold;
               border-radius: 12px;
               margin: 2px;
               color: black;
            """
        )
        self.title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # If parent has a name add it in a titlebar, else none
        if title := parent.windowTitle():
            self.title.setText(title)
        title_bar_layout.addWidget(self.title)

        # Min button
        self.min_button = QToolButton(self)
        min_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_TitleBarMinButton
        )
        self.min_button.setIcon(min_icon)
        self.min_button.clicked.connect(self.window().showMinimized)

        # Close button
        self.close_button = QToolButton(self)
        close_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_TitleBarCloseButton
        )
        self.close_button.setIcon(close_icon)
        self.close_button.clicked.connect(self.parentWidget().close)

        buttons = [
            self.min_button,
            self.close_button,
        ]
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(QSize(self.minimumHeight()-4, self.minimumHeight()-4))
            button.setStyleSheet(
                """QToolButton { border: 2px solid black;
                                 border-radius: 3px;
                                }
                """
            )
            title_bar_layout.addWidget(button)

    def mousePressEvent(self, evt):
        self.parentWidget().oldPos = evt.globalPos()

    def mouseMoveEvent(self, evt):
        delta = QtCore.QPoint(evt.globalPos() - self.parentWidget().oldPos)
        self.parentWidget().move(self.parentWidget().x() + delta.x(), self.parentWidget().y() + delta.y())
        self.parentWidget().oldPos = evt.globalPos()


class DetectionWindow(QtWidgets.QWidget):
    def __init__(self):
        super(DetectionWindow, self).__init__()
        self.dirty = True
        self.setWindowTitle('ECDetector')

        # ensure that the widget always stays on top, no matter what
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)

        self.title_bar = CustomTitleBar(self)

        # Creating vertical layout to hold inner interface elements
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        # limit widget AND layout margins
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create custom title bar
        self.layout().addWidget(self.title_bar)

        # create a "placeholder" widget for the screen grab geometry
        self.grabWidget = QtWidgets.QWidget()
        self.grabWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.grabWidget.setMinimumSize(self.title_bar.minimumWidth(), 40)

        # debug
        self.grabWidget.setAutoFillBackground(True)
        self.grabWidget.setStyleSheet(
            """
               background-color: gray;
            """
        )
        layout.addWidget(self.grabWidget)
        # Margins for frame to resize correctly
        self.setContentsMargins(2, 2, 2, 2)



    # TODO
    def setFrameColor(self, objectClass, multipleClasses=False):
        colors = {"gray": Qt.gray, "red": Qt.red }
        # Set frame color if car is detected or other signal sent
        if multipleClasses:
            pass
        # Single class - emergency car
        else:
            framePalette = self.grabWidget.palette()
            framePalette.setColor(self.grabWidget.backgroundRole(), Qt.red)
            self.grabWidget.setPalette(framePalette)

    def updateMask(self):

        # print("self.grabWidget.frameGeometry()", self.grabWidget.frameGeometry())
        # print("self.frameGeometry()", self.frameGeometry())
        # print("self.title_bar.frameGeometry()", self.title_bar.frameGeometry())
        # print("self.pos()", self.pos())
        # print(self.frameGeometry().getRect())
        # print(self.title_bar.frameGeometry().getRect())
        # print(self.grabWidget.frameGeometry().getRect())

        # get the *whole* window geometry, including its titlebar and borders
        frameRect = self.frameGeometry()

        # get the child widgets geometry and remap it to global coordinates
        titlebarG = self.title_bar.geometry()
        titlebarG.moveTopLeft(self.grabWidget.mapToGlobal(QtCore.QPoint(0, 0)))
        h2 = self.grabWidget.geometry()
        grabGeometry= self.grabWidget.geometry()
        grabGeometry.moveTopLeft(self.grabWidget.mapToGlobal(QtCore.QPoint(0, 0)))


        # get the actual margins between the grabWidget and the window margins
        left = frameRect.left() - grabGeometry.left()
        top = frameRect.top() - grabGeometry.top()
        right = frameRect.right() - grabGeometry.right()
        bottom = frameRect.bottom() - grabGeometry.bottom()
        print(left, top, right, bottom)

        # reset the geometries to get rectangles for the mask
        frameRect.moveTopLeft(QtCore.QPoint(0, 0))
        titlebarG.moveTopLeft(QtCore.QPoint(0, 0))
        grabGeometry.moveTopLeft(QtCore.QPoint(self.contentsMargins().top(), self.contentsMargins().left()))
        grabGeometry.moveTop(titlebarG.bottom()+self.contentsMargins().top())

        # create the base mask region, adjusted to the margins between the
        # grabWidget and the window as computed above
        region = QtGui.QRegion(frameRect.adjusted(left, top, right, bottom))
        # a = frameRect
        region = QtGui.QRegion(region)
        # "subtract" the grabWidget rectangle to get a mask that only contains
        # the window titlebar and margins
        region -= QtGui.QRegion(grabGeometry)
        self.setMask(region)

    def resizeEvent(self, event):
        super(DetectionWindow, self).resizeEvent(event)
        # the first resizeEvent is called *before* any first-time showEvent and
        # paintEvent, there's no need to update the mask until then; see below
        if not self.dirty:
            print("RESIZE")
            self.updateMask()

    def paintEvent(self, event):
        super(DetectionWindow, self).paintEvent(event)
        # on Linux the frameGeometry is actually updated "sometime" after show()
        # is called; on Windows and MacOS it *should* happen as soon as the first
        # non-spontaneous showEvent is called (programmatically called: showEvent
        # is also called whenever a window is restored after it has been
        # minimized); we can assume that all that has already happened as soon as
        # the first paintEvent is called; before then the window is flagged as
        # "dirty", meaning that there's no need to update its mask yet.
        # Once paintEvent has been called the first time, the geometries should
        # have been already updated, we can mark the geometries "clean" and then
        # actually apply the mask.
        if self.dirty:
            print("PAINT")
            self.updateMask()
            self.dirty = False


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())