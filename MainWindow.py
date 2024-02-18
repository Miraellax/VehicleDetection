import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QToolButton, QStyle
import CustomTitleBar

# def InitSettings():
#     with open("./settings", )
def ReadClassDict(path):
    with open(path, "r") as f:
        file = yaml.safe_load(f)
        class_list = file["names"]
        class_colors = file["colors"]

        class_dict = dict(enumerate(class_list, start=0))
        class_dict_colors = dict(enumerate(class_colors, start=0))
        print(class_dict)
        print(class_dict_colors)

    return class_dict, class_dict_colors

class DetectionWindow(QtWidgets.QWidget):
    def __init__(self):
        super(DetectionWindow, self).__init__()
        self.dirty = True
        self.setWindowTitle('ECDetector')

        data_path = "./Emergency Vehicles Russia.v3i.yolov8"
        self.objectClassesDict, self.objectColorsDict = ReadClassDict("./" + data_path + "/data.yaml")

        self.currentFrameColor = -1
        self.multipleClasses = False

        # ensure that the widget always stays on top, no matter what
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)

        self.title_bar = CustomTitleBar.CustomTitleBar(self)

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
        # Set frame color if car is detected or other signal sent

        framePalette = self.palette()

        if objectClass == self.currentFrameColor:
            return

        if objectClass == -1:
            framePalette.setColor(self.backgroundRole(), getattr(Qt, "gray"))

        if multipleClasses:
            # TODO поддержка нескольких объектов в кадре - смена цвета,
            #  приоритеты цветов, красить в несколько фон?

            color = self.objectColorsDict[objectClass]
            try:
                framePalette.setColor(self.backgroundRole(), getattr(Qt, color))
            except AttributeError:
                print(f"Attribute error, color {color} for {objectClass} does not exist.")
        #         log in error logs


        # Single class - emergency car
        else:
            color = "red"
            framePalette.setColor(self.backgroundRole(), getattr(Qt, color))
        self.setPalette(framePalette)

    def takeScreenshot(self):
        # TODO учитывать наличие нескольких экранов
        print("Screen")
        screen = QtWidgets.QApplication.primaryScreen()
        screenZone = self.grabWidget.geometry()
        screenZone.moveTopLeft(self.grabWidget.mapToGlobal(QtCore.QPoint(0, 0)))
        screenZone = screenZone.getRect()
        # will not work on IOS?
        screenshot = screen.grabWindow(
                                           0,  # window voidptr
                                           screenZone[0],  # x
                                           screenZone[1],  # y
                                           screenZone[2],  # width
                                           screenZone[3],  # height
                                       )
        screenshot.save('shot.jpg', 'jpg')
        print(screenshot.size())
        return screenshot

    def updateMask(self):

        # get the *whole* window geometry, including its titlebar and borders
        frameRect = self.frameGeometry()

        # get the child widgets geometry and remap it to global coordinates
        titlebarG = self.title_bar.geometry()
        titlebarG.moveTopLeft(self.grabWidget.mapToGlobal(QtCore.QPoint(0, 0)))
        grabGeometry= self.grabWidget.geometry()
        grabGeometry.moveTopLeft(self.grabWidget.mapToGlobal(QtCore.QPoint(0, 0)))


        # get the actual margins between the grabWidget and the window margins
        left = frameRect.left() - grabGeometry.left()
        top = frameRect.top() - grabGeometry.top()
        right = frameRect.right() - grabGeometry.right()
        bottom = frameRect.bottom() - grabGeometry.bottom()

        # reset the geometries to get rectangles for the mask
        frameRect.moveTopLeft(QtCore.QPoint(0, 0))
        titlebarG.moveTopLeft(QtCore.QPoint(0, 0))
        grabGeometry.moveTopLeft(QtCore.QPoint(self.contentsMargins().top(), self.contentsMargins().left()))
        grabGeometry.moveTop(titlebarG.bottom()+self.contentsMargins().top()+1)

        # create the base mask region, adjusted to the margins between the
        # grabWidget and the window as computed above
        region = QtGui.QRegion(frameRect.adjusted(left, top, right, bottom))

        # "subtract" the grabWidget rectangle to get a mask that only contains
        # the window titlebar and margins
        region -= QtGui.QRegion(grabGeometry)
        self.setMask(region)

    def resizeEvent(self, event):
        super(DetectionWindow, self).resizeEvent(event)
        # the first resizeEvent is called *before* any first-time showEvent and
        # paintEvent, there's no need to update the mask until then
        if not self.dirty:
            self.updateMask()

    def paintEvent(self, event):
        super(DetectionWindow, self).paintEvent(event)
        if self.dirty:
            self.updateMask()
            self.dirty = False


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DetectionWindow()
    window.show()
    window.setFrameColor(2, True)

    data_path = "./Emergency Vehicles Russia.v3i.yolov8"
    ReadClassDict("./" + data_path + "/data.yaml")
    sys.exit(app.exec_())