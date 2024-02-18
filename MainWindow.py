import math

import PIL
import numpy as np
from torchvision import transforms
from torchvision.transforms.v2 import Compose
import yaml
import logging
import asyncio

from PIL import ImageQt
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QToolButton, QStyle

import CustomTitleBar
import ECModel


# def InitSettings():
#     with open("./settings", )
def read_class_dict(path: str) -> tuple:
    """

    :type path: str
    """
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

        dict_path = "./Emergency Vehicles Russia.v3i.yolov8"
        self.objectClassesDict, self.objectColorsDict = read_class_dict("./" + dict_path + "/data.yaml")

        self.model = ECModel.ECModel()

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
    def set_frame_color(self, object_class: object, multiple_classes: object = False) -> object:
        """
        Set frame color if car is detected or other signal sent
        :param object_class:
        :param multiple_classes:
        :return:
        """
        frame_palette = self.palette()

        if object_class == self.currentFrameColor:
            return

        if object_class == -1:
            frame_palette.setColor(self.backgroundRole(), getattr(Qt, "gray"))

        if multiple_classes:
            # TODO поддержка нескольких объектов в кадре - смена цвета,
            #  приоритеты цветов (class asyncio.PriorityQueue), красить в несколько фон?

            color = self.objectColorsDict[object_class]
            try:
                frame_palette.setColor(self.backgroundRole(), getattr(Qt, color))
            except AttributeError:
                print(f"Attribute error, color {color} for {object_class} does not exist.")
        #         log in error logs

        # Single class - emergency car
        else:
            color = "red"
            frame_palette.setColor(self.backgroundRole(), getattr(Qt, color))
        self.setPalette(frame_palette)

    def take_screenshot(self):
        # TODO учитывать наличие нескольких экранов
        print("Screen")
        screen = QtWidgets.QApplication.primaryScreen()
        screen_zone = self.grabWidget.geometry()
        screen_zone.moveTopLeft(self.grabWidget.mapToGlobal(QtCore.QPoint(0, 0)))
        screen_zone = screen_zone.getRect()
        # will not work on IOS?
        screenshot = screen.grabWindow(
                                           0,  # window voidptr
                                           screen_zone[0],  # x
                                           screen_zone[1],  # y
                                           screen_zone[2],  # width
                                           screen_zone[3],  # height
                                       )
        screenshot.save('shot.jpg', 'jpg')
        print(screenshot.size())
        return screenshot

    def predict(self):
        image = self.take_screenshot().toImage()
        channels = 3
        s = image.bits().asstring(image.width() * image.height() * channels)
        image_array = np.fromstring(s, dtype=np.uint8).reshape((image.height(), image.width(), channels))
        # pil_image = Image.frombytes("RGB", (image.width(), image.height()), image)
        # pil_image = ImageQt.fromqimage(image)
        result = self.model.predict(image_array)
        print(result)




    def update_mask(self):
        # get the *whole* window geometry, including borders
        frame_rect = self.frameGeometry()

        # get the child widgets geometry and remap it to global coordinates
        titlebar_geometry = self.title_bar.geometry()
        titlebar_geometry.moveTopLeft(self.grabWidget.mapToGlobal(QtCore.QPoint(0, 0)))
        grab_geometry = self.grabWidget.geometry()
        grab_geometry.moveTopLeft(self.grabWidget.mapToGlobal(QtCore.QPoint(0, 0)))

        # get the actual margins between the grabWidget and the window margins
        left = frame_rect.left() - grab_geometry.left()
        top = frame_rect.top() - grab_geometry.top()
        right = frame_rect.right() - grab_geometry.right()
        bottom = frame_rect.bottom() - grab_geometry.bottom()

        # reset the geometries to get rectangles for the mask
        frame_rect.moveTopLeft(QtCore.QPoint(0, 0))
        titlebar_geometry.moveTopLeft(QtCore.QPoint(0, 0))
        grab_geometry.moveTopLeft(QtCore.QPoint(self.contentsMargins().top(), self.contentsMargins().left()))
        grab_geometry.moveTop(titlebar_geometry.bottom()+self.contentsMargins().top()+1)

        # create the base mask region, adjusted to the margins between the
        # grabWidget and the window as computed above
        region = QtGui.QRegion(frame_rect.adjusted(left, top, right, bottom))

        # "subtract" the grabWidget rectangle to get a mask that only contains
        # the window titlebar and margins
        region -= QtGui.QRegion(grab_geometry)
        self.setMask(region)

    def resizeEvent(self, event):
        super(DetectionWindow, self).resizeEvent(event)
        # the first resizeEvent is called *before* any first-time showEvent and
        # paintEvent, there's no need to update the mask until then
        if not self.dirty:
            self.update_mask()

    def paintEvent(self, event):
        super(DetectionWindow, self).paintEvent(event)
        if self.dirty:
            self.update_mask()
            self.dirty = False


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DetectionWindow()
    window.show()
    window.set_frame_color(2, True)

    data_path = "./Emergency Vehicles Russia.v3i.yolov8"
    read_class_dict("./" + data_path + "/data.yaml")
    sys.exit(app.exec_())
