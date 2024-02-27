import io
import math
import os
import time
import traceback
from asyncio import sleep

import yaml
import logging
import asyncio

from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize, QBuffer, QTimer, QObject, QThread, pyqtSignal, pyqtSlot, QMutex
from PyQt5.QtGui import QWindow
from PyQt5.QtWidgets import QLabel, QSizePolicy, QFrame

from MainWindow import DetectionWindow


class BBoxWidget(QFrame):
    def __init__(self, parent:DetectionWindow, title: str, color:str, coords:tuple, size:tuple):
        super(BBoxWidget, self).__init__()

        self.parent = parent
        self.coords = coords
        self.size = size
        self.setGeometry(self.coords[0], self.coords[1], self.size[0], self.size[1])

        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.MSWindowsFixedSizeDialogHint)
        # self.setWindowFlags(Qt.CustomizeWindowHint)
        self.setFrameStyle(QFrame.NoFrame)

        self.class_color = color
        self.setAutoFillBackground(True)
        frame_palette = self.palette()
        frame_palette.setColor(self.backgroundRole(), getattr(Qt, self.class_color))
        self.setPalette(frame_palette)


        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)


        self.title = QLabel()
        self.title.setText(title)
        self.title.setAlignment(Qt.AlignLeft)
        self.title.setStyleSheet("font-weight: bold")
        layout.addWidget(self.title)

        # limit widget AND layout margins
        # layout.setContentsMargins(0, 0, 0, 0)
        # self.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)

        # # Margins for frame to resize correctly
        # self.setContentsMargins(2, 2, 2, 2)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)

    window = BBoxWidget(None, "help 0.84", "red", (200, 400), (200, 400))
    window.show()

    sys.exit(app.exec_())