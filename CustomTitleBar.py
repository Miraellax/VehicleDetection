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
        title_bar_layout.setContentsMargins(1, 1, 1, 1)

        # Placeholder for a title
        self.title = QLabel(f"{self.__class__.__name__}", self)
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

        # debug screenshot button
        self.screen_button = QToolButton(self)
        screen_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_FileIcon
        )
        self.screen_button.setIcon(screen_icon)
        self.screen_button.clicked.connect(self.parentWidget().predict)

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
            self.screen_button,
            self.min_button,
            self.close_button
        ]
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(QSize(self.minimumHeight() - 4, self.minimumHeight() - 4))
            button.setStyleSheet(
                """QToolButton {
                                 background-color: white;
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
