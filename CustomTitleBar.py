from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPalette, QIcon
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QToolButton, QStyle, QToolTip, QMenu, QAction, QMenuBar, QSizePolicy


class CustomTitleBar(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(20)

        self.setAutoFillBackground(True)
        self.initial_pos = None

        title_bar_layout = QtWidgets.QHBoxLayout(self)
        title_bar_layout.setContentsMargins(1, 1, 1, 1)

        self.objects_to_color = [self]

        # File menu
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setFixedWidth(36)
        self.menu_bar.setAutoFillBackground(True)
        self.menu = self.menu_bar.addMenu("")
        self.menu.setIcon(QIcon("menu_icon.png"))
        title_bar_layout.addWidget(self.menu_bar)

        self.newAction = QAction("1", self)
        self.openAction = QAction("2", self)
        self.saveAction = QAction("3", self)
        self.exitAction = QAction("4", self)
        self.menu.addAction(self.newAction)
        self.menu.addAction(self.openAction)
        self.menu.addAction(self.saveAction)
        self.menu.addAction(self.exitAction)

        self.menu_bar.show()

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
        
        self.model_indicator = QToolButton(self)
        warning_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MessageBoxWarning
        )
        self.model_indicator.setIcon(warning_icon)
        self.model_indicator.setToolTip("<b>WARNING:</b> model is not loaded")
        self.model_indicator.setFixedSize(QSize(self.minimumHeight() - 2, self.minimumHeight() - 2))
        self.model_indicator.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        title_bar_layout.addWidget(self.model_indicator)

        self.setStyleSheet("""QToolTip {  
                                           color: red;
                                       }""")

        # debug screenshot button
        # self.screen_button = QToolButton(self)
        # screen_icon = self.style().standardIcon(
        #     QStyle.StandardPixmap.SP_FileIcon
        # )
        # self.screen_button.setIcon(screen_icon)
        # self.screen_button.clicked.connect(self.parentWidget().take_screenshot)

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
            # self.screen_button,
            self.min_button,
            self.close_button
        ]
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(QSize(self.minimumHeight() - 4, self.minimumHeight() - 4))
            button.setStyleSheet("background-color: white;")
            title_bar_layout.addWidget(button)

    def set_color(self, color: str):
        self.menu_bar.setStyleSheet(f"background-color: {color};")
        for elem in self.objects_to_color:
            # elem.setStyleSheet(f"background-color: {color};")
            palette = elem.palette()
            palette.setColor(self.backgroundRole(), getattr(Qt, color))
            elem.setPalette(palette)

    def mousePressEvent(self, evt):
        self.parentWidget().oldPos = evt.globalPos()

    def mouseMoveEvent(self, evt):
        delta = QtCore.QPoint(evt.globalPos() - self.parentWidget().oldPos)
        self.parentWidget().move(self.parentWidget().x() + delta.x(), self.parentWidget().y() + delta.y())
        self.parentWidget().oldPos = evt.globalPos()
