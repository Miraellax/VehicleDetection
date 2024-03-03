from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QLabel, QToolButton, QStyle, QAction, QMenuBar, QWidgetAction, QHBoxLayout, QSlider


class SliderAction(QWidgetAction):
    def __init__(self, current_level):
        super().__init__(self)
        self.layout = QHBoxLayout()
        self.label = QLabel(str(current_level))
        self.slider = QSlider()

        self.layout.addWidget(self.label)

    def valueChanged(self, event):
        super(self.slider, self).valueChanged(event)

        print("CHANGED")


class CustomTitleBar(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(20)

        self.setAutoFillBackground(True)
        self.initial_pos = None

        # TODO check in settings file
        self.is_showing_fps = True
        self.is_showing_class_name = True

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

        # TODO add widget to show fps text and show/hide it

        # self.slider_action = SliderAction(0.2)
        self.show_class_name = QAction("Показывать имя класса", self, checkable=True)
        self.show_class_name.setChecked(self.is_showing_class_name)
        self.show_class_name.triggered.connect(self.change_show_class_name_state)

        self.show_fps = QAction("Показывать FPS", self, checkable=True)
        self.show_fps.setChecked(self.is_showing_fps)
        self.show_fps.triggered.connect(self.change_show_fps_state)

        # self.menu.addAction(self.newAction)
        self.menu.addAction(self.show_class_name)
        self.menu.addAction(self.show_fps)

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

    def change_show_fps_state(self):
        # logic to show fps in title bar or not
        self.is_showing_fps = self.show_fps.isChecked()

    def change_show_class_name_state(self):
        # logic to class name in bboxes or not
        self.is_showing_class_name = self.show_class_name.isChecked()

    def set_color(self, color: str):
        self.menu_bar.setStyleSheet(f"background-color: {color};")
        for elem in self.objects_to_color:
            palette = elem.palette()
            palette.setColor(self.backgroundRole(), getattr(Qt, color))
            elem.setPalette(palette)

    def mousePressEvent(self, evt):
        self.parentWidget().oldPos = evt.globalPos()

    def mouseMoveEvent(self, evt):
        delta = QtCore.QPoint(evt.globalPos() - self.parentWidget().oldPos)
        self.parentWidget().move(self.parentWidget().x() + delta.x(), self.parentWidget().y() + delta.y())
        self.parentWidget().oldPos = evt.globalPos()
