import logging
import sys
import traceback

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QLabel, QToolButton, QStyle, QAction, QMenuBar, QWidgetAction, QHBoxLayout, QSlider, \
    QActionGroup


class CustomTitleBar(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.setMinimumWidth(100)
        self.setMinimumHeight(20)
        self.fps = 0

        self.setAutoFillBackground(True)
        self.initial_pos = None

        self.is_showing_fps = False
        self.is_showing_class_name = False
        self.is_doing_sound = True

        title_bar_layout = QtWidgets.QHBoxLayout(self)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)

        self.objects_to_color = [self]

        self.menu_bar = QMenuBar(self)
        self.menu_bar.setFixedWidth(36)
        self.menu_bar.setAutoFillBackground(True)
        self.menu = self.menu_bar.addMenu("")
        self.menu.setIcon(QIcon("./resources/menu_icon.png"))
        title_bar_layout.addWidget(self.menu_bar)

        self.show_class_name = QAction(text="Показывать имя класса", parent=self, checkable=True)
        self.show_class_name.setChecked(self.is_showing_class_name)
        self.show_class_name.triggered.connect(self.change_show_class_name_state)

        self.show_fps = QAction(text="Показывать FPS", parent=self, checkable=True)
        self.show_fps.setChecked(self.is_showing_fps)
        self.show_fps.triggered.connect(self.change_show_fps_state)

        self.menu.addSeparator()

        self.model_menu = self.menu.addMenu("Выбор активной модели")
        self.load_model_menu()

        self.do_sound = QAction(text="Аудио-уведомления", parent=self, checkable=True)
        self.do_sound.setChecked(self.is_doing_sound)
        self.do_sound.triggered.connect(self.change_do_sound_state)

        self.menu.addAction(self.show_class_name)
        self.menu.addAction(self.show_fps)
        self.menu.addAction(self.do_sound)

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

        self.fps_text = QLabel(f"{self.fps} FPS", self)
        self.fps_text.setStyleSheet(
            """font-weight: bold;
               border-radius: 12px;
               margin: 2px;
               color: black;
            """
        )
        self.fps_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.fps_text.setHidden(not self.is_showing_fps)
        title_bar_layout.addWidget(self.fps_text)
        title_bar_layout.addStretch()
        
        self.model_indicator = QToolButton(self)
        warning_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MessageBoxWarning
        )
        self.model_indicator.setIcon(warning_icon)
        self.model_indicator.setToolTip("<b>ВНИМАНИЕ:</b> модель не загружена")
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
        self.fps_text.setHidden(not self.is_showing_fps)

    def change_show_class_name_state(self):
        # logic to class name in bboxes or not
        self.is_showing_class_name = self.show_class_name.isChecked()

    def change_do_sound_state(self):
        # logic to class name in bboxes or not
        self.is_doing_sound = self.do_sound.isChecked()
        self.parent.sound_player.setMuted(not self.is_doing_sound)

    def update_fps(self, new_fps):
        self.fps_text.setText(f"{new_fps} FPS")

    def hide_warning(self):
        self.model_indicator.setHidden(True)

    def set_color(self, color: str):
        self.menu_bar.setStyleSheet(f"background-color: {color};")
        for elem in self.objects_to_color:
            palette = elem.palette()
            palette.setColor(self.backgroundRole(), getattr(Qt, color))
            elem.setPalette(palette)

    def load_model_menu(self):
        checked_model = self.parent.model_name
        dict = self.parent.weights_dict
        self.model_group = QActionGroup(self)

        for model in dict:
            action = QAction(text=model, parent=self, checkable=True)
            action.setObjectName(model)
            self.model_group.addAction(action)
            if model == self.parent.model_name:
                action.setChecked(True)

            action.triggered.connect(self.set_new_model_weight)
            self.model_menu.addAction(action)

    def set_new_model_weight(self):
    #     send signal to parent and custom actions

        current_model = self.model_group.checkedAction().text()
        self.parent.model_name = current_model
        self.parent.run_thread_init_task()
    def mousePressEvent(self, evt):
        self.parentWidget().oldPos = evt.globalPos()

    def mouseMoveEvent(self, evt):
        delta = QtCore.QPoint(evt.globalPos() - self.parentWidget().oldPos)
        self.parentWidget().move(self.parentWidget().x() + delta.x(), self.parentWidget().y() + delta.y())
        self.parentWidget().oldPos = evt.globalPos()
 