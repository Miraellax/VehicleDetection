from PyQt5 import  QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QFrame

import MainWindow


class BBoxWidget(QFrame):
    def __init__(self, parent: MainWindow, title: str, color:str, coords:tuple, size:tuple):
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

        self.transparent_window = QtWidgets.QWidget()
        self.transparent_window.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.transparent_window.setAutoFillBackground(True)
        frame_palette = self.transparent_window.palette()
        frame_palette.setColor(self.transparent_window.backgroundRole(), getattr(Qt, "white"))
        self.transparent_window.setPalette(frame_palette)
        layout.addWidget(self.transparent_window)

        # limit widget AND layout margins
        layout.setContentsMargins(1, 1, 1, 1)
        self.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # # Margins for frame to resize correctly
        # self.setContentsMargins(2, 2, 2, 2)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)

    window = BBoxWidget(None, "Имя класса (вероятность)", "yellow", (200, 300), (200, 150))
    window.show()

    sys.exit(app.exec_())
