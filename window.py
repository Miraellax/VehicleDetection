print("gg"[0])

# import sys
#
# from PyQt5.QtWidgets import QDesktopWidget
# from PyQt5.QtCore import Qt, QRect
# from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
# from PyQt6.uic.properties import QtWidgets
# from qtpy import QtCore
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("My App")
#         # self.setWindowFlags(Qt.FramelessWindowHint)
#         # self.setAttribute(Qt.WA_NoSystemBackground, True)
#         # self.setAttribute(Qt.WA_TranslucentBackground, True)
#
# #
# # app = QApplication(sys.argv)
# #
# # window = MainWindow()
# # layout = QtWidgets.QGridLayout(app)
# #
# # close_button = QtWidgets.QPushButton()
# # close_button.setText('close window')
# # close_button.clicked.connect(lambda: app.exit(0))
# # layout.addWidget(close_button)
# # window.show()
#
# # app.exec()
# app = QApplication(sys.argv)
#
# # Create the main window
# window = MainWindow()
#
# # Create the button
# pushButton = QPushButton(window)
# pushButton.setGeometry(QRect(240, 190, 90, 31))
# pushButton.setText("Finished")
# pushButton.clicked.connect(app.quit)
#
# # Center the button
# qr = pushButton.frameGeometry()
# cp = QDesktopWidget().availableGeometry().center()
# qr.moveCenter(cp)
# pushButton.move(qr.topLeft())
#
# # Run the application
# window.show()
# sys.exit(app.exec_())