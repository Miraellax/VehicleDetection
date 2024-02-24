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

import CustomTitleBar
import ECModel

logging.basicConfig(format='"%(asctime)s [%(levelname)s] %(name)s: %(message)s"',
                    filename='Main.log',
                    encoding='utf-8',
                    # filemode='w', # if need to erase logs each run
                    )

# set debug mode
os.environ["PYTHONASYNCIODEBUG"] = "1"

current_bboxes = None
current_image = None

mutex = QMutex()

# TODO settings
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
        # how often we try to check bboxes and send images to process, model might work slower than this
        self.fps_check = 1

        dict_path = "./Emergency Vehicles Russia.v3i.yolov8"
        self.objectClassesDict, self.objectColorsDict = read_class_dict("./" + dict_path + "/data.yaml")

        self.model = None
        self.run_thread_init_task()

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

        # Check bboxes for predictions TODO
        timer = QTimer(self, interval=1000 // self.fps_check, timeout=self.handle_timeout)
        timer.start()
        self.handle_timeout()

    def handle_timeout(self):
        global current_bboxes

        # if answers ready - visualise
        if mutex.tryLock():
            logging.info("Main: locked bboxes, updating interface")
            print("Main: locked bboxes, updating interface")
            self.check_bboxes()
            logging.info("Main: unlocking bboxes")
            print("Main: unlocking bboxes")
            mutex.unlock()

    def show_model_init_error(self):
        pass

    def check_bboxes(self):
        # threading, check if model bboxes are ready and can be visualized
        if current_bboxes is None:
            logging.info("Main: Tried to check bboxes, but they are None type")
            print("Main: Tried to check bboxes, but they are None type")
            return

        print("BEBOXES good")
        self.update_mask()


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
            #  приоритеты цветов (class asyncio.PriorityQueue), красить в несколько цветов рамку?

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
        # gebug save
        screenshot.save('shot.jpg', 'jpg')

        return screenshot

    def send_to_predict(self, image):
        # image = self.take_screenshot().toImage()

        # convert Qpixmap to PIL
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, "PNG")
        pil_im = Image.open(io.BytesIO(buffer.data()))
        # pil_im.show()

        logging.info("Sending image to model to process")
        print("Sending image to model to process")
        result = self.model.process_image(pil_im)
        return result

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

    def update(self) -> None:
        super(DetectionWindow, self).update()
        # After update need to check model and send image


    def run_thread_init_task(self):
        self.thread = QThread()
        self.thread.__PCHthreadName = "ModelThread"
        self.worker = ModelWorker(self)
        self.worker.moveToThread(self.thread)

        # Init model at first
        self.thread.started.connect(self.worker.init_model)
        # Start processing images now and after model done predicting and updated bboxes
        self.worker.signals.finished_init.connect(self.worker.process_image)
        self.worker.signals.updated_bboxes.connect(self.worker.process_image)
        # self.worker.signals.updated_bboxes.connect(self.check_bboxes)

        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.signals.error.connect(self.show_model_init_error)
        # self.worker.progress.connect(self.reportProgress)

        logging.info("Main: starting model init thread")
        print("Main: starting model init thread")
        self.thread.start()

        # # Final resets
        # self.longRunningBtn.setEnabled(False)
        # self.thread.finished.connect(
        #     lambda: self.longRunningBtn.setEnabled(True)
        # )
        # self.thread.finished.connect(
        #     lambda: self.stepLabel.setText("Long-Running Step: 0")
        # )

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
        finished - No data
        error - tuple (exctype, value, traceback.format_exc() )
        result - object data returned from processing, anything
    '''
    finished = pyqtSignal()
    finished_init = pyqtSignal()
    updated_bboxes = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class ModelWorker(QObject):
    '''
        Worker thread
        Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
        :param callback: The function callback to run on this worker thread. Supplied args and
                         kwargs will be passed through to the runner.
        :type callback: function
        :param args: Arguments to pass to the callback function
        :param kwargs: Keywords to pass to the callback function
        '''

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.signals = WorkerSignals()
        self.mutex = QMutex()

    def process_image(self):

        # global current_image
        global current_bboxes
        # look on current image, lock it
        if self.mutex.tryLock():
            logging.info(f"{self} started to process image")
            print(f"{self} started to process image")
            # send it to process
            image = self.parent.take_screenshot()
            results = self.parent.send_to_predict(image)
            # after finished get bboxes and update locked bbox_storage
            self.mutex.unlock()
            self.signals.updated_bboxes.emit()
        else:
            logging.info(f"ModelWorker: mutex is already locked")
            print(f"ModelWorker: mutex is already locked")
        # unlock storage and notify

        # try:
        #     print("starting model")
        #     self.parent.model = ECModel.ECModel()
        # except:
        #     traceback.print_exc()
        #     exctype, value = sys.exc_info()[:2]
        #     self.signals.error.emit((exctype, value, traceback.format_exc()))
        # # else:
        #     self.signals.updated_BBoxes.emit()
        # finally:
        #     self.signals.finished.emit()  # Done
        #     print("model done")
        logging.info(f"ModelWorker: ended to process image")
        print(f"ModelWorker: ended to process image")
    def init_model(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            print("starting model")
            self.parent.model = ECModel.ECModel()
            self.parent.model.moveToThread(self.parent.thread)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        # else:
        #     self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished_init.emit()  # Done


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)

    window = DetectionWindow()
    window.show()

    sys.exit(app.exec_())
