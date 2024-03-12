import logging
import io
import os
import traceback
from logging.handlers import MemoryHandler

import numpy as np
import yaml


from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QBuffer, QTimer, QObject, QThread, pyqtSignal, pyqtSlot, QMutex, QUrl
from PyQt5.QtMultimedia import QSoundEffect

import CustomTitleBar
import ECModel
import BBoxWidget

logging.basicConfig(filename='Main_log.log',
                            filemode='a'
                            )

# set debug mode
os.environ["PYTHONASYNCIODEBUG"] = "1"

current_bboxes = None
current_image = None

mutex = QMutex()


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

    return class_dict, class_dict_colors


class DetectionWindow(QtWidgets.QWidget):
    timer_signal = pyqtSignal()

    def __init__(self):
        super(DetectionWindow, self).__init__()
        self.dirty = True
        self.setWindowTitle('ECDetector')
        self.model_name = "YOLOv8s"  # default model

        # Read dicts from path
        dict_path = "./Emergency Vehicles Russia.v3i.yolov8"
        self.objectClassesDict, self.objectColorsDict = read_class_dict("./" + dict_path + "/data.yaml")
        self.weights_dict = np.load('resources/model_weight_dict.npy', allow_pickle=True).item()
        print(self.objectColorsDict)
        self.logger = self.set_logger()

        # set max detection rate
        self.fps_check = 30
        self.timer = QTimer(self, interval=1000 // self.fps_check, timeout=self.handle_timeout)

        self.sound_player = QSoundEffect()
        self.sound_player.setSource(QUrl.fromLocalFile("resources/short_alarm.wav"))

        self.title_bar = CustomTitleBar.CustomTitleBar(self)

        # list for bbox frames to include in mask
        self.bbox_widgets = []

        self.currentFrameColor = -1
        self.multipleClasses = False

        # ensure that the widget always stays on top, no matter what
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)

        # init model and model thread
        self.model = None
        self.run_thread_init_task()

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
        self.grabWidget.setLayout(QtWidgets.QVBoxLayout())

        layout.addWidget(self.grabWidget)

        # Margins for frame to resize correctly
        self.setContentsMargins(2, 2, 2, 2)

    def set_logger(self):
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        memory_handler = MemoryHandler(1024*100, flushLevel=logging.ERROR, flushOnClose=True)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)

        self.logger = logging.getLogger('main_logger')
        logger.addHandler(handler)
        logger.addHandler(memory_handler)
        logger.setLevel(logging.INFO)
        print()

        return logger
    def handle_timeout(self):
        # some logic and start image processing in model thread
        self.timer_signal.emit()

    def check_bboxes(self):
        global current_bboxes

        # threading, check if model bboxes are ready and can be visualized
        if current_bboxes is None:
            self.logger.debug("Main: Tried to check bboxes, but they are None type")
            # print("Main: Tried to check bboxes, but they are None type")
            return

        if len(current_bboxes) == 0:
            self.logger.debug("Main: Tried to check bboxes, but there are no bboxes")
            # print("Main: Tried to check bboxes, but there are no bboxes")
            self.update_mask()
            return

        # if answers ready - visualise
        if mutex.tryLock():
            self.logger.debug("Main: locked bboxes, updating interface")
            # print("Main: locked bboxes, updating interface")

            self.update_mask()

            self.logger.debug("Main: unlocking bboxes")
            # print("Main: unlocking bboxes")
            mutex.unlock()

    def show_model_init_error(self):
        pass


    def set_frame_color(self, object_class: int, multiple_classes: bool = False):
        """
        Set frame color if car is detected or other signal sent
        :param object_class:
        :param multiple_classes:
        :return:
        """
        frame_palette = self.palette()

        if object_class != -1 and object_class == self.currentFrameColor:
            return

        if object_class == -1:
            frame_palette.setColor(self.backgroundRole(), getattr(Qt, "gray"))
            self.title_bar.set_color("gray")
            self.setPalette(frame_palette)
            return

        if multiple_classes:
            color = self.objectColorsDict[object_class]
            try:
                frame_palette.setColor(self.backgroundRole(), getattr(Qt, color))
                self.title_bar.set_color(color)
            except AttributeError:
                # print(f"Attribute error, color {color} for {object_class} class does not exist.")
                self.logger.error(f"Attribute error, color {color} for {object_class} class does not exist.")

        # Single class - emergency car
        else:
            color = "red"
            frame_palette.setColor(self.backgroundRole(), getattr(Qt, color))
            self.title_bar.set_color(color)
        self.setPalette(frame_palette)

    def take_screenshot(self):
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
        # # gebug save
        # screenshot.save('shot.jpg', 'jpg')

        return screenshot

    def send_to_predict(self, image):
        # convert Qpixmap to PIL
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, "PNG")
        pil_im = Image.open(io.BytesIO(buffer.data()))
        # pil_im.show()

        self.logger.debug("Sending image to model to process")
        # print("Sending image to model to process")
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

        self.bbox_widgets.clear()
        self.grabWidget.layout().children().clear()
        if (current_bboxes is not None) and (len(current_bboxes) > 0):
            for i in range(len(current_bboxes)):
                try:
                    # bbox -> [detections.xyxy[i], labels[i], conf[i], class_id[i]]
                    bbox_class = current_bboxes[i][1]
                    bbox_conf = current_bboxes[i][2]
                    bbox_color = self.objectColorsDict[current_bboxes[i][3]]
                    # color the frame
                    self.set_frame_color(bbox_color)

                    # Get outer border of detected + header for mask
                    bbox_xyxy = current_bboxes[i][0]
                    bbox_width = bbox_xyxy[2] - bbox_xyxy[0]
                    bbox_height = bbox_xyxy[3] - bbox_xyxy[1]

                    # create and add bbox widget
                    if self.title_bar.is_showing_class_name:
                        bbox_title = f"{bbox_class} {bbox_conf}"
                    else:
                        bbox_title = f"{bbox_conf}"

                    bbox = BBoxWidget.BBoxWidget(self, title=bbox_title, color=bbox_color, coords=(bbox_xyxy[0], bbox_xyxy[1]), size=(bbox_width, bbox_height))
                    self.bbox_widgets.append(bbox)
                    self.grabWidget.layout().addChildWidget(bbox)
                    if (not self.sound_player.isPlaying() and bbox_class != "non emergency car"):
                        self.sound_player.play()
                    bbox.show()

                    bbox_region = QtGui.QRegion(int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_width), int(bbox_height))
                    bbox_region.translate(self.contentsMargins().left(), titlebar_geometry.bottom() + self.contentsMargins().top()*2 + 1)

                    region += bbox_region

                    # make transparent window in bbox
                    transparent_geometry = bbox.transparent_window.geometry()
                    transparent_geometry.translate(bbox_region.rects()[0].x(), bbox_region.rects()[0].y()-bbox.contentsMargins().top())
                    region -= QtGui.QRegion(transparent_geometry)
                except Exception:
                    self.logger.error("Error occurred while drawing mask ")
        else:
            self.set_frame_color(-1)

        # draw bboxes and frame
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


    def run_thread_init_task(self):
        # delete existing thread if present
        if type(self.thread) is QThread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
            # Finally, detach (and probably garbage collect) the objects
            # used by this class.
            del self.worker
            del self.thread

        self.thread = QThread()
        self.thread.__PCHthreadName = "ModelThread"
        self.worker = ModelWorker(self, self.model_name)
        self.worker.moveToThread(self.thread)

        # Init model at first
        self.thread.started.connect(self.worker.init_model)
        self.worker.signals.finished_init.connect(self.timer.start)
        self.worker.signals.finished_init.connect(self.title_bar.hide_warning)
        # Start processing images after model init and according to timer
        self.timer_signal.connect(self.worker.process_image)

        self.worker.signals.updated_bboxes.connect(self.check_bboxes)

        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.signals.error.connect(self.show_model_init_error)

        self.logger.info(f"Main: starting {self.model_name} model init thread")
        self.thread.start()


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
    '''

    def __init__(self, parent, model_name):
        super().__init__()
        self.parent = parent
        self.logger = parent.logger
        self.model_name = model_name
        self.signals = WorkerSignals()
        self.mutex = QMutex()

    def process_image(self):

        # global current_image
        global current_bboxes
        # look on current image, lock it
        if self.mutex.tryLock():
            self.logger.debug(f"{self} started to process image")
            # print(f"{self} started to process image")
            # send it to process
            image = self.parent.take_screenshot()
            results = self.parent.send_to_predict(image)
            # after finished get bboxes and update locked bbox_storage
            current_bboxes = results
            self.parent.title_bar.update_fps(self.parent.model.last_fps)
            self.mutex.unlock()
            self.signals.updated_bboxes.emit()
        else:
            self.logger.debug(f"ModelWorker: mutex is already locked")
            # print(f"ModelWorker: mutex is already locked")
        # unlock storage and notify

        self.logger.debug(f"ModelWorker: ended to process image")
        # print(f"ModelWorker: ended to process image")

    def init_model(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        try:
            # print("starting model")
            self.parent.model = ECModel.ECModel(self.parent, self.model_name)
            self.parent.model.moveToThread(self.parent.thread)
        except Exception:
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
