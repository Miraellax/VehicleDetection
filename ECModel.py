import asyncio
import math

import PIL
import numpy as np
import torch
import ultralytics
from PIL import Image
from PyQt5.QtCore import QObject
from torchvision import transforms
from torchvision.transforms.v2 import Compose
from ultralytics import YOLO
import logging
import supervision as sv

logging.basicConfig(format='"%(asctime)s [%(levelname)s] %(name)s: %(message)s"',
                    filename='Main.log',
                    # filemode='w', # if need to erase logs each run
                    )

weights_path = "C:\\Users\\Alex\\PycharmProjects\\Cursach3\\runs\\detect\\train11\\weights\\best.pt"


class ECModel(QObject):
    # simple constructor
    threshold = 0.6
    def __init__(self):
        super().__init__()
        # check if file is legit or shut the app, or download not trained model
        try:
            self.model = YOLO(weights_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            # print("Cant load model")
            logging.warning("Model: couldn't load the model weights")
            # print("Model: couldn't load the model weights")
        #     logging
        # if model ok - logging

    def image_augmentation_no_padding(self, image, size):


        augmentation = Compose([
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor()
        ])

        result = augmentation(image)
        # [batch_size, channels, height, width]
        result = result.permute(2, 0, 1).unsqueeze(0)

        return result

    def get_augment_params_no_padding(self, image):
        #     new_width, new_height
        new_width = math.ceil(image.width / 32) * 32
        new_height = math.ceil(image.width / 32) * 32

        return (new_width, new_height)

    def get_augment_params(self, image):
        #     new_width, new_height, padding/2, is_wider
        width = image.width
        height = image.height
        if width >= height:
            is_wider = True
            new_width = math.ceil(width / 32) * 32
            new_height = math.ceil(height / 2) * 2
            padding = (math.ceil(new_height / 32) * 32 - height) // 2
        else:
            is_wider = False
            new_width = math.ceil(width / 2) * 2
            new_height = math.ceil(height / 32) * 32
            padding = (math.ceil(new_width / 32) * 32 - width) // 2

        return new_width, new_height, padding, is_wider

    def image_augmentation(self, image, params, padding=True):

        if params[3] == True:
            wider_augmentation = Compose([
                transforms.Resize((params[1], params[0])),
                transforms.Pad((0, params[2])),
                transforms.ToTensor()
            ])
            result = wider_augmentation(image)
            # [batch_size, channels, height, width]
            result = result.unsqueeze(0)

            return result

        taller_augmentation = Compose([
            transforms.Resize((params[1], params[0])),
            transforms.Pad((params[2], 0)),
            transforms.ToTensor()
        ])

        result = taller_augmentation(image)
        # [batch_size, channels, height, width]
        result = result.permute(2, 0, 1).unsqueeze(0)

        return result

    def predict(self, image: PIL) -> np.ndarray:
        logging.info("Model: Got image, starting to process")
        # print("Model: Got image, starting to process")

        # aug_params = self.get_augment_params(image)
        # image_tensor = self.image_augmentation(image, aug_params).to(self.device)
        # results = self.model(image_tensor)


        aug_params = self.get_augment_params_no_padding(image)
        image_tensor = self.image_augmentation_no_padding(image, aug_params).to(self.device)
        results = self.model(image_tensor)

        logging.log(f"Model: got predictions\n{results}")
        # print(f"Model: got predictions\n{results}")
        #     log results and time
        return results

    def process_image(self, image: PIL):
        results = self.model.predict(image)[0]

        # Detections(xyxy=array([[46.362, 43.064, 233.17, 129.77]], dtype=float32), mask=None, confidence=array([0.94702], dtype=float32), class_id=array([1]), tracker_id=None)
        detections = sv.Detections.from_ultralytics(results)

        # ['fire fighting vehicle 0.95']
        labels = [f"{self.model.names[class_id]}" for _, _, _, class_id, _ in detections]
        conf = [f"{confidence:0.2f}" for _, _, confidence, _, _ in detections]
        class_id = [class_id for _, _, _, class_id, _ in detections]

        bboxes = []
        for i in range(len(detections.confidence)):
            if float(conf[i]) >= self.threshold:
                bboxes.append([detections.xyxy[i], labels[i], conf[i], class_id[i]])

        logging.info(f"Model: bboxes are ready, sending them\n{bboxes}")
        # print(f"Model: bboxes are ready, sending them\n{bboxes}")
        return bboxes

    def set_model_threshold(self, new_threshold: float):
        if 0 <= new_threshold <= 1:
            self.model.threshold = new_threshold
