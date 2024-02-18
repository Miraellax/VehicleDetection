import math

import PIL
import torch
import ultralytics
from PIL import Image
from torchvision import transforms
from torchvision.transforms.v2 import Compose
from ultralytics import YOLO
import logging

logging.basicConfig(format='%(asctime)s %(message)s',
                    filename='example.log',
                    encoding='utf-8',
                    # filemode='w', # if need to erase logs each run
                    )

weights_path = "C:\\Users\\Alex\\PycharmProjects\\Cursach3\\runs\\detect\\train11\\weights\\best.pt"

def get_augment_params(image):
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


def image_augmentation(image, params, padding=True):
    wider_augmentation = Compose([
        transforms.Resize((params[1], params[0])),
        transforms.Pad((0, params[2])),
        transforms.PILToTensor(),
    ])
    taller_augmentation = Compose([
        transforms.Resize((params[1], params[0])),
        transforms.Pad((params[2], 0)),
        transforms.PILToTensor(),
    ])
    if params[3] == True:
        return wider_augmentation(image)
    return taller_augmentation(image)
class ECModel:
    def __init__(self):
        # check if file is legit or shut the app, or download not trained model
        try:
            self.model = YOLO(weights_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            print("Cant load model")
        #     logging
        # if model ok - logging


    def predict(self, image):
        # resize to
        image_tensor = torch.from_numpy(image).to(self.device)
        print(self.device)
        results = self.model(image_tensor, visualize=True)
        print(results)
        #     SendSignal(results)
        #     log results and time
        return results.cpu()

    def send_signal(self, pred):
        # send code for class and play sound?
        pass

i = Image.open("./shot.jpg")
params = get_augment_params(i)
print(params)
sss = transforms.ToPILImage()
a = sss(image_augmentation(i, params))
a.show()
