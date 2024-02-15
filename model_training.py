import os
import random

import torch
from xml.etree import ElementTree as ET
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import torchvision
import torchvision.models
from torchvision.models import ResNet50_Weights
from tqdm.notebook import tqdm
from torch import nn
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import torchvision.transforms as transforms
import yaml

# Путь к данным обучения
data_path = "./Emergency Vehicles Russia.v1i.yolov8"


def seed_everything(seed):
    """
        Фисксация всех сидов в программе для корректного
        сравнения оптимизаторов и обучаемых моделей

        :seed: число для фиксации сидов
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 42
seed_everything(seed)

# Выбор устройства для работы модели
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

# Проверка доступности гпу
# print(device)


def ReadClassDict(path):
    '''
        Загружаем список классов изображений из yaml файла

    :param path:
    :return:
    '''

    with open(path, "r") as f:
        file = yaml.safe_load(f)
        class_list = file["names"]
        class_dict = dict(enumerate(class_list, start=0))

    return class_dict

class_dict = ReadClassDict("./" + data_path + "/data.yaml")

# print(class_dict) # {0: 'ambulance', 1: 'fire fighting vehicle', 2: 'intensive care unit', 3: 'non emergency car', 4: 'police car'}

def collate_fn(batch, downsample=32):
    imgs, batch_boxes = map(list, (zip(*[(b["image"], b["bboxes"]) for b in batch])))

    imgs = torch.stack(imgs)
    b, _, h, w = imgs.shape

    target = imgs.new_zeros(b, 6, h // downsample, w // downsample)

    # Add sample index to targets
    for i, boxes in enumerate(batch_boxes):
        xmin, ymin, xmax, ymax, classes = map(
            torch.squeeze, torch.split(imgs.new_tensor(boxes), 1, dim=-1)
        )

        # Нормализуйте ширину и высоту, поделив на ширину и высоту исходного изображения
        w_box = (xmax - xmin)/512 # ширина бокса отнормированная
        h_box = (ymax - ymin)/512 # высота бокса отнормированная

        # координаты центра и сдвиги
        cx =  (xmin + xmax)/2
        cy =  (ymin + ymax)/2

        cx_idx = (cx // downsample).to(torch.long) # индекс центра на карте признаков размера 16x16. Это будут как раз координаты пикселя, куда мы запишем параметры коробки
        cy_idx = (cy // downsample).to(torch.long) # .to(torch.long)

        cx_box = cx - (cx//downsample)*downsample # сдивиги относительно cx_idx
        cy_box = cy - (cy//downsample)*downsample # сдивиги относительно cy_idx

        target[i, :, cy_idx, cx_idx] = torch.stack(
            [cx_box, cy_box, w_box, h_box, torch.ones_like(cx_box), classes]
        )

    return {"image": imgs, "target": target}


def get_txt_data(image_path, class_dict):
    """
        Получение списка данных по всем bbox'ам на изображении

        :image_name: имя файла
        :path: путь (test train valid)
        :class_dict: словарь с расшифровкой классов
        :return:
    """
    # TODO заменить на убирание всех форматов лишних, не только jpg
    if image_path.rfind(".jpg"):
        image_path = image_path.replace(".jpg", "")

    # читаем соответствующий изображению txt
    with open(str(image_path).replace("images", "labels") + ".txt", "r") as f:
        # итерация через bbox'ы объектов в файле и их сохранение в один массив
        bboxes = []
        for line in f:
            data = line.split(sep=" ") # class_id center_x center_y width height
            # print(data)
            center_x = float(data[1])
            center_y = float(data[2])
            width = float(data[3])
            height = float(data[4])
            card_class = class_dict[int(data[0])]

            res = (center_x, center_y, width, height, card_class)
            bboxes.append(res)
        return bboxes


# Проверка работы
# print(get_txt_data("Emergency Vehicles Russia.v1i.yolov8/train/images/0c273b07-_44_PNG.rf.24d358e5a9859dde93cf9823287e0f94.jpg", class_dict))
# print(get_txt_data("./Emergency Vehicles Russia.v1i.yolov8/train/images\-10_PNG.rf.0f13a65f5ba37714e528b76181b3dee8.jpg", class_dict))


# Класс датасета для доступа к данным во время обучения
class PascalDataset(torch.utils.data.Dataset):
    def __init__(self, *, transform, root=data_path, data_type_path="train", seed=seed):
        self.root = Path(root)
        self.transform = transform

        assert self.root.is_dir(), f"No data at `{root}`"

        # Проверка на корректный путь TODO
        if data_type_path not in ["train", "test", "valid"]:
            data_type_path = "train"
        self.filenames = np.array(glob.glob(root + "/" + data_type_path + "/images/*"))

        self.class_dict = ReadClassDict("./" + data_path + "/data.yaml")
        self.class_dict_inverted = {v: k for k, v in self.class_dict.items()}

        np.random.seed(seed)
        # Перестановка файлов для рандома, сид зафиксирован
        permutation = np.random.permutation(len(self.filenames))

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = np.asarray(Image.open(fname))
        bboxes = get_txt_data(fname, self.class_dict)

        return self.transform(image=image, bboxes=bboxes)

    def __get_raw_item__(self, idx):
        fname = self.filenames[idx]
        return fname, get_txt_data(fname, self.class_dict)

    def __len__(self):
        return len(self.filenames)

# Определение нормализации и аугментаций изображений для обучения
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# train_transform = A.Compose(
#     [
#         A.Resize(512, 512),
#         A.augmentations.transforms.Normalize(mean=mean, std=std),
#         ToTensorV2(),
#     ],
#     bbox_params=dict(format="pascal_voc", min_visibility=0.3),
# )
#
# test_transform = A.Compose(
#     [
#         A.Resize(512, 512),
#         A.augmentations.transforms.Normalize(mean=mean, std=std),
#         ToTensorV2(),
#     ],
#     bbox_params=dict(format="pascal_voc", min_visibility=0.5),
# )
train_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.augmentations.transforms.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ],
    bbox_params=dict(format="yolo", min_visibility=0.3),
)

test_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.augmentations.transforms.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ],
    bbox_params=dict(format="yolo", min_visibility=0.5),
)

# Применение аугментаций и создание датасетов для обучения и тестирования
train_ds = PascalDataset(root=data_path, transform=train_transform, data_type_path="train")
test_ds = PascalDataset(root=data_path, transform=test_transform, data_type_path="train")


print(train_ds.__getitem__(0))

def visualize(images, bboxes, mean, std):
    '''
        Функция для визуализации размеченных данных, изображение и bbox объект

        :param images:
        :param bboxes:
        :param mean:
        :param std:
        :return:
    '''
    fig, axes = plt.subplots(
        2, len(images) // 2 + len(images) % 2, figsize=(10, 8), dpi=100
    )

    for i, ax in enumerate(axes.reshape(-1)):

        ax.axis(False)
        if i >= len(images):
            break
        # Смена порядков каналов изображения из torch для отображения в matplotlib
        im = images[i].permute(1, 2, 0)

        # # Откат нормализации, обратное преобразование
        # im[:,:,0] = im[:,:,0].mul(std[0]) + mean[0]
        # im[:,:,1] = im[:,:,1].mul(std[1]) + mean[1]
        # im[:,:,2] = im[:,:,2].mul(std[2]) + mean[2]

        ax.imshow(im, vmin=0, vmax=1)

        for bbox in bboxes[i]:
          # bbox = (center_x center_y width height card_class)
          class_name = bbox[4]

          w = bbox[2] * im.shape[0]
          h = bbox[3] * im.shape[1]

          xmin = (bbox[0] * im.shape[0])- w//2
          ymin = (bbox[1] * im.shape[1]) - h//2

          p = plt.Rectangle((xmin, ymin), w, h, fill=False, color="red")
          ax.add_patch(p)

          ax.text(xmin+5, ymin-5, class_name, size=10, color='red')

    fig.tight_layout()
    plt.show()

# Визуализация данных
out = [train_ds[i] for i in range(6)]
visualize([o["image"] for o in out], [o["bboxes"] for o in out], mean, std)


#
# def decode_prediction(pred, upsample=32, threshold=0.7):
#     # b - batch size, c - class channels
#     b, c, h, w = pred.shape
#     # print("c", c)
#     # print(pred.shape)
#     img_w, img_h = w * upsample, h * upsample
#
#     res =[]
#
#     for image_num in range(b):
#       # print("im_num", image_num)
#       # print("im", pred[image_num].shape)
#
#       bboxes = []
#       # [0-cx_box, 1-cy_box, 2-w_box, 3-h_box, 4-confidence, 5+-classes]
#
#       # Индексы клеток где confidence >= threshold и есть центр bbox'a
#       indices = (pred[image_num][4] > threshold).nonzero()
#       # print("ind", indices)
#       # print("ind len", len(indices))
#
#       # Для координат каждой клетки с центром найдем параметры bbox'a
#       for center in indices:
#         w_box = (pred[image_num][2][center[0]][center[1]].item())*512 # ширина бокса
#         h_box = (pred[image_num][3][center[0]][center[1]].item())*512 # высота бокса
#
#         # Сдвиг центра в клетке
#         cx_box = pred[image_num][0][center[0]][center[1]].item()
#         cy_box = pred[image_num][1][center[0]][center[1]].item()
#
#         # Координаты центра в изображении
#         cx = (center[1].item()*upsample + cx_box)
#         cy = (center[0].item()*upsample + cy_box)
#
#         xmin = (cx - w_box/2)
#         xmax = (cx + w_box/2)
#         ymin = (cy - h_box/2)
#         ymax = (cy + h_box/2)
#         # print("w_box", w_box)
#         # print("h_box", h_box)
#         # print("cx_box", cx_box)
#         # print("cy_box", cy_box)
#         # print("cx", cx)
#         # print("cy", cy)
#         # print("xmin", xmin)
#         # print("xmax", xmax)
#         # print("ymin", ymin)
#         # print("ymax", ymax)
#
#         if c == 6: #target
#           card_class = int(pred[image_num][5][center[0]][center[1]].item())
#         else: #prediction
#           # print(pred)
#           # print("size", pred[image_num].shape)
#           # print("center", center[0].item(), center[1].item())
#           card_class = 0
#           # card_class = torch.argmax(pred[image_num][5:][center[0].item()][center[1].item()-5], dim=0)
#
#
#         bbox = (xmin, ymin, xmax, ymax, card_class)
#         # print("bbox", bbox)
#         # print(bbox)
#         # # Если предсказание - добавляем ббокс только если уверенность больше-равна threshold
#         # if c==6 or (c > 6 and pred[image_num][5+card_class] >= threshold):
#         bboxes.append(bbox)
#       res.append(bboxes)
#     print(len(res))
#     return res
#
# # Проверка работы кодировщика и декодировщика
# test_bboxes = decode_prediction(test_col["target"])
# visualize(test_col["image"], test_bboxes)
#
# def annotation2txt(bboxes, w_im, h_im):
#     # (xmin, ymin, xmax, ymax, class in dict) -> [class_id center_x center_y width height]
#     res = []
#     for box in bboxes:
#       s = f"{box[4]} {((box[2] + box[0])/2)/w_im} {((box[3] + box[1])/2)/h_im} {(box[2] - box[0])/w_im} {(box[2] - box[0])/h_im}"
#       res.append(s)
#
#     return res
#
# import ultralytics
# from ultralytics import YOLO
#
# # Обучение с нуля - save_dir: PosixPath('runs/detect/train5')
# model = YOLO('yolov8n.yaml')
# model.train(data='/content/data.yaml', epochs=100, imgsz=512, device=device)
#
# # Загрузка весов из обучения выше
# # model = YOLO('runs/detect/train5/weights/best.pt')
#
# from IPython.display import Image
# Image('/content/runs/detect/train/results.png')
#
# imgs = []
# bxs = []
#
# results = model.predict(images, save=True, imgsz=[512, 512], conf=0.5, rect=True)
# for result in results:
#     box_res = []
#     # Собираем данные о бибоксах и картинки в списки для visualize
#     # bbox = (xmin, ymin, xmax, ymax, card_class)
#     boxes = result.boxes.xyxy.tolist()
#     im = torch.from_numpy(result.orig_img.astype(np.uint8))
#     im = im.permute(2, 0, 1)
#     cls = result.boxes.cls.tolist()
#
#     for box in range(len(boxes)):
#       b = (boxes[box][0], boxes[box][1], boxes[box][2],  boxes[box][3], int(cls[box]))
#       box_res.append(b)
#
#     imgs.append(im)
#     bxs.append(box_res)
#
# visualize(imgs, bxs)