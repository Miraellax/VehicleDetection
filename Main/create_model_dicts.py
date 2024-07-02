import os
import numpy as np


def create_dicts():

    absolute_path = os.path.dirname(__file__)
    # Save
    weights_dictionary = {
        "YOLOv8s": absolute_path + "/resources/model_weights/YOLOv8s_11.pt",
        "YOLOv8n": absolute_path + "/resources/model_weights/YOLOv8n_13.pt",
        "YOLOv8m": absolute_path + "/resources/model_weights/YOLOv8m_22.pt",
        "YOLOv5n": absolute_path + "/resources/model_weights/YOLOv5n_24.pt",
        }

    np.save(absolute_path + '/resources/model_weight_dict.npy', weights_dictionary)

'''
# Load
read_dictionary = np.load('my_file.npy',allow_pickle=True).item()
print(read_dictionary['hello']) # displays "world"
'''
