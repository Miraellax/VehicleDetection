import numpy as np

# Save
weights_dictionary = {
    "YOLOv8s": "./resources/model_weights/YOLOv8s_11.pt",
    "YOLOv8n": "./resources/model_weights/YOLOv8n_13.pt",
    "YOLOv8m": "./resources/model_weights/YOLOv8m_22.pt",
    "YOLOv5n": "./resources/model_weights/YOLOv5n_24.pt",
    }

np.save('resources/model_weight_dict.npy', weights_dictionary)

'''
# Load
read_dictionary = np.load('my_file.npy',allow_pickle=True).item()
print(read_dictionary['hello']) # displays "world"
'''
