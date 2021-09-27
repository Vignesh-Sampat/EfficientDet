import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
import xmltodict
import os
import pandas as pd

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White

Type = 'IAAFlipud' #change here


original_data = pd.read_csv('/media/sailab/Transcend/ERREKA_MAGNAUTO/2. Main-Project/Linear Model/data/All_data/original_data_dataframe_resize_600.csv')

save_directory = '/media/sailab/Transcend/ERREKA_MAGNAUTO/2. Main-Project/Linear Model/data/Augmented/' + Type

def transform_IAAFlipud(csv_data,save_directory): #change here
    transform = A.Compose([A.IAAFlipud(always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']), ) #change here
    groupby_data = csv_data.groupby('image_name')
    data = []
    for item in groupby_data:
        image_path = item[0]
        filename = save_directory + '/' + os.path.splitext(image_path)[0] + '_' + Type + '.png'
        bboxes = item[1][['x_min', 'y_min', 'x_max', 'y_max']].values
        image = cv2.imread('/media/sailab/Transcend/ERREKA_MAGNAUTO/2. Main-Project/Linear Model/data/resized_images/' + image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        category_ids = np.ones(len(bboxes)).tolist()
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        cv2.imwrite(filename, transformed['image'])
        for i in range(len(transformed['bboxes'])):
            x_min, y_min, x_max, y_max = transformed['bboxes'][i]
            data.append((os.path.basename(filename), x_min, y_min, x_max, y_max, 'Defect'))
    data_frame = pd.DataFrame(data, columns=['image_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name'])
    data_frame.to_csv(save_directory + '/'+ Type + '_dataframe.csv', index=False)
    return data_frame


transform_IAAFlipud(original_data,save_directory)

