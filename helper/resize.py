import os.path

import cv2
import albumentations as A
import pandas as pd
from PIL import Image

original_data = pd.read_csv('/media/Transcend/Main-Project/Linear Model/data/All_data/original_data_dataframe2.csv')

groupby_data = original_data.groupby('image_name')

Height = []
width = []
image_name = []
for item in groupby_data:
    image_path = item[0]
    image = cv2.imread('/media/Transcend/Main-Project/Linear Model/data/resized_images/' + image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image)
    im = im.resize((600, 600))
    im.save('/media/Transcend/Main-Project/Linear Model/data/resized_images/' + os.path.splitext(image_path)[0]+'.png')

