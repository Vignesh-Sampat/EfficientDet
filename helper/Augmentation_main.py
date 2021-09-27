import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import xmltodict
import os
import pandas as pd

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def get_file_directories(path):
    file_dir = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        filenames = sorted(filenames)
        for file in filenames:
            file_dir.append(dirpath+'/'+file)
    return file_dir

image_paths = get_file_directories('/working_directory/Errekka/Model-1/png') # Give images path here
xml_paths = ['/working_directory/Errekka/Model-1/labels/'+ os.path.splitext(os.path.basename(i))[0]+'.xml' for i in image_paths]

-
def extract_cordinates(xml_paths,image_paths):
    bbox = []
    for item in xml_paths:
        x = xmltodict.parse(open(item, 'rb'))
        bndbox = x['annotation']['object']['bndbox']
        bndbox = [int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])]
        bbox.append(bndbox)
    image_paths = pd.Series(image_paths)
    data_frame = pd.DataFrame(bbox,columns=['x_min','y_min','x_max','y_max'])
    data_frame['image_name'] = image_paths
    data_frame['class_name'] = 'Defect'
    return data_frame

original_data = extract_cordinates(xml_paths,image_paths)

category_ids = [1]


def transform_horizontalFlip(csv_data,save_directory):
    transform = A.Compose([A.HorizontalFlip(p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'HorizontalFlip' + '.png'
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    return data_frame

horizontalFlip_dataframe = transform_horizontalFlip(original_data, '/working_directory/Errekka/Model-1/HorizontalFlip')
horizontalFlip_dataframe.to_csv('/working_directory/Errekka/Model-1/HorizontalFlip/horizontalFlip_dataframe.csv',index=False)

def transform_ShiftScaleRotate(csv_data,save_directory):
    transform = A.Compose([A.ShiftScaleRotate(p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'ShiftScaleRotate' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    return data_frame

ShiftScaleRotate_dataframe = transform_ShiftScaleRotate(original_data, '/working_directory/Errekka/Model-1/ShiftScaleRotate')
ShiftScaleRotate_dataframe.to_csv('/working_directory/Errekka/Model-1/ShiftScaleRotate/ShiftScaleRotate_dataframe.csv',index=False)



def transform_CenterCrop(csv_data,save_directory):
    transform = A.Compose([A.CenterCrop(height=1200, width=1200, p=1)],bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.3, label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'CenterCrop' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    return data_frame

CenterCrop_dataframe = transform_CenterCrop(original_data, '/working_directory/Errekka/Model-1/CenterCrop')
CenterCrop_dataframe.to_csv('/working_directory/Errekka/Model-1/CenterCrop/CenterCrop_dataframe.csv',index=False)


def transform_CenterCrop_1500(csv_data,save_directory):
    transform = A.Compose([A.CenterCrop(height=1500, width=1500, p=1)],bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.3, label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'CentreCrop_1500' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    return data_frame

CentreCrop_1500_dataframe = transform_CenterCrop_1500(original_data, '/working_directory/Errekka/Model-1/CentreCrop_1500')
CentreCrop_1500_dataframe.to_csv('/working_directory/Errekka/Model-1/CentreCrop_1500/CentreCrop_1500_dataframe.csv',index=False)


def transform_IAAAffine(csv_data,save_directory):
    transform = A.Compose([A.IAAPerspective (scale=(0.05, 0.1), keep_size=True, always_apply=False, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'IAAAffine' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    return data_frame

IAAAffine_dataframe = transform_IAAAffine(original_data, '/working_directory/Errekka/Model-1/IAAAffine')
IAAAffine_dataframe.to_csv('/working_directory/Errekka/Model-1/IAAAffine/IAAAffine_dataframe.csv',index=False)


def transform_CenterCrop_1400(csv_data,save_directory):
    transform = A.Compose([A.CenterCrop(height=1400, width=1400, p=1)],bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.3, label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'CentreCrop_1400' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name','x_min', 'y_min', 'x_max', 'y_max','class_name']]
    return data_frame

CentreCrop_1400_dataframe = transform_CenterCrop_1400(original_data, '/working_directory/Errekka/Model-1/CentreCrop_1400')
CentreCrop_1400_dataframe.to_csv('/working_directory/Errekka/Model-1/CentreCrop_1400/CentreCrop_1400_dataframe.csv',index=False)

def transform_FancyPCA(csv_data,save_directory):
    transform = A.Compose([A.FancyPCA(alpha=0.1,always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'FancyPCA' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

FancyPCA_dataframe = transform_FancyPCA(original_data, '/working_directory/Errekka/Model-1/FancyPCA')
FancyPCA_dataframe.to_csv('/working_directory/Errekka/Model-1/FancyPCA/FancyPCA_dataframe.csv',index=False)

def transform_Blur(csv_data,save_directory):
    transform = A.Compose([A.Blur(blur_limit=7,always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'Blur' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

Blur_dataframe = transform_Blur(original_data, '/working_directory/Errekka/Model-1/Blur')
Blur_dataframe.to_csv('/working_directory/Errekka/Model-1/Blur/Blur_dataframe.csv',index=False)

def transform_ColorJitter_1(csv_data,save_directory):
    transform = A.Compose([A.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0,always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'ColorJitter_1' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

ColorJitter_1_dataframe = transform_ColorJitter_1(original_data, '/working_directory/Errekka/Model-1/ColorJitter_1')
ColorJitter_1_dataframe.to_csv('/working_directory/Errekka/Model-1/ColorJitter_1/ColorJitter_1_dataframe.csv',index=False)


def transform_ColorJitter_2(csv_data,save_directory):
    transform = A.Compose([A.ColorJitter(brightness=0,contrast=0.5,saturation=0,hue=0,always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'ColorJitter_2' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

ColorJitter_2_dataframe = transform_ColorJitter_2(original_data, '/working_directory/Errekka/Model-1/ColorJitter_2')
ColorJitter_2_dataframe.to_csv('/working_directory/Errekka/Model-1/ColorJitter_2/ColorJitter_2_dataframe.csv',index=False)


def transform_ColorJitter_3(csv_data,save_directory):
    transform = A.Compose([A.ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0,always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'ColorJitter_3' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame


ColorJitter_3_dataframe = transform_ColorJitter_3(original_data, '/working_directory/Errekka/Model-1/ColorJitter_3')
ColorJitter_3_dataframe.to_csv('/working_directory/Errekka/Model-1/ColorJitter_3/ColorJitter_3_dataframe.csv',index=False)


def transform_RandomRotate90(csv_data,save_directory):
    transform = A.Compose([A.RandomRotate90(p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'RandomRotate90' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

RandomRotate90_dataframe = transform_RandomRotate90(original_data, '/working_directory/Errekka/Model-1/RandomRotate90')
RandomRotate90_dataframe.to_csv('/working_directory/Errekka/Model-1/RandomRotate90/RandomRotate90_dataframe.csv',index=False)


def transform_GaussianBlur(csv_data,save_directory):
    transform = A.Compose([A.GaussianBlur(p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'GaussianBlur' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

GaussianBlur_dataframe = transform_GaussianBlur(original_data, '/working_directory/Errekka/Model-1/GaussianBlur')
GaussianBlur_dataframe.to_csv('/working_directory/Errekka/Model-1/GaussianBlur/GaussianBlur_dataframe.csv',index=False)


def transform_GaussNoise(csv_data,save_directory):
    transform = A.Compose([A.GaussNoise(p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'GaussNoise' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

GaussNoise_dataframe = transform_GaussNoise(original_data, '/working_directory/Errekka/Model-1/GaussNoise')
GaussNoise_dataframe.to_csv('/working_directory/Errekka/Model-1/GaussNoise/GaussNoise_dataframe.csv',index=False)

def transform_IAAAdditiveGaussianNoise(csv_data,save_directory):
    transform = A.Compose([A.IAAAdditiveGaussianNoise(always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'IAAAdditiveGaussianNoise' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

IAAAdditiveGaussianNoise_dataframe = transform_IAAAdditiveGaussianNoise(original_data, '/working_directory/Errekka/Model-1/IAAAdditiveGaussianNoise')
IAAAdditiveGaussianNoise_dataframe.to_csv('/working_directory/Errekka/Model-1/IAAAdditiveGaussianNoise/IAAAdditiveGaussianNoise_dataframe.csv',index=False)

def transform_IAAEmboss(csv_data,save_directory):
    transform = A.Compose([A.IAAEmboss(always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'IAAEmboss' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

IAAEmboss_dataframe = transform_IAAEmboss(original_data, '/working_directory/Errekka/Model-1/IAAEmboss')
IAAEmboss_dataframe.to_csv('/working_directory/Errekka/Model-1/IAAEmboss/IAAEmboss_dataframe.csv',index=False)


def transform_IAAAffine(csv_data,save_directory):
    transform = A.Compose([A.IAAAffine(always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'IAAAffine' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

IAAAffine_dataframe = transform_IAAAffine(original_data, '/working_directory/Errekka/Model-1/IAAAffine')
IAAAffine_dataframe.to_csv('/working_directory/Errekka/Model-1/IAAAffine/IAAAffine_dataframe.csv',index=False)

def transform_IAACropAndPad(csv_data,save_directory):
    transform = A.Compose([A.IAACropAndPad(always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'IAACropAndPad' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

IAACropAndPad_dataframe = transform_IAACropAndPad(original_data, '/working_directory/Errekka/Model-1/IAACropAndPad')
IAACropAndPad_dataframe.to_csv('/working_directory/Errekka/Model-1/IAACropAndPad/IAACropAndPad_dataframe.csv',index=False)


def transform_IAAFliplr(csv_data,save_directory):
    transform = A.Compose([A.IAAFliplr(always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'IAAFliplr' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame

IAAFliplr_dataframe = transform_IAAFliplr(original_data, '/working_directory/Errekka/Model-1/IAAFliplr')
IAAFliplr_dataframe.to_csv('/working_directory/Errekka/Model-1/IAAFliplr/IAAFliplr_dataframe.csv',index=False)


def transform_IAAFlipud(csv_data,save_directory):
    transform = A.Compose([A.IAAFlipud(always_apply=True, p=1)],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    bbox = []
    file_names = []
    for index, item in csv_data.iterrows():
        image_path = item['image_name']
        filename = save_directory + '/'+ os.path.splitext(os.path.basename(image_path))[0]+ '_' + 'IAAFlipud' + '.png'
        print(filename)
        x_min = item['x_min']
        y_min = item['y_min']
        x_max = item['x_max']
        y_max = item['y_max']
        bboxes = [[x_min,y_min,x_max,y_max]]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        bboxes = [int(i) for i in list(transformed['bboxes'][0])]
        cv2.imwrite(filename, transformed['image'])
        bbox.append(bboxes)
        file_names.append(filename)
    data_frame = pd.DataFrame(bbox, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    data_frame['image_name'] = pd.Series(file_names)
    data_frame['class_name'] = 'Defect'
    data_frame = data_frame[['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    return data_frame


IAAFlipud_dataframe = transform_IAAFlipud(original_data, '/working_directory/Errekka/Model-1/IAAFlipud')
IAAFlipud_dataframe.to_csv('/working_directory/Errekka/Model-1/IAAFlipud/IAAFlipud_dataframe.csv',index=False)





original_data.to_csv('Original_dataframe.csv',index=False)











