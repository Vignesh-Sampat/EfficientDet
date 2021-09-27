import os
import pandas as pd
import xmltodict


def get_file_directories(path):
    file_dir = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        filenames = sorted(filenames)
        for file in filenames:
            file_dir.append(dirpath+'/'+file)
    return file_dir

xml_paths = sorted(get_file_directories('/Path to Xlms'))

def extract_cordinates(xml_paths):
    bbox = []
    image_name = []
    for item in xml_paths:
        x = xmltodict.parse(open(item, 'rb'))
        item_bbx = []
        anno_dict = dict(x['annotation'])
        if list(anno_dict['object'])[0] != 'name':
            for i in range(len(x['annotation']['object'])):
                bndbox = x['annotation']['object'][i]['bndbox']
                bndbox = [int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])]
                item_bbx.append(bndbox)
        else:
            bndbox = x['annotation']['object']['bndbox']
            bndbox = [int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])]
            item_bbx.append(bndbox)
        bbox.append(item_bbx)
        image_name.append(x['annotation']['filename'])


    data_frame1 = pd.DataFrame(columns=['x_min', 'y_min', 'x_max', 'y_max','image_name','class_name'])
    for items,image_path in zip(bbox,image_name):
        for item in items:
            data_frame = pd.DataFrame([item],columns=['x_min','y_min','x_max','y_max'])
            data_frame['image_name'] = str(os.path.basename(image_path))
            data_frame['class_name'] = 'Defect'
            data_frame1 = pd.concat([data_frame1,data_frame])
    return data_frame1

original_data = extract_cordinates(xml_paths,image_paths)
original_data.to_csv('paths_to_save_csv/original_data_dataframe2.csv',index=False)


