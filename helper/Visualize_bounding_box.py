import bbox_visualizer as bbv
import cv2
import pandas as pd

data = pd.read_csv('/media/sailab/Transcend/ERREKA_MAGNAUTO/2. Main-Project/Linear Model/data/train/all_data.csv',header=None)
print(data.head(20))

for item in data.groupby(0):
    print(item[1].loc[:,1:4].values.tolist())

    img = cv2.imread('/media/sailab/Transcend/ERREKA_MAGNAUTO/2. Main-Project/Linear Model/data/train/' + item[0])
    bbxes = item[1].loc[:, 1:4].values.astype('int').tolist()

    for bbx in bbxes:
        img = bbv.draw_rectangle(img, bbx)


    cv2.imwrite('/media/sailab/Transcend/ERREKA_MAGNAUTO/2. Main-Project/Linear Model/data/check/' + item[0], img)
