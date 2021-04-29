import cv2
import numpy as np
# import glob
from tqdm import tqdm
import os
 
path = 'C:\\Users\\16200\\Desktop\\project_dataset\\frames\\soccer'
img_array = []

namelist = os.listdir(path)
namelist = sorted(namelist, key = lambda x: int(x[5:-4]))
# print(namelist[0:10])
# print(namelist[-10:])
# exit(0)

for filename in tqdm(namelist):
    imgpath = os.path.join(path, filename)
    #print("*********" + imgpath)
    img = cv2.imread(imgpath)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in tqdm(range(len(img_array))):
    out.write(img_array[i])
out.release()