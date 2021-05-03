import cv2
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import math


def face_scorer(img_array, face_cascade):
    """
    detect faces in each frame
    """
    data = []

    for frame_number, frame in tqdm(enumerate(img_array), desc='face detection'):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        frame_info = {
            'fid': frame_number,
            'face_num': len(faces),
            'face_loc' : faces
        }
        
        data.append(frame_info)

    return data
    

if __name__ == '__main__':
    # parameters
    path = '..\\project_dataset\\frames\\soccer'
    
    # import video
    img_array = []
    namelist = os.listdir(path)
    namelist = sorted(namelist, key = lambda x: int(x[5:-4]))
    for filename in tqdm(namelist, desc='load video img'):
        imgpath = os.path.join(path, filename)
        img = cv2.imread(imgpath)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    # face_detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load the cascade
    detect_face(img_array, face_cascade)