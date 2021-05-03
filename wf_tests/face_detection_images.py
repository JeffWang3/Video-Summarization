import cv2
import os
from tqdm import tqdm


def detect_face(img_array, face_cascade):
    for img in img_array:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break



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
    