import cv2
from PyQt5 import QtGui,QtWidgets,QtCore
import mainWin

import sys
import numpy as np
from tqdm import tqdm
import os

def cvImgtoQtImg(cvImg): #opencv image to qt image
    QtImgBuf = cv2.cvtColor(cvImg,  cv2.COLOR_BGR2BGRA)

    QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGB32)
    
    return QtImg



class mainwin(QtWidgets.QMainWindow,mainWin.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.bClose = False
        self.fps = 30
        self.waitKeyTime = int(1000/self.fps)
        self.sleep = False
        self.index = 0
        self.images = []

        self.actionshowImg.triggered.connect(self.playVideoFile) # connect showImg button with playVideoFile method
        self.actionPlayOrPause.triggered.connect(self.playOrPause) # connect actionPlayOrPause button with playOrPause method

    def playVideoFile(self): # play the image series as a video
        if len(self.images) == 0:
            self.images = self.getFrameArray()
    
        print("start reading image series from index " + str(self.index))
        for index in tqdm(range(self.index, len(self.images))):
            if self.sleep:
                break
            QtImg = cvImgtoQtImg(self.images[index])  # convert image to qt style
            self.ImgDisp.setPixmap(QtGui.QPixmap.fromImage(QtImg))
            size = QtImg.size() 
            self.ImgDisp.resize(size)
            self.ImgDisp.show() # refresh the image
            cv2.waitKey(self.waitKeyTime) # sleep according to fps
        self.index = index
        print("pause or exit from index " + str(self.index))
        self.index = index


    def getFrameArray(self, folderpath='C:\\Users\\16200\\Desktop\\project_dataset\\frames\\soccer'):
        path = folderpath
        img_array = []
        namelist = sorted(os.listdir(path), key = lambda x: int(x[5:-4]))
        print("start loading images from local file")
        for image in tqdm(namelist):
            imgpath = os.path.join(path, image)
            img = cv2.imread(imgpath)
            # height, width, layers = img.shape
            # size = (width,height)
            img_array.append(img)
        print("finish loading images from local file")
        return img_array 
    
    def playOrPause(self):
        # self.sleep = not self.sleep
        if self.sleep == False:
            self.sleep = True
        else:
            self.sleep = False
            self.playVideoFile()



if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = mainwin()
    w.show()
    sys.exit(app.exec_())