import mainWin

import cv2
from PyQt5 import QtGui,QtWidgets,QtCore, QtMultimedia
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

import sys
import numpy as np
from tqdm import tqdm
import os
from video_summrizer import VideoSummrizer

def cvImgtoQtImg(cvImg): #opencv image to qt image
    QtImgBuf = cv2.cvtColor(cvImg,  cv2.COLOR_BGR2BGRA)
    QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGB32)
    return QtImg



class mainwin(QtWidgets.QMainWindow, mainWin.Ui_MainWindow):
    def __init__(self, argv):
        super().__init__()
        self.setupUi(self)
        self.bClose = False
        self.fps = 30
        self.waitKeyTime = int(1000/self.fps)
        self.sleep = False
        self.currentTupleIndex = 0
        self.index = 0
        self.images = []
        self.image_indices = []
        self.audioPositions = []
        self.folderpath = argv[1]
        self.audiopath = argv[2]
        self.audioPlayer = QMediaPlayer()
        self.audioPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.audiopath)))
        self.frameTuples = []

        self.openbtn.clicked.connect(self.videoSummarize)
        self.playbtn.clicked.connect(self.playOrPause)
        self.slider.sliderMoved.connect(self.changeSliderValue)
        #self.actionshowImg.triggered.connect(self.videoSummarize) # connect showImg button with playVideoFile method
        #self.actionPlayOrPause.triggered.connect(self.playOrPause) # connect actionPlayOrPause button with playOrPause method
        
        self.video_summrizer = VideoSummrizer()

    def videoSummarize(self):
        # process by Fei
        # self.frameTuples = [(0, 16200-1)]
        # self.frameTuples = [(0, 300), (16000, 16200-1)]
        self.video_summrizer.import_video_from_path(self.folderpath, self.audiopath)
        self.frameTuples = self.video_summrizer.summrize()
        self.renderVideo()
        return


    def renderVideo(self):
        '''
        render video of frame-id tuples-list [(s0, e0), (s1, e1), (s2, e2), ...] (ei inclusive);
        ensure that tuples are ordered, i.e. we have ei-1 <= si;
        '''
        frameTuples = self.frameTuples
        if len(self.images) == 0: # if empty, load the images; else we do not need to load it again
            self.images, self.image_indices = self.getFrameArray(self.folderpath, frameTuples)
            self.audioPositions = self.getAudioPositionTuples(frameTuples)


        self.audioPlayer.play()
        print("start reading image series from index " + str(self.index))
        for index in tqdm(range(self.index, len(self.images))):
            self.setSliderValue(index)
            if index == 0 or (self.image_indices[index] - self.image_indices[index-1] != 1) :
                self.audioPlayer.setPosition(int(self.image_indices[index] * 1000 / 30))
            if self.sleep:
                break
            QtImg = cvImgtoQtImg(self.images[index])  # convert image to qt style
            self.ImgDisp.setPixmap(QtGui.QPixmap.fromImage(QtImg))
            size = QtImg.size() 
            self.ImgDisp.resize(size)
            self.ImgDisp.show() # refresh the image
            cv2.waitKey(self.waitKeyTime) # sleep according to fps
        
        self.index = index
        # print("\n *** pause or exit from index " + str(self.index))
        self.audioPlayer.pause()

        return


    def getAudioPositionTuples(self, frameTuples):
        '''
        compute audio position tuples according to the frame tuples and fps = 30;
        return position tuples in milisecond rate  [(s0, e0), (s1, e1), (s2, e2), ...] (ei inclusive);
        '''
        audioPositions = []
        for s, e in frameTuples:
            audioPositions.append((int(s*1000/30), int(e*1000/30)))

        return audioPositions

    def getFrameArray(self, folderpath, frameTuples): 
        '''
        render video of frame-id tuples-list [(s1, e1), (s2, e2), ...]  (ei inclusive);
        ensure that tuples are ordered, i.e. we have ei-1 <= si;
        return frames in the same order
        '''
        if folderpath == '':
            print('ERROR: empty path for the frames')
            exit(0)

        path = folderpath
        img_array = []
        img_index_array = []

        print("start loading images from local file: " + folderpath)
        for s, e in tqdm(frameTuples):
            for i in tqdm(range(s, e+1)): # append image from si to ei(inclusive)
                imgpath = os.path.join(path, 'frame' + str(i) + '.jpg')
                img = cv2.imread(imgpath)
                img_array.append(img)      
                img_index_array.append(i)      
        print("finish loading images from local file, frame number is " + str(len(img_array)))
        return img_array, img_index_array
    
    def playOrPause(self):
        '''
        if the video is playing, pause it;
        else awake and resume
        '''
        if self.sleep == False:
            self.sleep = True
        else:
            self.sleep = False
            self.renderVideo()

    def changeSliderValue(self, value):
        if self.sleep == False:
            self.sleep = True
            self.iconchange()
        self.index = int(value / 100 * len(self.images))
        QtImg = cvImgtoQtImg(self.images[self.index])  # convert image to qt style
        self.ImgDisp.setPixmap(QtGui.QPixmap.fromImage(QtImg))
        size = QtImg.size() 
        self.ImgDisp.resize(size)
        self.ImgDisp.show() # refresh the image

    def setSliderValue(self, index):
        self.slider.setValue(int(index / len(self.images) * 100))



if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = mainwin(sys.argv)
    w.show()
    sys.exit(app.exec_())