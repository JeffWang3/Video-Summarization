import os
import cv2
import numpy as np
from tqdm import tqdm


class VideoSummrizer():
    def __init__(
        self, 
        face_cascade_path='haarcascade_frontalface_default.xml'
    ):
        
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        self.img_array = None
        self.wav_array = None
        self.frame_info = None
        self.shot_index = None  # frame indexes for shot starts 
        
    def import_video(self, img_array, wav_array=None):
        self.img_array = img_array
        self.wav_array = wav_array
        self.frame_info = [{'fid':fid} for fid in range(len(self.img_array))]
        
    def motion_detection(self, num_feature_points=50):
        orb = cv2.ORB_create(nfeatures=num_feature_points)
        lastFrame = None
        for fid, frame in tqdm(enumerate(self.img_array), desc='motion detection'):
            if fid == 0:
                last_kp, last_des = orb.detectAndCompute(frame, None)
                self.frame_info[fid]['motion_feature_loc'] = last_kp
                self.frame_info[fid]['motion_matched_num'] = num_feature_points
                continue

            curr_kp, curr_des = orb.detectAndCompute(frame, None)
            
            try:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(last_des, curr_des)
                matches_dist = np.array([x.distance for x in matches])
                
                self.frame_info[fid]['motion_feature_loc'] = curr_kp
                self.frame_info[fid]['motion_matched_num'] = len(matches)
                # self.frame_info[fid]['motion_matched_dist'] = matches_dist
                self.frame_info[fid]['motion_max_dist'] = np.max(matches_dist)
                self.frame_info[fid]['motion_min_dist'] = np.min(matches_dist)
                self.frame_info[fid]['motion_mean_dist'] = np.mean(matches_dist)
            except:
                self.frame_info[fid]['motion_matched_num'] = 0
                pass

            last_kp, last_des = curr_kp, curr_des
            
    def face_detection(self):
        for fid, raw_frame in tqdm(enumerate(self.img_array), desc='face detection'):
            frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                
            faces = self.face_cascade.detectMultiScale(frame_gray, 1.1, 4)
            
            self.frame_info[fid]['face_num'] = len(faces)
            self.frame_info[fid]['face_loc'] = faces
    
    def color_detection(self, hist_bin_num=60):
        for fid, frame in tqdm(enumerate(self.img_array), desc='color detection'):
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)      

            if fid == 0:
                lastFrame = curr_frame
                self.frame_info[fid]['hist_diff'] = 0
                continue

            curr_hist = cv2.calcHist([curr_frame],[0],None,[hist_bin_num],[0,180])
            last_hist = cv2.calcHist([lastFrame],[0],None,[hist_bin_num],[0,180])
            hist_diff = cv2.compareHist(curr_hist, last_hist, cv2.HISTCMP_BHATTACHARYYA)
            
            self.frame_info[fid]['hist_diff'] = hist_diff
            
            last_frame = curr_frame
            
    def wave_detection(self):
        pass

    def shot_segmentation(self, num_shots=60, min_frame_per_shot=5):
        motion_matched_nums = [finfo['motion_matched_num'] for finfo in self.frame_info]
        scores = sorted(motion_matched_nums)
        diff_threshold = scores[num_shots]
        
        shot_index = []
        for fid, finfo in enumerate(self.frame_info):
            if finfo['motion_matched_num'] <= diff_threshold:
                shot_index.append(fid)

        # merge short shots greedily
        # todo: short last shot 
        reduced_shot_index = []
        for p in range(len(shot_index) - 1):
            if shot_index[p+1] - shot_index[p] < min_frame_per_shot:
                continue
            else:
                reduced_shot_index.append(shot_index[p])
                
        self.shot_index = reduced_shot_index
        
        return self.shot_index
    
    def shot_selection(self):
        pass
    
    def export_shots(self, output_dir='result'):
        shot_idx = 0
        shot_path = os.path.join(output_dir, 'shot_' + str(shot_idx) + '.avi')
        out = cv2.VideoWriter(shot_path,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        for i in tqdm(range(len(img_array)), desc='export video'):
            if shot_idx < len(self.shot_index) and i == self.shot_index[shot_idx]:
                out.release()
                shot_idx += 1
                shot_path = os.path.join(output_dir, 'shot_' + str(shot_idx) + '.avi')
                out = cv2.VideoWriter(shot_path,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
            out.write(img_array[i])
        out.release()
    
    def export_annonated_shots(self, output_dir='result'):
        pass

if __name__ == '__main__':
    # preprocess video
    path = '.\\project_dataset\\frames\\soccer'
    img_array = []
    namelist = os.listdir(path)
    namelist = sorted(namelist, key = lambda x: int(x[5:-4]))
    for filename in tqdm(namelist, desc='load video img'):
        imgpath = os.path.join(path, filename)
        img = cv2.imread(imgpath)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
    # init summrizer
    video_summrizer = VideoSummrizer()
    video_summrizer.import_video(img_array=img_array)
    # frame stats
    video_summrizer.motion_detection()
    # video_summrizer.face_detection()
    # video_summrizer.color_detection()
    # shot segmentation
    video_summrizer.shot_segmentation()
    # shot selection
    # export video
    video_summrizer.export_shots()