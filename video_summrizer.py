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
        self.shot_span = None  # frame indexes for shot starts 
        self.shot_score = None
        self.selected_shots = None
        
    def import_video(self, img_array, wav_array=None):
        self.img_array = img_array
        self.wav_array = wav_array
        self.frame_info = [{'fid':fid} for fid in range(len(self.img_array))]
        
    def import_video_from_path(self, img_folder, wav_path):
        self.img_array = []
        namelist = os.listdir(img_folder)
        namelist = sorted(namelist, key = lambda x: int(x[5:-4]))
        for filename in tqdm(namelist, desc='load video img'):
            imgpath = os.path.join(img_folder, filename)
            img = cv2.imread(imgpath)
            height, width, layers = img.shape
            size = (width,height)
            self.img_array.append(img)
            
        self.wav_array = None
        
        self.frame_info = [{'fid':fid} for fid in range(len(self.img_array))]
        
    def motion_detection(self, num_feature_points=200):
        orb = cv2.ORB_create(nfeatures=num_feature_points)
        lastFrame = None
        for fid, frame in tqdm(enumerate(self.img_array), desc='motion detection'):
            if fid == 0:
                last_kp, last_des = orb.detectAndCompute(frame, None)
                self.frame_info[fid]['motion_feature_loc'] = [kp.pt for kp in last_kp]
                self.frame_info[fid]['motion_matched_num'] = num_feature_points
                self.frame_info[fid]['motion_mean_dist'] = 0
                continue

            curr_kp, curr_des = orb.detectAndCompute(frame, None)
            
            try:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(last_des, curr_des)
                matches_dist = np.array([x.distance for x in matches])
                
                self.frame_info[fid]['motion_feature_loc'] = [kp.pt for kp in curr_kp]
                
                matched_num = np.sum(matches_dist < 60)
                if len(curr_kp) < num_feature_points * 0.3 and len(last_kp) < num_feature_points * 0.3:
                    self.frame_info[fid]['motion_matched_num'] = 1
                else:
                    self.frame_info[fid]['motion_matched_num'] = matched_num / (len(curr_kp) + len(last_kp) - matched_num)  # Jaccard Distance
                    
                # self.frame_info[fid]['motion_matched_dist'] = matches_dist
                # self.frame_info[fid]['motion_max_dist'] = np.max(matches_dist)
                # self.frame_info[fid]['motion_min_dist'] = np.min(matches_dist)
                self.frame_info[fid]['motion_mean_dist'] = np.mean(matches_dist) if len(matches_dist) > 0 else 60
            except:
                self.frame_info[fid]['motion_feature_loc'] = []
                if len(curr_kp) < num_feature_points * 0.3 and len(last_kp) < num_feature_points * 0.3:
                    self.frame_info[fid]['motion_matched_num'] = 1
                else:
                    self.frame_info[fid]['motion_matched_num'] = 0
                self.frame_info[fid]['motion_mean_dist'] = 60
                pass

            last_kp, last_des = curr_kp, curr_des
            
    def face_detection(self):
        for fid, raw_frame in tqdm(enumerate(self.img_array), desc='face detection'):
            if fid % 2 == 0:
                frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                
                faces = self.face_cascade.detectMultiScale(frame_gray, 1.1, 4)
            
                self.frame_info[fid]['face_num'] = len(faces)
                self.frame_info[fid]['face_loc'] = faces
            else:
                self.frame_info[fid]['face_num'] = 0
                self.frame_info[fid]['face_loc'] = []
    
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

    def shot_segmentation(self, num_shots=200, min_frame_per_shot=3):
        motion_matched_nums = [finfo['motion_matched_num'] for finfo in self.frame_info]
        scores = sorted(motion_matched_nums)
        diff_threshold = scores[num_shots]
        
        shot_index = []
        for fid, finfo in enumerate(self.frame_info):
            if finfo['motion_matched_num'] <= diff_threshold:
                shot_index.append(fid)
        # print('key frames', shot_index)
        # merge short shots greedily
        # todo: short last shot 
        reduced_shot_index = []
        for p in range(len(shot_index) - 1):
            if shot_index[p+1] - shot_index[p] < min_frame_per_shot:
                continue
            else:
                reduced_shot_index.append(shot_index[p])
        
        reduced_shot_index = [0] + reduced_shot_index + [len(self.img_array)]
        # print('shot spans', reduced_shot_index)
        self.shot_span = [(reduced_shot_index[i], reduced_shot_index[i+1]) for i in range(len(reduced_shot_index)-1)]
        
        return self.shot_span
    
    def shot_selection(self, max_frame_per_shot=500, total_frame=2700):
        self.shot_score = [0 for _ in range(len(self.shot_span))]
        for sid, span in tqdm(enumerate(self.shot_span), desc='select shots'):
            score = 0
            masked_frames = 0
            for fid in range(span[0], span[1]):
                # have_face; avg_motion_distance; color_distribution_difference; anti_mask
                if len(self.frame_info[fid]['motion_feature_loc']) < 50:
                    masked_frames += 1
                score += 4.0 * int(self.frame_info[fid]['face_num'] > 0) + 3.0 * self.frame_info[fid]['motion_mean_dist'] / 60 + 2.0 * self.frame_info[fid]['hist_diff'] # + 5 * len(self.frame_info[i]['motion_feature_loc']) / 200
            if masked_frames > min(15, (span[1] - span[0]) / 2.0 ):
                self.shot_score[sid] = 0
            else:
                self.shot_score[sid] = score / (span[1] - span[0])
            # print(sid, masked_frames)
        self.shot_score = np.array(self.shot_score)
        sorted_index = np.flip(np.argsort(self.shot_score))

        selected_index = []
        total_frames = 0
        for idx in sorted_index:
            shot_frames = self.shot_span[idx][1] - self.shot_span[idx][0]
            if shot_frames > max_frame_per_shot or shot_frames + total_frames > total_frame:
                continue
            selected_index.append(idx)
            total_frames += shot_frames
        selected_index = np.sort(np.array(selected_index))
        # print('selected shots', selected_index)
        self.selected_shots = [self.shot_span[i] for i in selected_index]
        return self.selected_shots
        
    def summrize(self):
        self.motion_detection()
        self.face_detection()
        self.color_detection()
        
        self.shot_segmentation()
        
        self.shot_selection()
        
        return self.selected_shots
    
    def export_shots(self, output_dir='result'):
        for i, span in tqdm(enumerate(self.shot_span), desc='export shots'):
            shot_path = os.path.join(output_dir, 'shot_' + str(i) + '.avi')
            out = cv2.VideoWriter(shot_path,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
            for j in range(span[0], span[1]):
                out.write(self.img_array[j])
            out.release()
    
    def export_annotated_video(self, output_dir='result'):
        path = os.path.join(output_dir, 'annotated.avi')
        out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        for i, (img, info) in tqdm(enumerate(zip(self.img_array, self.frame_info)), desc='export annotated video'):
            for (x, y, w, h) in info['face_loc']:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            for (x, y) in info['motion_feature_loc']:
                cv2.circle(img, (round(x),round(y)), radius=1, color=(0, 255, 0), thickness=-1)
            out.write(img)
        out.release()
    
    def export_summrization(self, output_dir='result'):
        path = os.path.join(output_dir, 'summrization.avi')
        out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        for i, span in tqdm(enumerate(self.selected_shots), desc='export summrization'):
            for j in range(span[0], span[1]):
                out.write(self.img_array[j])
        out.release()

if __name__ == '__main__':
    # preprocess video
    path = '.\\project_dataset\\frames\\test_video_2'
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
    video_summrizer.face_detection()
    video_summrizer.color_detection()
    # shot segmentation
    video_summrizer.shot_segmentation()
    # shot selection
    video_summrizer.shot_selection()
    # export video
    video_summrizer.export_summrization()
    video_summrizer.export_shots()
    video_summrizer.export_annotated_video()