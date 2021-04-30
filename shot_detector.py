# 基于特征点追踪的shot分割

import cv2
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import math

from scene_detector import detece_scene


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


def scale(img, xScale, yScale):
    res = cv2.resize(img, None,fx=xScale, fy=yScale, interpolation = cv2.INTER_AREA)
    return res


def resize(img, width, heigth):
    res = cv2.resize(img, (width, heigth), interpolation = cv2.INTER_AREA)
    return res
    
 
def calculateFrameStats(img_array, verbose=False, after_frame=0):  # 提取相邻帧的差别
    data = {
        "frame_info": []
    }

    lastFrame = None
    for frame_number, frame in tqdm(enumerate(img_array)):

        # Convert to grayscale, scale down and blur to make
        # calculate image differences more robust to noise
        
        gray = frame
        
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # 提取灰度信息
        # gray = scale(gray, 0.25, 0.25)      # 缩放为原来的四分之一
        # gray = cv2.GaussianBlur(gray, (9,9), 0.0)   # 做高斯模糊

        if frame_number < after_frame:
            lastFrame = gray
            continue
        
        

        if lastFrame is not None:
            
            n_points = 50
            
            orb = cv2.ORB_create(nfeatures=n_points)
            kp1, des1 = orb.detectAndCompute(gray, None)
            kp2, des2 = orb.detectAndCompute(lastFrame, None)
            
            try:
                # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                n_outlier = 0
            
                diffMag = 1 - (len(matches) - n_outlier) / n_points
            except:
                diffMag = 0
                
            # if des1[0][0] != des2[0][0]:
                # print([x.distance for x in matches])
                # exit()
            
            # n_outlier = np.sum(np.array([x.distance > 50 for x in matches]))
           
        
        
        
        
            # p0 = cv2.goodFeaturesToTrack(lastFrame, mask=None, **feature_params)
            
            # if p0 is not None:
                # p1, st, err = cv2.calcOpticalFlowPyrLK(gray, lastFrame, p0, None, **lk_params)
                
                # diffMag = 1 - st.mean()
            # else:
                # diffMag = 0
                
            # q1 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            # if q1 is not None:
                # q0, stq, errq = cv2.calcOpticalFlowPyrLK(lastFrame, gray, q1, None, **lk_params)
                
            
                # diffMagq = 1 - stq.mean()
            # else:
                # diffMagq = 0
            # diffMag = max(diffMag, diffMagq)
            
            
            
            
                
            # if st.sum() == 0:
            
                # print(p0)
                # print(p1)
                # print(st)
                # print(err)
                # exit()
            
            # # 根据状态选择
            # good_new = p1  # [st == 1]
            # good_old = p0  # [st == 1]

            # # 绘制跟踪线
            # diff = 0
            # for i, (new, old) in enumerate(zip(good_new,good_old)):
                # a,b = new.ravel()
                # c,d = old.ravel()
                # diff += math.sqrt(math.pow(a-b, 2) + math.pow(c-d, 2))
        
            # diffMag = diff / (len(good_new) + 1e-6)
                
            
            
            
            
            frame_info = {
                "frame_number": int(frame_number),
                "diff_count": float(diffMag)
            }
            data["frame_info"].append(frame_info)

            if verbose:
                cv2.imshow('diff', diff)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Keep a ref to this frame for differencing on the next iteration
        lastFrame = gray

    #compute some states
    diff_counts = [fi["diff_count"] for fi in data["frame_info"]]
    data["stats"] = {
        "num": len(diff_counts),
        "min": np.min(diff_counts),
        "max": np.max(diff_counts),
        "mean": np.mean(diff_counts),
        "median": np.median(diff_counts),
        "sd": np.std(diff_counts)   # 计算所有帧之间, 像素变化个数的标准差
    }
    greater_than_mean = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["mean"]]
    greater_than_median = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["median"]]
    greater_than_one_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["sd"] + data["stats"]["mean"]]
    greater_than_two_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > (data["stats"]["sd"] * 2) + data["stats"]["mean"]]
    greater_than_three_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > (data["stats"]["sd"] * 3) + data["stats"]["mean"]]

    # 统计其他信息
    data["stats"]["greater_than_mean"] = len(greater_than_mean)
    data["stats"]["greater_than_median"] = len(greater_than_median)
    data["stats"]["greater_than_one_sd"] = len(greater_than_one_sd)
    data["stats"]["greater_than_three_sd"] = len(greater_than_three_sd)
    data["stats"]["greater_than_two_sd"] = len(greater_than_two_sd)
        
    print(data["stats"])
    
    return data


def detect_shot(img_array):
    data = calculateFrameStats(img_array)

    # diff_threshold = (data["stats"]["sd"] * 1.85) + data["stats"]["mean"]
    diff_threshold = (data["stats"]["sd"] * 3) + (data["stats"]["mean"])

    scene_points = []
    for index, fi in enumerate(data["frame_info"]):
        if fi["diff_count"] >= diff_threshold:
            scene_points.append(index)

    return scene_points
    

if __name__ == '__main__':
    # import video
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
    
    # scene detection
    # scene_points = detece_scene(img_array)
    # print(scene_points)
    scene_points = []
    scene_points = [0] + scene_points + [len(img_array)]
    
    # shot detection
    shot_points = []
    scene_idx = 0
    for i in range(len(scene_points) - 1):
        shots = detect_shot(img_array[scene_points[i]: scene_points[i+1]])
        shots = [x + scene_points[i] for x in shots]
        shot_points.extend(shots)
        shot_points.append(scene_points[i+1])
    shot_points = np.unique(np.array(shot_points))
    
    # merge short shots greedily
    reduced_points = []
    for p in range(len(shot_points) - 1):
        if shot_points[p+1] - shot_points[p] < 8:
            continue
        else:
            reduced_points.append(shot_points[p])
    shot_points = reduced_points
    shot_points.append(len(img_array))
    print(shot_points)
    
    # export video
    shot_idx = 0
    out = cv2.VideoWriter('result\\project_shot_' + str(shot_idx) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in tqdm(range(len(img_array)), desc='export video'):
        out.write(img_array[i])
        if i == shot_points[shot_idx]:
            out.release()
            shot_idx += 1
            out = cv2.VideoWriter('result\\project_shot_' + str(shot_idx) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)