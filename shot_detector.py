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
    
 
def calculateFrameStats(img_array, num_feature_points=50, verbose=False, after_frame=0):  # 提取相邻帧的差别
    data = {
        "frame_info": []
    }

    lastFrame = None
    for frame_number, frame in tqdm(enumerate(img_array), desc='shot detection'):
        if frame_number < after_frame:
            lastFrame = frame
            continue

        if lastFrame is not None:
            orb = cv2.ORB_create(nfeatures=num_feature_points)
            kp1, des1 = orb.detectAndCompute(frame, None)
            kp2, des2 = orb.detectAndCompute(lastFrame, None)
            
            try:
                # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                # TODO: 考虑matches间的距离
                diffMag = 1.0 - len(matches) / num_feature_points
            except:
                diffMag = 1.0
            
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
        lastFrame = frame

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


def detect_shot(data, num_shots):
    
    
    scores = [fi["diff_count"] for fi in data["frame_info"]]
    scores = sorted(scores, reverse=True)
    diff_threshold = scores[num_shots]
    
    # diff_threshold = (data["stats"]["sd"] * 1.85) + data["stats"]["mean"]
    # diff_threshold = (data["stats"]["sd"] * 3) + (data["stats"]["mean"])

    scene_points = []
    for index, fi in enumerate(data["frame_info"]):
        if fi["diff_count"] >= diff_threshold:
            scene_points.append(index)

    return scene_points
    

if __name__ == '__main__':
    # parameters
    path = '.\\project_dataset\\frames\\soccer'
    num_shots = 60  # 目标shot总数
    num_feature_points = 50  # 特征点数量
    min_frame_per_shot = 8  # frame数少于该值的shot会被merge
    
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
    
    # (skip) scene detection
    # scene_points = detece_scene(img_array)
    # print(scene_points)
    scene_points = []
    scene_points = [0] + scene_points + [len(img_array)]
    
    # shot detection
    shot_points = []
    scene_idx = 0
    for i in range(len(scene_points) - 1):
        data = calculateFrameStats(img_array[scene_points[i]: scene_points[i+1]], num_feature_points)
        shots = detect_shot(data, num_shots)
        shots = [x + scene_points[i] for x in shots]
        shot_points.extend(shots)
        shot_points.append(scene_points[i+1])
    shot_points = np.unique(np.array(shot_points))
    
    # merge short shots greedily
    reduced_points = []
    for p in range(len(shot_points) - 1):
        if shot_points[p+1] - shot_points[p] < min_frame_per_shot:
            continue
        else:
            reduced_points.append(shot_points[p])
    shot_points = reduced_points
    shot_points.append(len(img_array))
    print(shot_points)
    
    # export video
    shot_idx = 0
    out = cv2.VideoWriter('result\\shot_' + str(shot_idx) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in tqdm(range(len(img_array)), desc='export video'):
        out.write(img_array[i])
        if i == shot_points[shot_idx]:
            out.release()
            shot_idx += 1
            out = cv2.VideoWriter('result\\shot_' + str(shot_idx) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)