# 基于hue hist距离的场景分割

import cv2
import argparse
import json
import os
import numpy as np
from tqdm import tqdm


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
    for frame_number, frame in tqdm(enumerate(img_array), desc='calculate frame stats'):

        # Convert to grayscale, scale down and blur to make
        # calculate image differences more robust to noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)      # 提取灰度信息
        gray = scale(gray, 0.25, 0.25)      # 缩放为原来的四分之一
        gray = cv2.GaussianBlur(gray, (9,9), 0.0)   # 做高斯模糊

        if frame_number < after_frame:
            lastFrame = gray
            continue


        if lastFrame is not None:
            n_bin = 60
            curr_hist = cv2.calcHist([gray],[0],None,[n_bin],[0,180])
            last_hist = cv2.calcHist([lastFrame],[0],None,[n_bin],[0,180])
            diffMag = cv2.compareHist(curr_hist, last_hist, cv2.HISTCMP_BHATTACHARYYA)

            # diff = cv2.subtract(gray, lastFrame)        # 用当前帧减去上一帧
            # diffMag = cv2.countNonZero(diff)        # 计算两帧灰度值不同的像素点个数

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

    return data


def detece_scene(img_array):
    data = calculateFrameStats(img_array)

    # diff_threshold = (data["stats"]["sd"] * 1.85) + data["stats"]["mean"]
    diff_threshold = (data["stats"]["sd"] * 6) + (data["stats"]["mean"])

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
    
    namelist = namelist[:4000]
    
    for filename in tqdm(namelist, desc='load video img'):
        imgpath = os.path.join(path, filename)
        img = cv2.imread(imgpath)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    # scene detection
    scene_points = detece_scene(img_array)
    scene_points.append(len(img_array))
    
    # export video
    scene_idx = 0
    out = cv2.VideoWriter('result\\project_scene_' + str(scene_idx) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in tqdm(range(len(img_array)), desc='export video'):
        out.write(img_array[i])
        if i == scene_points[scene_idx]:
            out.release()
            scene_idx += 1
            out = cv2.VideoWriter('result\\project_scene_' + str(scene_idx) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)