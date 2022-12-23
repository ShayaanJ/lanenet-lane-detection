import cv2
import os
import numpy as np

from modules.yolo_util import getAnnotations, removeNoncar, itsBboxTime, overlay

def line_eq(pts):
    pts = np.append(pts, np.ones((pts.shape[0], 1), dtype=np.int64), axis=1)
    lane = np.cross(pts[0], pts[1])
    return lane

def dst_pts(src):
    frac = 0.25
    # horizontal difference between src2 and src1
    length = src[1][0] - src[0][0]
    dst1 = [int(src[0][0] + length*frac), src[0][1]]
    dst2 = [int(src[1][0] - length*frac), src[0][1]]
    dst3 = [int(src[0][0] + length*frac), src[2][1]]
    dst4 = [int(src[1][0] - length*frac), src[2][1]]
    return [dst1, dst2, dst3, dst4]

def src_pts(pts1, pts2, w):
    pts1 = [np.mean(pts1[:3], axis=0), np.mean(pts1[3:6], axis=0)]
    pts2 = [np.mean(pts2[:3], axis=0), np.mean(pts2[3:6], axis=0)]
    lane1 = line_eq(np.array(pts1))
    lane2 = line_eq(np.array(pts2))
    y1 = w // 2
    y2 = w // 2 + (w // 2 // 2)

    # ax + by + c = 0
    # x = -(by + c)/a
    src1 = [int(-(lane1[1]*y1 + lane1[2])/lane1[0]), y1]
    src2 = [int(-(lane2[1]*y1 + lane2[2])/lane2[0]), y1]
    src3 = [int(-(lane1[1]*y2 + lane1[2])/lane1[0]), y2]
    src4 = [int(-(lane2[1]*y2 + lane2[2])/lane2[0]), y2]
    return [src1, src2, src3, src4]

def bbox_mid_pts(txt_path):
    bbox_txt = getAnnotations(txt_path)
    bbox_txt = removeNoncar(bbox_txt)
    bottom_mid_pt = [itsBboxTime(i) for i in bbox_txt]
    bottom_mid_pt = np.array([np.array([i[0], i[1]]) for i in bottom_mid_pt], dtype=np.float32)
    bottom_mid_pt = bottom_mid_pt.reshape(-1, 1, 2)
    return bottom_mid_pt

def homoify(pts, data_path="../test_set", yolo_path="outputs/yolo", const_image=(1312, 1312)):
    final_frames = []
    yolo_folder = sorted(os.listdir(yolo_path), key=lambda x: int(x.replace("exp", "0")))[-1]
    for frame in pts.keys():
        frame_num = frame.split("/")[-1].split(".")[0]
        txt_path = f"{yolo_path}/{yolo_folder}/labels/{frame_num}.txt"
        img_path = f"{data_path}/{frame}"
        
        #reading image, lanes, and bottom mid point of bboxes
        img = cv2.imread(img_path)
        lanes = pts[frame]
        img_h, img_w, _ = img.shape
        bottom_mid_pt = bbox_mid_pts(txt_path)

        # getting 2 lanes closest to center
        diff = list(map(lambda x: abs(x[-1][0] - img_w / 2), lanes))
        diff = np.argsort(diff)
        diff = diff[:2]
        diff.sort()
        lane1 = lanes[diff[0]]
        lane2 = lanes[diff[1]]
        src = np.array(src_pts(lane1, lane2, img_h), dtype=np.float32)
        dst = np.array(dst_pts(src), dtype=np.float32)

        # homography matrix
        H = cv2.getPerspectiveTransform(src, dst)
        zeros = np.identity(3)
        zeros[:2, 2]  += [0, 750]
        H = zeros @ H

        # warp image
        warp = cv2.warpPerspective(img, H, const_image)
        warped_pts = cv2.perspectiveTransform(bottom_mid_pt, H)
        warped_pts = np.array([i[0] for i in warped_pts], dtype=np.int32)
        warp = overlay(warp, warped_pts)
        
        final_frames.append({"img":warp, "num_lanes":len(lanes), "num_objs":len(bottom_mid_pt)})
    return final_frames
