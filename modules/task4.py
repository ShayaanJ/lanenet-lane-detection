import cv2
import os
import numpy as np

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

def homoify(pts, data_path="../test_set", const_image=(1312, 1312)):
    final_frames = []
    for frame in pts.keys():
        img_path = data_path +"/"+ frame
        img = cv2.imread(img_path)
        lanes = pts[frame]
        img_h, img_w, _ = img.shape
        
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
        final_frames.append(warp)
    return final_frames
