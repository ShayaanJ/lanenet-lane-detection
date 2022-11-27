import json
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2


def line_eq(pts):
    pts = np.append(pts, np.ones((pts.shape[0], 1), dtype=np.int64), axis=1)
    lane = np.cross(pts[0], pts[1])
    return lane


def vanishing(pts1, pts2):
    pts_lane1 = np.array(random.sample(pts1, k=2))
    pts_lane2 = np.array(random.sample(pts2, k=2))

    lane1 = line_eq(pts_lane1)
    lane2 = line_eq(pts_lane2)

    vanishing_pt = np.cross(lane1, lane2)
    return vanishing_pt


def dst_pts(src):
    frac = 0.25
    # horizontal difference between src2 and src1
    length = src[1][0] - src[0][0]
    dst1 = [src[0][0] + length*frac, src[0][1]]
    dst2 = [src[1][0] - length*frac, src[0][1]]
    dst3 = [src[0][0] + length*frac, src[2][1]]
    dst4 = [src[1][0] - length*frac, src[2][1]]
    return [dst1, dst2, dst3, dst4]


def src_pts(pts1, pts2):
    src1 = pts1[int(len(pts1)*0.5)]
    src3 = pts1[int(len(pts1)*0.75)]
    x_30 = pts2[int(len(pts2)*0.3)]
    x_80 = pts2[int(len(pts2)*0.8)]
    eq = line_eq([x_30, x_80])
    # ax + by + c = 0
    # x = -(by + c)/a
    src2 = [-(eq[1]*src1[1] + eq[2])/eq[0], src1[1]]
    src4 = [-(eq[1]*src3[1] + eq[2])/eq[0], src3[1]]
    return [src1, src2, src3, src4]


JSON_PATH = "json/output2.json"
f = open(JSON_PATH)
pts = json.load(f)

key = list(pts.keys())

key1 = key[0]
pts1 = pts[key1]

lane1 = pts1[0]
lane2 = pts1[1]

# lines = vanishing(lane1, lane2)
img = plt.imread("../test_set/" + key1)
# lines = lines / lines[2]
# lines = lines[:-1].astype(int)
# print(lines)

# img = cv2.circle(img, (lines[0], lines[1]), 5, (0, 255, 0), -1)
src = src_pts(lane1, lane2)
dst = dst_pts(src)
img = cv2.circle(img, src[0], 5, (0, 255, 0), -1)
img = cv2.circle(img, src[1], 5, (0, 255, 0), -1)
img = cv2.circle(img, src[2], 5, (0, 255, 0), -1)
img = cv2.circle(img, src[3], 5, (0, 255, 0), -1)
img = cv2.circle(img, dst[0], 5, (255, 0, 0), -1)
img = cv2.circle(img, dst[1], 5, (255, 0, 0), -1)
img = cv2.circle(img, dst[2], 5, (255, 0, 0), -1)
img = cv2.circle(img, dst[3], 5, (255, 0, 0), -1)
plt.imshow(img)
plt.show()

# find 50th percentile point, find line between 30th and 80th percentile
# find point corresponding to exact y
# find 75th percentile point, find corresponding point to exact y
# find dest points using +- difference * fraction
# combine coords to find remaining 2 points
