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

JSON_PATH = "json/output2.json"
f = open(JSON_PATH)
pts = json.load(f)

key = list(pts.keys())

key1 = key[0]
pts1 = pts[key1]

lane1 = pts1[0]
lane2 = pts1[1]

lines = vanishing(lane1, lane2)
img = plt.imread("../test_set/" + key1)
lines = lines / lines[2]
lines = lines[:-1].astype(int)
print(lines)

img = cv2.circle(img, (lines[0], lines[1]), 5, (0, 255, 0), -1)
plt.imshow(img)
plt.show()