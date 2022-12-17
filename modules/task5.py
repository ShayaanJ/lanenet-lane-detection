import cv2
import os
import numpy as np

from modules.task4 import src_pts, dst_pts

def videoify(pts, vid_path, data_path="../test_set", const_image=(1312, 1312)):
    final_frames = []
    for frame in pts.keys():
        img = cv2.imread(os.path.join(data_path, frame))
        lanes = pts[frame]
        
        img_h, img_w, _ = img.shape
        diff = list(map(lambda x: abs(x[-1][0] - img_w / 2), lanes))
        diff = np.argsort(diff)
        diff = diff[:2]
        diff.sort()
        lane1 = lanes[diff[0]]
        lane2 = lanes[diff[1]]

        src = np.array(src_pts(lane1, lane2, img_h), dtype=np.float32)
        dst = np.array(dst_pts(src), dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        zeros = np.identity(3)
        zeros[:2, 2]  += [0, 750]
        H = zeros @ H
        warp = cv2.warpPerspective(img, H, const_image)
        final_frames.append(warp)
    
    vid = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 4, const_image)
    for i in final_frames:
        vid.write(i)
    vid.release()
    
    return "Video Written"
