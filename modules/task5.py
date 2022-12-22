import cv2
import os
import numpy as np

from modules.task4 import src_pts, dst_pts

def videoify(final_frames, vid_path, const_image=(1312, 1312)):
    vid = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 4, const_image)
    for i in final_frames:
        vid.write(i)
    vid.release()
    
    return "Video Written"
