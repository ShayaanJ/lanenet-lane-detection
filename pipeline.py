import os
import json
from collections import OrderedDict


from evaluate_lanenet_on_tusimple import eval_lanenet
from modules.task5 import videoify
from ..yolov7.detect import detect

DATA_PATH = "../test_set"
JSON_PATH = "outputs/json/"
YOLO_PATH = "outputs/yolo/"
SAVE_DIR = "outputs/seg_results/"
VID_DIR = "outputs/vids/"
CONST_IMAGE = (1312, 1312)

    
def main(image_dir, weights="weights/tusimple_lanenet.ckpt"):
    clip_num = "_".join(image_dir.split("/")[3:])
    save_json = os.path.join(JSON_PATH, clip_num + ".json")
    vid_path = os.path.join(VID_DIR, clip_num + ".mp4")

    # --------------------------------
    # Detect lanes using lanenet 
    # Task 2
    #--------------------------------
    print("Detecting lanes")
    lanes_json = eval_lanenet(image_dir, weights, SAVE_DIR, save_json)
    print("Lanes detected")
    lanes_json = OrderedDict(sorted(lanes_json.items(), key=lambda x: int(x[0].split("/")[-1].split(".")[0])))
    
    # run yolo here

    # --------------------------------
    # Writing video
    # Task 4 and 5
    #--------------------------------
    print("Writing Video")
    videoify(lanes_json, vid_path)
    print(f"Video written in {vid_path}")
    
    return

main("../test_set/clips/0531/1492626499813320696")