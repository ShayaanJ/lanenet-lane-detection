import cv2
import numpy as np
import os

def read_imgs(frames_dir):
    frames = [frames_dir + "/" + x for x in os.listdir(frames_dir)]
    frames = sorted(frames, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    frames = [cv2.imread(x) for x in frames]
    frames = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frames]
    return frames

def TextingOnImg(const_image_size, num_lanes, num_objs, num_left, num_right, num_changing):
    black_img = np.zeros(const_image_size, np.uint8)
    text1 = "Textual summary of current scene: "
    text2 = f"Total number of lanes: {num_lanes}"
    text3 = f"Total number of objects: {num_objs}"
    text4 = f"Total number of left lanes: {num_left}"
    text5 = f"Total number of right lanes: {num_right}"
    text6 = f"Total number of changing lanes: {num_changing}"
    

    black_img = cv2.putText(black_img, text1, (20, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    black_img = cv2.putText(black_img, text2, (20, 100 + 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    black_img = cv2.putText(black_img, text3, (20, 100 + 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    black_img = cv2.putText(black_img, text4, (20, 100 + 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    black_img = cv2.putText(black_img, text5, (20, 100 + 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    black_img = cv2.putText(black_img, text6, (20, 100 + 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    return black_img

def cat_images(top_view, blackImg, instanceSeg, source, max_w, max_h):
    canvas = np.zeros((max_h, max_w, 3), np.uint8)
    canvas[:source.shape[0], :source.shape[1]] = source
    canvas[:instanceSeg.shape[0], source.shape[1]:source.shape[1] + instanceSeg.shape[1]] = instanceSeg
    canvas[source.shape[0]:source.shape[0] + top_view.shape[0], :top_view.shape[1]] = top_view
    canvas[source.shape[0]:source.shape[0] + blackImg.shape[0], top_view.shape[1]:top_view.shape[1] + blackImg.shape[1]] = blackImg

    return canvas

def videoify(clip_num, final_frames, vid_path, save_dir, const_image=(1312, 1312)):
    laneNetResultsPath = os.path.join(save_dir,clip_num)
    instanceSegResultsPath = f"{laneNetResultsPath}/instance_seg"
    sourceResultsPath = f"{laneNetResultsPath}/result"
    instanceSeg = read_imgs(instanceSegResultsPath)
    source = read_imgs(sourceResultsPath)
    instanceSeg = [cv2.resize(x, (1280, 720)) for x in instanceSeg]
    instanceSeg = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in instanceSeg]

    max_w = max(const_image[1], instanceSeg[0].shape[1], source[0].shape[1]) * 2
    max_h = const_image[0] + source[0].shape[0]

    vid = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 4, (max_w, max_h))
    for idx in range(len(final_frames)):
        top_view = final_frames[idx]['img']
        num_lanes = final_frames[idx]['num_lanes']
        num_objs = final_frames[idx]['num_objs']
        num_left = final_frames[idx]['left']
        num_right = final_frames[idx]['right']
        num_changing = final_frames[idx]['changing']
        seg_result_img = final_frames[idx]['seg']
        blackImg = TextingOnImg((const_image[0], const_image[1], 3), num_lanes, num_objs, num_left, num_right, num_changing)
        canvas = cat_images(top_view, blackImg, instanceSeg[idx], seg_result_img, max_w, max_h)

        vid.write(canvas)
    vid.release()
    
    return "Video Written"