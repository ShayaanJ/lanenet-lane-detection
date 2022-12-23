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

def homoify(pts, data_path="../test_set", yolo_path="outputs/yolo", save_path="outputs/seg_results", const_image=(1312, 1312)):
    final_frames = []
    yolo_folder = sorted(os.listdir(yolo_path), key=lambda x: int(x.replace("exp", "0")))[-1]
    for frame in pts.keys():
        frame_num = frame.split("/")[-1].split(".")[0]
        temp = "/".join(frame.split("/")[1:-1])
        txt_path = f"{yolo_path}/{yolo_folder}/labels/{frame_num}.txt"
        if not os.path.exists(txt_path):
            print(f"txt file not found for {txt_path}")
            continue
        img_path = f"{data_path}/{frame}"
        if not os.path.exists(img_path):
            print(f"image not found for {img_path}")
            continue
        seg_result = f"{save_path}{temp}result/images/{frame_num}.jpeg"
        if not os.path.exists(seg_result):
            print(f"segmentation result not found for {seg_result}")
            continue
        

        print(seg_result)
        
        #reading image, lanes, and bottom mid point of bboxes
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lanes = pts[frame]
        if len(lanes) < 2:
            continue
        img_h, img_w, _ = img.shape
        bottom_mid_pt = bbox_mid_pts(txt_path)
        seg_result_img = cv2.imread(seg_result)
        seg_result_img = cv2.cvtColor(seg_result_img, cv2.COLOR_BGR2RGB)

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
        
        line1 = line_eq(np.array([src[0], src[2]]))
        line2 = line_eq(np.array([src[1], src[3]]))
        line3 = line_eq(np.array([dst[0], dst[2]]))
        line4 = line_eq(np.array([dst[1], dst[3]]))
        
        thresh = 0.01
        w = thresh*1280
        w2 = thresh*const_image[0]
        left = 0
        right = 0
        changing = 0
        for i in bottom_mid_pt:
            ctr = 0
            i = i[0]
            i = i.astype(np.int32)
            i2 = (H @ np.append(i, 1))
            i2 = i2 / i2[2]
            i2 = i2[:2]
            i2 = i2.astype(np.int32)
            lp1 = -(line1[2] + line1[1]*i[1]) / line1[0]
            lp2 = -(line2[2] + line2[1]*i[1]) / line2[0]
            lp3 = -(line3[2] + line3[1]*i2[1]) / line3[0]
            lp4 = -(line4[2] + line4[1]*i2[1]) / line4[0]
            left += 1 if i[0] <= lp1 else 0
            right += 1 if i[0] >= lp2 else 0
            # if (i[0] > lp1 - w and i[0] < lp1 + w):
            #     ctr += 1
            #     seg_result_img = cv2.circle(seg_result_img, i, 30, (255, 0, 0), -1)
            # elif (i[0] > lp2 - w and i[0] < lp2 + w):
            #     ctr += 1
            #     seg_result_img = cv2.circle(seg_result_img, i, 30, (0, 255, 0), -1)

            if (i2[0] > lp3 - w2 and i2[0] < lp3 + w2):
                ctr += 1
                warp = cv2.circle(warp, i2, 30, (255, 0, 0), -1)
                seg_result_img = cv2.circle(seg_result_img, i, 30, (255, 0, 0), -1)
            elif (i2[0] > lp4 - w2 and i2[0] < lp4 + w2):
                ctr += 1
                warp = cv2.circle(warp, i2, 30, (0, 255, 0), -1)
                seg_result_img = cv2.circle(seg_result_img, i, 30, (0, 255, 0), -1)
            changing += 0 if ctr == 0 else 1
            # print("changing") if ctr >= 1 else None
        final_frames.append({"img": warp, "num_lanes": len(
            lanes), "num_objs": len(bottom_mid_pt), "left": left, "right": right, "changing": changing, "seg": seg_result_img})

    return final_frames
