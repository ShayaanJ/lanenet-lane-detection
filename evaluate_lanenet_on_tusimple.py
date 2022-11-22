#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-5-16 下午6:26
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : evaluate_lanenet_on_tusimple.py
# @IDE: PyCharm
"""
Evaluate lanenet model on tusimple lane dataset
"""
import argparse
import glob
import os
import os.path as ops
import time
import json

import cv2
import numpy as np
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
import random as rand

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_eval')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='The source tusimple lane test data dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--save_dir', type=str, help='The test output save root dir')
    parser.add_argument('--save_json', type=str, help='The test output save root json')


    return parser.parse_args()

def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def eval_lanenet(src_dir, weights_path, save_dir, save_json):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)

    os.makedirs(save_dir, exist_ok=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        image_list = glob.glob('{:s}/**/*.jpg'.format(src_dir), recursive=True)
        clip_0530, clip_0531, clip_0601 = [], [], []
        for i in image_list:
            if i.split("clips")[1].split("/")[1] == "0530":
                clip_0530.append(i)
            elif i.split("clips")[1].split("/")[1] == "0531":
                clip_0531.append(i)
            elif i.split("clips")[1].split("/")[1] == "0601":
                clip_0601.append(i)
        
        image_list = []
        for i in range(5):
            image_list.append(clip_0530[rand.randint(0, len(clip_0530)-1)])
            image_list.append(clip_0531[rand.randint(0, len(clip_0531)-1)])
            image_list.append(clip_0601[rand.randint(0, len(clip_0601)-1)])
        
        avg_time_cost = []
        dict = {}

        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            avg_time_cost.append(time.time() - t_start)
            
            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
            )

            mask_image = postprocess_result['mask_image']

            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)

            # #save images
            ptsList = postprocess_result['ptsList']
            image_name = "/".join(image_path.split('/')[2:])
            dict[image_name] = ptsList
 
            ####### shwoing images heree ########
            # plt.figure('mask_image')
            # plt.imshow(mask_image[:, :, (2, 1, 0)])
            # plt.figure('src_image')
            # plt.imshow(image_vis[:, :, (2, 1, 0)])
            # plt.figure('instance_image')
            # plt.imshow(embedding_image[:, :, (2, 1, 0)])
            # plt.figure('binary_image')
            # plt.imshow(binary_seg_image[0] * 255, cmap='gray')
            # plt.show()

            ####### saving one output file here ########
            if index % 100 == 0:
                LOG.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                avg_time_cost.clear()

            input_image_dir = ops.split(image_path.split('clips')[1])[0][1:]
            input_image_name = ops.split(image_path)[1]
            output_image_dir = ops.join(save_dir, input_image_dir)
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = ops.join(output_image_dir, input_image_name)
            if ops.exists(output_image_path):
                continue
            cv2.imwrite(output_image_path, postprocess_result['source_image'])
            cv2.imwrite(output_image_path.replace(".jpg", "") + 'binary_seg_image.jpg', binary_seg_image[0] * 255)
            cv2.imwrite(output_image_path.replace(".jpg", "") +  'instance_seg_image.jpg', embedding_image)
            cv2.imwrite(output_image_path.replace(".jpg", "") + 'mask_image.jpg', mask_image)
            # print("Images written")
        
        with open(save_json, "w") as outfile:
            json.dump(dict, outfile)
        # print("Json written")

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    eval_lanenet(
        src_dir=args.image_dir,
        weights_path=args.weights_path,
        save_dir=args.save_dir,
        save_json=args.save_json
    )
