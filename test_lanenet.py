#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='src dir')
    parser.add_argument('--save_path', type=str, help='result dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, save_image_path, with_lane_fit=True):
    """
    :param image_path:
    :param weights_path:
    :param with_lane_fit:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    # saver = tf.compat.train.Saver(variables_to_restore)
    saver = tf.compat.v1.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        loop_times = 500
        for i in range(loop_times):
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
        t_cost = time.time() - t_start
        t_cost /= loop_times
        LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=with_lane_fit,
            data_source='tusimple'
        )
        mask_image = postprocess_result['mask_image']
        if with_lane_fit:
            lane_params = postprocess_result['fit_params']
            LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
            for i in range(len(lane_params)):
                LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))

        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        #save images
        cv2.imwrite(save_image_path + 'binary_seg_image.jpg', binary_seg_image[0] * 255)
        cv2.imwrite(save_image_path +  'instance_seg_image.jpg', embedding_image)
        cv2.imwrite(save_image_path + 'mask_image.jpg', mask_image)

        # plt.figure('mask_image')
        # plt.imshow(mask_image[:, :, (2, 1, 0)])
        # plt.figure('src_image')
        # plt.imshow(image_vis[:, :, (2, 1, 0)])
        # plt.figure('instance_image')
        # plt.imshow(embedding_image[:, :, (2, 1, 0)])
        # plt.figure('binary_image')
        # plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        # plt.show()

    sess.close()

    return

def preparing_dir(dir, old_dir):
    if not ops.isfile(dir):
        try:
            copy = old_dir
            old_dir = list(map(lambda x: os.path.join(old_dir, x), os.listdir(old_dir)))
            old_dir = list(filter(lambda x: os.path.isdir(x), old_dir))
            old_dir = list(map(lambda x: x.replace(copy, dir), old_dir))
            for path in old_dir:
                print(path)
                os.mkdir(path)
                preparing_dir(path, path.replace(dir, copy))

        except Exception as e:
            print(e)

# #getting all files in the directory
# def get_files(dir):
#     contents = list(map(lambda x: ops.join(dir, x), os.listdir(dir)))
#     files = list(filter(lambda x: ops.isfile(x) and x.endswith(".jpg"), contents))
#     dirs = list(filter(lambda x: ops.isdir(x), contents))
#     x = files
#     for dir in dirs:
#         x += get_files(dir)
#     return x

# if __name__ == '__main__':
#     """
#     test code
#     """
#     # init args
#     args = init_args()
#     data_dir = args.image_path
#     results_dir = args.save_path

#     if not ops.exists(results_dir):
#         os.mkdir(results_dir)
#     preparing_dir(results_dir, data_dir)
#     print("Prepared Directories")

#     files = get_files(data_dir)
#     print("Got {x} Files".format(x=len(files)))

#     for file in tqdm(files):
#         print(file)
#         save_image_path = file.replace(data_dir, results_dir)
#         test_lanenet(file, args.weights_path, results_dir + "/", with_lane_fit=args.with_lane_fit)

    