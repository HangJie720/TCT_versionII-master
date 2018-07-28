# -*- coding: utf-8 -*-
import cv2
import numpy as np
import functools
import time

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return retargs
    return newfunc


def RGB_convert_BGR(img_rgb):
    r,g,b = cv2.split(img_rgb)
    new_img = cv2.merge([b,g,r])
    return new_img

#图像尺寸处理成target_size大小
def process_size(img_bgr, target_size=300):
    height, width, channel = img_bgr.shape
    if height > target_size or width > target_size:
        p_h = float(height) / target_size
        p_w = float(width) / target_size
        p_f = max(p_h, p_w)
        img_bgr = cv2.resize(img_bgr, None, fx=1 / p_f, fy=1 / p_f)
    h, w, c = img_bgr.shape
    img = np.zeros([target_size, target_size, 3])
    offset_h = int((target_size-h)/2)
    offset_w = int((target_size-w)/2)
    img[offset_h:offset_h+h, offset_w:offset_w+w,:]=img_bgr
    return img

def BGR_to_RGB(img_bgr):
    b,g,r = cv2.split(img_bgr)
    img_rgb = cv2.merge([r,g,b])
    return img_rgb


@timeit
def data_process(img_rgb):
    img_bgr = RGB_convert_BGR(img_rgb)#转化为bgr
    img_bgr = cv2.bilateralFilter(img_bgr, 3, 150, 150)#进行双边滤波处理
    img_rgb = BGR_to_RGB(img_bgr)  # 将bgr转换为rgb通道，同原始网络权重
    img_rgb = img_rgb / 255.
    img_rgb = process_size(img_rgb, target_size=300)
    img_pre = np.expand_dims(img_rgb, axis=0)
    return img_pre