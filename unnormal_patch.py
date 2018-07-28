#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2
import time
import functools

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


#GLCM特征
def feature_extract_GLCM(img_grey):
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    result = greycomatrix(img_grey, [1], angles, symmetric=True, normed=True)

    contrast = greycoprops(result, 'contrast')#对比度
    homogeneity = greycoprops(result, 'homogeneity')#逆差矩
    asm = greycoprops(result, 'ASM')#能量
    correlation = greycoprops(result,'correlation')#自相关

    feature = []
    for i in range(len(angles)):
        sub_feature = [contrast[0][i], homogeneity[0][i], asm[0][i], correlation[0][i]]
        feature = feature + sub_feature
    feature = np.array(feature)
    return feature


@timeit
def invalid_based_filter_one(filter_model, img_data):
    if img_data.sum() == 0:  # 切出全零区域
        return True#全零区域为无效区域
    feature = feature_extract_GLCM(img_data)
    feature = np.expand_dims(feature, axis=0)
    # feature_change = pca_model.transform(feature)
    label = filter_model.predict(feature)
    if label==1:#表示无效区域
        return True
    else:
        return False


def data_process(img_data):
    img_data = cv2.bilateralFilter(img_data, 3, 150, 150)
    img_data = img_data / 255.
    img_data = np.expand_dims(img_data, axis=0)
    img_data = np.expand_dims(img_data, axis=3)
    return img_data


@timeit
def focus_based_autoencoder(feature_extractor, svm_one_class_model_list, img_data):
    '''
    采用autoencoder提取焦点区域
    :param feature_extractor: autoencoder feature extractor
    :param svm_one_class_model_list: 采用模型集成投票策略，包含不同模式训练出的svm_one_class model
    :param img_data: 灰度图，图像大小采用128*128
    :return: if focus , return true, else, return false 
    '''
    data = data_process(img_data)
    data_pre = feature_extractor.predict(data)
    label = []
    for model in svm_one_class_model_list:
        label.append(model.predict(data_pre)[0])
    if 1 in label:#表示焦点区域
        return True
    else:
        return False