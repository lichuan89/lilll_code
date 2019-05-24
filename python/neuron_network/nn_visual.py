#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import base64
import numpy as np
#import itertools
import os
import math
import sys
import random
from limg_common import equalization_gray_hist, even_random_idxs, gray_2_rgb, rgb_resize, rgb_resize, merge_images, border_image 
from lcv2_common import images_2_giffile, file_2_cv2image
from plt_common import show_images, show_array, show_predict_numbers  


def show_nn_iterations(X, Ys, T, label_arr, costs_arr, path='output/'):
    """
    X是图像数据集, 矩阵为 n, c, r, channel或者n, c, r
    Ys是预测结果的每次迭代的集合
    T是真实的结果
    """ 
    gif_fpath = '%s/data.train' % path
    tmp_fpath = '%s/tmp.bmp' % path

    # 按分类随机等量抽样
    sample_idxs = even_random_idxs(T, v_max_num=30) # 每个分类小图像随机选择v_max_num个 
    X = X[sample_idxs]
    Ys = Ys[:, sample_idxs]
    T = T[sample_idxs]
    
    #  灰度图像直方图均衡化
    X = equalization_gray_hist(255 - X, 0, 255)
       

    # 灰度转RGB
    if len(X.shape) == 3:
        X = gray_2_rgb(X) # 转换为n个rgb图像

    # 缩放图像尺寸
    #X = rgb_resize(X, (None, sample_width=8))


    # 绘制迭代过程: 把小图像合并拼接成大图像,再把大图像排成序列     
    iter_images = np.array([merge_images(X, Y, col_num=30,  delt=-1) for Y in Ys])

    # 减少大图像量 
    iter_idxs = np.array([i for i in range(0, len(Ys)) if i % 30 == 0 or i >= len(Ys) - 20])
    iter_images = iter_images[iter_idxs]

    # 转成gif
    images_2_giffile(iter_images, gif_fpath + '.image.gif', tmp_fpath=tmp_fpath, duration=0.2)

    
    # 绘制动态损失曲线每一帧
    iter_images = []
    for i in iter_idxs:
        show_array(costs_arr, label_arr, title='Decrease of cost over backprop iteration', show_fpath=tmp_fpath, stop=i)
        iter_images.append(file_2_cv2image(tmp_fpath))
    iter_images = np.array(iter_images)
    # 转成gif 
    images_2_giffile(iter_images, gif_fpath + '.curve.gif', tmp_fpath=tmp_fpath, duration=0.2) 

    # 绘制迭代过程中混淆函数的变化
    iter_images = []
    for i in iter_idxs:
        show_predict_numbers(T, Ys[i], show_fpath=tmp_fpath) 
        iter_images.append(file_2_cv2image(tmp_fpath))   
    iter_images = np.array(iter_images)
    # 转成gif 
    images_2_giffile(iter_images, gif_fpath + '.table.gif', tmp_fpath=tmp_fpath, duration=0.2) 

    # 合并gif 
    show_nn_iterations_gif(gif_fpath + '.image.gif', gif_fpath + '.curve.gif', gif_fpath + '.curve_image.gif')
    show_nn_iterations_gif(gif_fpath + '.curve.gif', gif_fpath + '.table.gif', gif_fpath + '.curve_table.gif')


def show_nn_iterations_gif(fpath1, fpath2, fpath3, tmp_fpath='tmp.bmp'):
    """
    合并多个图像
    """
    from lcv2_common import giffile_2_images, images_2_giffile
    from limg_common import border_image, region_image, rgb_resize
    image1 = np.array(giffile_2_images(fpath1))
    sample_image = border_image(image1[0][:, :, :3], 255, delt=4)
    l, r, u, d = region_image(sample_image, max_gray=100, delt=5)
    image1 = image1[:, u: d, l: r, :]
    image1 = rgb_resize(image1, (400, None))

    image2 = np.array(giffile_2_images(fpath2))
    sample_image = border_image(image2[0][:, :, :3], 255, delt=4)
    l, r, u, d = region_image(sample_image, max_gray=100, delt=5)
    image2 = image2[:, u: d, l: r, :] 
    image2 = rgb_resize(image2, (400, None))

    image12 = np.concatenate((image1, image2), axis=2)
    images_2_giffile(image12, gif_fpath=fpath3, tmp_fpath=tmp_fpath, duration=0.2)
