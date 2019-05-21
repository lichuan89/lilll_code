#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""
import cv2
import random
import lcommon
import os 
import sys
import numpy as np
import struct
import hashlib
import logging
import json 
import datetime
import re
import fileinput
import urllib
import urllib2
import threading
import urlparse
import time 
import multiprocessing
import fcntl 
g_open_debug=True
class Processor(multiprocessing.Process):
    """
    processor
    """
    def __init__(self, worker, lines, args):
        multiprocessing.Process.__init__(self)
        self.worker = worker
        self.lines = lines
        self.args = args[: -1]
        self.share = args[-1]
    
    def run(self):
        """ 
        run 
        """
        result = self.worker(self.lines, self.args)
        id = os.getpid()
        if type(self.share) == unicode or type(self.share) == type(''):
            # 用临时文件存储结果
            str_2_file('\n'.join(result).encode('utf8', 'ignore'), '%s/tmp.%s' % (self.share, id))
        else:
            self.share.setdefault(id, result)

def collect_train_data():
    thread_num = 11
    files = {}
    train_image_fpath = 'data/train_data.list'
    train_image_fpath = 'url.list'
    if os.path.exists('%s.fea' % train_image_fpath):
        data = np.loadtxt('%s.fea' % train_image_fpath, delimiter=",")
        T = np.loadtxt('%s.val' % train_image_fpath, delimiter=",")
    else:
        lines = [line[: -1] for line in open(train_image_fpath).readlines()]
        pairs = muti_process(lines, thread_num, single_thread_collect_train_data, args=[])
        data = None
        T = None
        for pair in pairs:
            sub_data, sub_T = pair
            data = np.vstack((sub_data, data)) if data is not None else sub_data
            T = np.vstack((sub_T, T)) if T is not None else sub_T
        np.savetxt('%s.fea' % train_image_fpath, data, fmt="%d", delimiter=",")
        np.savetxt('%s.val' % train_image_fpath, T, fmt="%d", delimiter=",")

    #data = data[:, 1600:]
    data = data[:, :1600]
    print data.shape, T.shape
    #data = data[range(0, data.shape[0], 3), :]
    #T = T[range(0, T.shape[0], 3), :]
    print data.shape, T.shape
    X_train, T_train, X_validation, T_validation, X_test, T_test = split_data(data, T)
    return X_train, T_train, X_validation, T_validation, X_test, T_test 
def split_data(data, T):
    from sklearn import datasets, cross_validation, metrics
    X_train, X_test, T_train, T_test = cross_validation.train_test_split(
            data, T, test_size=0.4)
    X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
            X_test, T_test, test_size=0.5)
    if True:
        print '训练集、校验集、测试集的X量级：', X_train.shape, X_validation.shape, X_test.shape
        print '训练集、校验集、测试集的Y量级：', T_train.shape, T_validation.shape, T_test.shape
    return X_train, T_train, X_validation, T_validation, X_test, T_test
def single_thread_collect_train_data(lines, args):
    """
    单线程处理部分数据
    """
    files = {} 
    for line in lines:
        fpath, tag = line.split('\t')
        files[fpath] = tag
    data, T = collect_image_fea(files, expend_num=2, use_fea='pixel')
    return [(data, T)]
def rgb_2_feature(image, gray_size=(50, 50), hist_size=(100, 100), rgb_hist_dim=16, gray_hist=False):
    """
    参数如果是False,表示不用这个特征
    """ 
    vec = np.ones((1, 0))
    if rgb_hist_dim != False or gray_hist != False:
        fea_img = rgb_resize(image, hist_size)
    # rgb直方图
    if rgb_hist_dim != False:
        rgb_hist = rgb_2_hist(fea_img, l=rgb_hist_dim)
    else:
        rgb_hist = np.ones((1, 0))

    # 灰度直方图
    if gray_hist != False: 
        gray_img = rgb_2_gray(fea_img)
        gray_hist = gray_2_hist(gray_img)
    else:
        gray_hist = np.ones((1, 0))

    # 像素矩阵
    if gray_size != False: 
        img = rgb_resize(image, gray_size)
        gray_fea = rgb_2_gray(img).reshape(1, gray_size[0] * gray_size[1])
    else:
       gray_fea = np.ones((1, 0))
    vec = np.hstack((rgb_hist, gray_hist, gray_fea)) 
    return vec
def rgb_resize(image, size, pattern='random'):
    """
    图像是numpy矩阵, shape是col_num * row_num * channel_num, 也可以是多一个图像个数维度
    缩放到某个尺寸size = (heigh, width), 如果height或width为None，则等比例缩放, 如果size是数值, 则所发size倍
    """
    if pattern == 'sobel':
        gray_image = rgb_2_gray(image)
        sobel_image = sobel_gray_image(gray_image)
    if len(image.shape) == 3:
        height, width, channel = image.shape
        if type(size) == int or type(size) == float: # 参数是缩放尺寸, 则根据尺寸定制宽高
            dstHeight, dstWidth = max(1, int(height * size)), max(1, int(width * size))
        else:
            dstHeight, dstWidth = size
            # 如果宽高有一个没设置，则按比例设置
            if dstWidth is None:
                dstWidth = dstHeight * width / height
            if dstHeight is None:
                dstHeight = dstWidth * height / width
        dstImage = np.zeros((dstHeight, dstWidth, channel), np.uint8)

        for row in range(dstHeight):
            for col in range(dstWidth):
                if pattern != 'random':
                    # 如果是缩小尺寸，则目标图像像素映射到原图的一个区间
                    low_row = int(np.floor(row * (height * 1.0 / dstHeight)))
                    low_col = int(np.floor(col * (width * 1.0 / dstWidth)))
                    high_row = int(np.ceil((row + 1) * (height * 1.0 / dstHeight)))
                    high_col = int(np.ceil((col + 1) * (width * 1.0 / dstWidth)))
                    if low_row >= high_row:
                        high_row = low_row + 1
                    if low_col >= high_col:
                        high_col = low_col + 1
                    region = image[low_row: high_row, low_col: high_col, :]
                    if pattern == 'sobel':
                        sobel_region = sobel_image[low_row: high_row, low_col: high_col]
                        idx = np.argmax(sobel_region)
                        val = (region[:, :, 0].reshape(-1)[idx], region[:, :, 1].reshape(-1)[idx], region[:, :, 2].reshape(-1)[idx])
                    if pattern == 'avg': 
                        n = (high_row - low_row) * (high_col - low_col)
                        val = (np.sum(region[:, :, 0]) / n, np.sum(region[:, :, 1]) / n, np.sum(region[:, :, 2]) / n)
                else:
                    oldRow = int(row * (height * 1.0 / dstHeight))
                    oldCol = int(col * (width * 1.0 / dstWidth))
                    val = image[oldRow, oldCol]
                dstImage[row, col] = val 
    else:
        dstImage = np.array([rgb_resize(src, size) for src in image])
    return dstImage

def resize_image(image, width, height): 
    return cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
def rgb_2_gray(image):
    """
    gray = r * 0.3 + g * 0.59 + b * 0.11
    等价于: np.asarray(image.convert('L'))
    等价于: cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    """
    gray = (image[:, :, 0] * 0.3 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.11).astype('int')
    return gray
def light_image(image, a=1, b=0):
    """
    亮度缩放a倍,偏置b
    """
    image = image.copy()
    image = image * a + b
    image[image[:, :, :] > 255] = 255
    image[image[:, :, :] < 0] = 0 
    return image
def random_expend_image(image):
    prob = random.random()
    a = max(0.3, prob * 2)
    prob = random.random()
    b = prob * 16 - 8
    image = light_image(image, a=a, b=b)

    prob = random.random()
    angle = prob * 60 - 30
    image = rotate_image(image, angle=angle)
    prob = random.random()
    x = prob * 20 -10 
    prob = random.random()
    y = prob * 20 - 10
    image = move_image(image, x, y)
    return image
def move_image(image, x, y):
    """
    左移x，下移y
    """
    image = image.copy()
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


def rotate_image(image, angle, center=None, scale=1.0):
    """
    左旋转angle度
    """
    image = image.copy()
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def collect_image_fea(files, expend_num=10, use_fea='pixel'):
    """
    files = {'image_file1': '1', 'image_url1': '0'}
    expend_num, 扩展图片的个数，针对1的case
    """
    mats = None
    data = None
    T = None
    idx = 0
    for fname in files:
        idx += 1

        # 抓图
        cv2image = file_2_cv2image(fname)
        if cv2image is None:
            print >> sys.stderr, 'failed to wget image:%s' % fname 
            continue
        image =  cv2image_2_image(cv2image)
        if image is None:
            print >> sys.stderr, 'failed to format image:%s' % fname 
            continue

        # 扩充图片
        imgs = [image]
        if files[fname] != '0': 
            imgs += [random_expend_image(image) for i in range(expend_num)]

        i = 0
        if g_open_debug:
            print >> sys.stderr, 'begin to process %s' % fname 
        for img in imgs:
            # 抽取特征
            if use_fea == 'pixel':
                vec = rgb_2_feature(image, gray_size=(40, 40), hist_size=False, rgb_hist_dim=False, gray_hist=False)
            elif use_fea == 'rgbhist':
                vec = rgb_2_feature(image, gray_size=False, hist_size=(100, 100), rgb_hist_dim=16, gray_hist=False)
            elif use_fea == 'pixel_rgbhist':
                vec = rgb_2_feature(image, gray_size=(40, 40), hist_size=(100, 100), rgb_hist_dim=16, gray_hist=False)
            data = np.vstack((data, vec)) if data is not None else vec
            if g_open_debug:
                mat = np.array(rgb_2_gray(resize_image(image, 50, 50)), ndmin=3)
                mats = np.vstack((mats, mat)) if mats is not None else mat
            t = np.array([[int(files[fname]), 1 - int(files[fname])]])
            T = np.vstack((T, t)) if T is not None else t 
            if g_open_debug:
                cv2image_2_file(image_2_cv2image(img), 'output/%s.%d.jpg' % (idx, i))
            i += 1
        if g_open_debug:
            print >> sys.stderr, 'finish to process %s' % fname 
    return data, T 

def cv2image_2_file(image, fpath):
    cv2.imwrite(fpath, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def image_2_cv2image(image):
    return format_cv2image(image)


def cv2image_2_image(image):
    return format_cv2image(image)
def format_cv2image(image):
    """
    rgb -> bgr or bgr -> rgb
    注: 
    cv2.imread(..) 读取的图像矩阵是BGR的,需要转换一下;
    cv2.imwrite(..)写入文件的图像矩阵也必须是BGR的,需要转换一下 
    """
    mat = np.zeros(image.shape)
    mat[:, :, 0] = image[:, :, 2]
    mat[:, :, 1] = image[:, :, 1]
    mat[:, :, 2] = image[:, :, 0]
    return mat
def muti_process(lines, thread_running_num, worker, args, use_share_path=None): 
    """ 
    多进程处理数据, 并输出 
    """
    if use_share_path is None:
        # 使用共享内存
        manager = multiprocessing.Manager()
        contexts = manager.dict()
    else:
        # 先并发输入文件,再统一搜集输出
        clear_dir(use_share_path)
        contexts = use_share_path 
    threadpool = []
    batch_arr = {}
    for i in range(len(lines)):
        k = i % thread_running_num 
        batch_arr.setdefault(k, [])
        batch_arr[k].append(lines[i])
    
    for idx in batch_arr:
        th = Processor(worker, batch_arr[idx], args + [contexts])
        threadpool.append(th)
    idx = 0 
    threads = []
    for th in threadpool:
        th.start()
        
    for th in threadpool:
        th.join()

    lines = []
    if use_share_path is not None:
        lines = read_dir(use_share_path)
        clear_dir(use_share_path)
    else: 
        for k, v in contexts.items():
            for line in v:
                lines.append(line)
    return lines

def clear_dir(path):
    """
    创建文件夹,或者删除文件夹下的直接子文件
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        arr = os.listdir(path)
        for v in arr:
            f = '%s/%s' % (path, v) 
            if os.path.isfile(f):
                os.remove(f)

def file_2_cv2image(fpath):
    try:
        if fpath.find('http') == 0:
            #fpath = fpath.replace("ms.bdimg.com", "su.bcebos.com")
            #fpath = fpath.replace("boscdn.bpc.baidu.com", "su.bcebos.com")
            context = urllib.urlopen(fpath).read()
            id = os.getpid() 
            fpath = '/tmp/tmp.%d.jpg' % id
            lcommon.str_2_file(context, fpath)
        image = cv2.imread(fpath)
        return image # BGR np.array
    except IOError as error:
        print >> sys.stderr, error
        return None

def test(tag):
    from lcv2_common import cv2image_2_file, file_2_cv2image, cv2image_2_image, image_2_cv2image

    fpath = '../../data/example/digits/1013_7.bmp'
    if not os.path.exists('output/'):
        os.mkdir('output/')

    if tag == 'collect_image_fea' or tag == 'all':
        files = {fpath : '7'}
        img_fea = collect_image_fea(files, expend_num=0, use_fea='pixel')
        print img_fea 
    if tag == '1':
        train_image_fpath = 'url.list'
        data = np.loadtxt('%s.fea' % train_image_fpath, delimiter=",")
        T = np.loadtxt('%s.val' % train_image_fpath, delimiter=",")
        print data.shape, T.shape
        data = data[:, : 1600].reshape((-1, 40, 40))
        T = np.argmax(T, axis=1) 
        fnames = ['%d_%d.bmp' % (i, T[i]) for i in range(T.shape[0])]
        from limgs_common import XImage_2_files 
        XImage_2_files(data, None, path='test/', fnames=fnames, prefix='')

if __name__ == "__main__":
    tag = 'all'
    test(tag)


