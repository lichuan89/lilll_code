#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import numpy as np
import sys
import random
import os
import cv2
import sys
import urllib
import io
import base64
import sys
import copy
import traceback
import common
import img_common
from common import log

g_open_debug = False 

def cv2image_2_file(image, fpath):
    image = np.array(image, dtype='uint8')
    cv2.imwrite(fpath, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def format_cv2image(image):
    """
    rgb -> bgr or bgr -> rgb
    注: 
    cv2.imread(..) 读取的图像矩阵是BGR的,需要转换一下;
    cv2.imwrite(..)写入文件的图像矩阵也必须是BGR的,需要转换一下 
    """
    if len(image.shape) == 4:
        mat = np.zeros(image.shape)
        mat[:, :, :, 0] = image[:, :, :, 2]
        mat[:, :, :, 1] = image[:, :, :, 1]
        mat[:, :, :, 2] = image[:, :, :, 0]
    elif len(image.shape) == 3 and image.shape[2] >= 3:
        mat = np.zeros(image.shape)
        mat[:, :, 0] = image[:, :, 2]
        mat[:, :, 1] = image[:, :, 1]
        mat[:, :, 2] = image[:, :, 0]
    elif len(image.shape) == 3 and image.shape[2] == 1:
        mat = np.zeros((image.shape[0], image.shape[1], 3))
        mat[:, :, 0] = image[:, :, 0]
        mat[:, :, 1] = image[:, :, 0]
        mat[:, :, 2] = image[:, :, 0]
    else: #  灰度图像
        mat = np.zeros((image.shape[0], image.shape[1], 3))
        mat[:, :, 0] = image
        mat[:, :, 1] = image
        mat[:, :, 2] = image
    return mat


def image_2_cv2image(image):
    return format_cv2image(image)


def cv2image_2_image(image):
    return format_cv2image(image)

def file_2_cv2image(fpath):
    try:
        if fpath.find('http') == 0:
            #fpath = fpath.replace("ms.bdimg.com", "su.bcebos.com")
            #fpath = fpath.replace("boscdn.bpc.baidu.com", "su.bcebos.com")
            context = urllib.urlopen(fpath).read()
            id = os.getpid() 
            fpath = '/tmp/tmp.%d.jpg' % id
            common.str_2_file(context, fpath)
        image = cv2.imread(fpath)
        return image # BGR np.array
    except IOError as error:
        print >> sys.stderr, error
        return None

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


def light_image(image, a=1, b=0):
    """
    亮度缩放a倍,偏置b
    """
    image = image.copy()
    image = image * a + b
    image[image[:, :, :] > 255] = 255
    image[image[:, :, :] < 0] = 0 
    return image

def resize_image(image, width, height): 
    return cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)


def sobel_gray_image(img):
    """ 
    sobel_gray_image 
    """
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  
  
    absX = cv2.convertScaleAbs(x)   
    absY = cv2.convertScaleAbs(y)  
  
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)    
    return dst



def template_image(template, image, point=255):
    """
    对模版template上,灰度为point的点，在image上取0值 
    """
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if template[x][y] != point:
                image[x][y] = 0
            elif image[x][y] == 0:
                image[x][y] = 1
    points = np.array([
            image[x][y] \
            for x in range(image.shape[0]) \
            for y in range(image.shape[1]) if template[x][y] == point
        ])
    return image, points


def check_gray_muzzy(image, width):
    try:
        image = cv2.resize(image, (width, width * image.shape[0] / image.shape[1]))
        laplacian_muzzy = cv2.Laplacian(image, cv2.CV_64F).var()

        #image = cv2.blur(image, (3,3))
        sobel_image = sobel_gray_image(image)
        sobel_q = sobel_image.var()
        
        ret, otsu_image = cv2.threshold(sobel_image, 0, 255, cv2.THRESH_OTSU)
    
        kernel = np.ones((3, 3), np.uint8)
        adjust_image = cv2.dilate(otsu_image, kernel, 1)
        adjust_image = cv2.erode(adjust_image, kernel, 1)
        kernel = np.ones((11, 11), np.uint8)
        adjust_image = cv2.dilate(adjust_image, kernel, 1)
       
        sobel_sobel_image = sobel_gray_image(sobel_image)
        target_image, points = template_image(adjust_image, sobel_sobel_image)
        q = points.var()
        check_image, _ = template_image(adjust_image, image)
        cache = {
                'sobel_image': sobel_image,
                'otsu_image': otsu_image,
                'sobel_sobel_image': sobel_sobel_image,
                'adjust_image': adjust_image,
                'target_image': target_image,
                'check_image': check_image,
            }
        return q, cache
    except IOError as error:
        return None


def random_expand_image(image):
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



def imagepath_2_matrix(files, width=200, height=200, tag_num=2, is_fill=True):
    """ 
    files = {'image_file1': '1', 'image_url1': '0'}
    从图片路径中取出图片, 转换成矩阵: 图像个数 * 行数 * 列数 * 通道数(RGB) -> 标签 
    """
    data = []  
    T = []
    shapes = {} 
    for fname, tag in files.items():
        # 抓图
        cv2image = file_2_cv2image(fname)
        if cv2image is None:
            print >> sys.stderr, 'failed to wget image:%s' % fname 
            continue
        image =  cv2image_2_image(cv2image)
        if image is None:
            print >> sys.stderr, 'failed to format image:%s' % fname 
            continue
        if not is_fill:
            image = resize_image(image, width, height)
        else:
            image = np.array(img_common.rgb_fill_resize(image, (width, height)), dtype='float')

        data.append(image)
        t = np.zeros((tag_num))
        t[int(tag)] = 1.0
        T.append(t)
        shapes[image.shape] = 1
    if len(shapes) == 1:
        data = np.array(data)
    T = np.array(T).reshape(-1, tag_num)
    return data, T


def border_image(image, t, delt=2):
    """
    rgb图像矩阵: n, r, c, channel 或 r, c, channel
    对边缘画框，值为t, 间隔为delt, delt为负表示往外扩展
    """
    image = image.copy()
    if delt is None:
        return image

    is_expand = False
    if delt < 0:
        delt = - delt
        is_expand = True
    if len(image.shape) == 4:
        n, r, c, channel = image.shape
        r_delt, c_delt = delt, delt
        if is_expand:
            image = np.pad(image, ((0, 0), (r_delt, r_delt), (c_delt, c_delt), (0, 0)), 'constant')
        n, r, c, channel = image.shape
        image[:, 0: r_delt, :, :] = t
        image[:, r - r_delt:, :, :] = t
        image[:, :, 0: c_delt, :] = t
        image[:, :, c - c_delt, : ] = t
    else:
        r, c, channel = image.shape
        r_delt, c_delt = delt, delt
        if is_expand:
            image = np.pad(image, ((r_delt, r_delt), (c_delt, c_delt), (0, 0)), 'constant')
        r, c, channel = image.shape
        image[0: r_delt, :, :] = t
        image[r - r_delt:, :, :] = t
        image[:, 0: c_delt, :] = t
        image[:, c - c_delt:, : ] = t
        
    return image

def merge_images(images, T, col_num=5, delt=2):
    """
    废弃g图像尺寸一样
    """
    color = {
        0: [255,   0,   0],
        1: [0,   255,   0],
        2: [0,     0, 255],
        3: [255, 255,   0],
        4: [255,   0, 255],
        5: [0,   255, 255],
        6: [128, 0, 0],
        7: [0, 128, 0],
        8: [0, 0, 128],
        9: [0, 128, 128],
        10: [128, 0, 128],
        -1: [128, 128, 0]
    } 
    arr = [] 
    elem = [] 
    i = 0
    for i in range(images.shape[0]):
        if T is None:
            v = color[-1]
        elif int(np.max(T)) > 1:
            v = int(T[i])
            v = color[v] if v in color else color[-1]
        else:
            v = int(np.argmax(T[i]))
            v = color[v] if v in color else color[-1]
        elem.append(border_image(images[i], v, delt))
        if len(elem) == col_num:
            arr.append(np.hstack(elem))           
            elem = []
    # 尾部图片数量不足, 丢弃尾部图片
    #if elem != []:
    #    arr.append(elem)
    arr = np.vstack(arr)
    return arr
     

def collect_image_fea(data, T, expand_num=10, use_fea='pixel'):
    """
    files = {'image_file1': '1', 'image_url1': '0'}
    expand_num, 扩展图片的个数，针对1的case
    废弃
    """
    expand_data = []
    expand_T = []
    for i in range(data.shape[0]):
        if int(np.argmax(T[i])) != 0: # 扩展badcase图片
            expand_data += [random_expand_image(data[i]) for i in range(expand_num)]
            expand_T += [T[i] for i in range(expand_num)]
    expand_data = np.array(expand_data)
    expand_T = np.array(expand_T)
    data = np.vstack((data, expand_data))
    T = np.vstack((T, expand_T))
    log('finish to expand images. image_num:', data.shape, '; expand_image_num:', expand_data.shape)
   
    fea_data = None
    for i in range(data.shape[0]):
        image = data[i]
        # 抽取特征
        if use_fea == 'pixel':
            vec = img_common.rgb_2_feature(image, gray_size=(40, 40), hist_size=False, rgb_hist_dim=False, gray_hist=False)
        elif use_fea == 'rgbhist':
            vec = img_common.rgb_2_feature(image, gray_size=False, hist_size=(100, 100), rgb_hist_dim=16, gray_hist=False)
        elif use_fea == 'pixel_rgbhist':
            vec = img_common.rgb_2_feature(image, gray_size=(40, 40), hist_size=(100, 100), rgb_hist_dim=16, gray_hist=False)
        fea_data = np.vstack((fea_data, vec)) if fea_data is not None else vec
    log('finish to extract feature. data:', fea_data.shape, '; T:', T.shape) 
    return fea_data, T 

def test_read_write():
    image = np.array( # 2行3列的RGB图像
        [
            [[254,   0,   5], [255,   4,   0], [  0,   0,   2], [ 98, 162, 224],],
            [[252,   2,   3], [  1, 255,   0], [  0,   3, 255], [254, 253, 249],],
        ], dtype='uint8'
        )
    cv2image_2_file(image_2_cv2image(image), 'code.bmp')
    image = cv2image_2_image(file_2_cv2image('code.bmp'))
    print image

def test_opt():
    image = cv2image_2_image(file_2_cv2image('test.jpg'))
    #image = rotate_image(image, 20)
    #image = light_image(image, 1, -40)
    #image = resize_image(image, width=100, height=50)
    #image = move_image(image, 50, 50)
    image = random_expand_image(image)
    cv2image_2_file(image_2_cv2image(image), 'test.bmp')


def ramdom_show_images(data, T, col_num, row_num, delt=3, is_gray=True, save_file=None):
    if col_num * row_num >= T.shape[0]:
        row_num = int(T.shape[0] / col_num)
        nums = np.arange(col_num * row_num)
    else:
        nums = np.random.choice(np.arange(data.shape[0]), size=(col_num * row_num))
    if is_gray:
        random_data = data[nums, :, :]
        random_T = T[nums] 
        imgs = np.zeros(list(random_data.shape) + [3])
        imgs[:, :, :, 0] = random_data 
        imgs[:, :, :, 1] = random_data
        imgs[:, :, :, 2] = random_data
        random_data = imgs
    else:
        random_data = data[nums, :, :, :]
        random_T = T[nums] 
    merge_image = merge_images(random_data, random_T, col_num=col_num, delt=delt)
    if save_file is not None:
        cv2image_2_file(image_2_cv2image(merge_image), save_file)
    return merge_image



def test_img_fea():

    # 抓取图片
    files = {
        'http://ms.bdimg.com/dsp-image/1371700633.jpg': 1, # badcase
        'http://ms.bdimg.com/dsp-image/1373219344.jpg': 1, 
        'http://ms.bdimg.com/dsp-image/259532757.jpg': 1,
        'http://ms.bdimg.com/dsp-image/828016156.jpg': 0,
        'http://ms.bdimg.com/dsp-image/949458037.jpg': 0,
        'http://ms.bdimg.com/dsp-image/553264679.jpg': 0,
    }
    data_fpath = 'data.txt'
    T_fpath = 'T.txt'
    data, T = imagepath_2_matrix(files, width=150, height=150, tag_num=2) # 根据图像路径载入图片内容矩阵


    # 序列化与反序列化
    common.str_2_file(img_common.matrix_2_str(data), data_fpath) # 矩阵序列化到文件中
    common.str_2_file(img_common.matrix_2_str(T), T_fpath)
    data2 = img_common.str_2_matrix(common.file_2_str(data_fpath)) # 文件中读取矩阵
    T2 = img_common.str_2_matrix(common.file_2_str(T_fpath), dtype='int') 
   
    # 合并图片
    merge_image = merge_images(data2, T2, col_num=2)
    cv2image_2_file(image_2_cv2image(merge_image), 'test.bmp')


    # 提取图片特征
    data, T = collect_image_fea(data, T, expand_num=2, use_fea='pixel_rgbhist')
    #data, T = collect_image_fea(data, T, expand_num=2, use_fea='pixel')
    #data, T = collect_image_fea(data, T, expand_num=2, use_fea='rgbhist')
    print T


def test_muzzy():
    image = file_2_cv2image('muzzy/1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_image = sobel_gray_image(image)
    cv2image_2_file(sobel_image, 'test.bmp')
    muzzy_q, cache = check_gray_muzzy(image, width=200)
     

def test_proces_dir(path):
    arr = os.listdir(path)
    for v in arr:
        if v.find('.') == 0:
            continue
        f = '%s/%s' % (path, v)
        if os.path.isfile(f):
            image = file_2_cv2image(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            muzzy_q, cache = check_gray_muzzy(image, width=500)
            for k, img in cache.items():  
                cv2image_2_file(img, 'output/%s.%s.bmp' % (v, k))
            print '%s\t%f' % (f, muzzy_q)


def mask_image(image, text='lichuan89', coordinates=(20, 20), transparency=1, size=1, color=(1, 1, 255), thickness=1):
    h, w, c = image.shape 
    mark_h, mark_w, mark_c = 200, 800, 3 
    mark = np.zeros((mark_h, mark_w, mark_c))
    #cv2image_2_file(image, 'data/test/test.bmp')
    # putText第三个参数表示左下角坐标
    cv2.putText(mark, text, (0, mark_h/2), cv2.FONT_HERSHEY_PLAIN, size, color, thickness=thickness)
    mark_sum = np.sum(mark, axis=2)
    idxs = np.where(mark_sum > 0)
    u, d, l, r = np.min(idxs[0]), np.max(idxs[0]), np.min(idxs[1]), np.max(idxs[1])
    l, r, u, d = max(l - 2, 0), min(r + 2, mark_w), max(u - 2, 0), min(d + 2, mark_h)
    mark = mark[u: d, l: r, :] 
    mark_h, mark_w, mark_c = mark.shape 

    if coordinates is None:
        if h - mark_h < 0 or w - mark_w < 0:
            return image
        else:
            u, d, l, r = h - mark_h, h, w - mark_w, w
    else:
        if coordinates[1] + mark_h >= h or coordinates[0] + mark_w >= w:
            return image
        else:
            u, d, l, r = coordinates[1], coordinates[1] + mark_h, coordinates[0], coordinates[0] + mark_w 

    bottom = np.copy(image[u: d, l: r, :])
    bottom = np.array(bottom, dtype='uint')
    mark = np.array(mark, dtype='uint')
    if bottom.shape != mark.shape:
        return image
    overlapping = cv2.addWeighted(bottom, transparency, mark, 1 - transparency, 0)
    bottom = np.copy(image[u: d, l: r, :])
    bottom[mark != 0] = overlapping[mark != 0]
    image[u: d, l: r, :] = bottom 
    return image


def images_2_giffile(images, gif_fpath, tmp_fpath='tmp.images_2_gif.bmp', duration=0.2): 
    import imageio
    frames = []
    for image in images:
        image = image_2_cv2image(image)
        cv2image_2_file(cv2image_2_image(image), tmp_fpath)
        frames.append(imageio.imread(tmp_fpath))    
    imageio.mimsave(gif_fpath, frames, 'GIF', duration=duration)
    

def giffile_2_images(gif_fpath):
    import imageio
    images = imageio.mimread(gif_fpath, memtest=False)
    images = image_2_cv2image(np.array(images, dtype='uint'))
    return images
 

def test(tag):
    fpath = '../data/material/x.jpg'

    if tag == 'mask_image' or tag == 'all':
        image = cv2image_2_image(file_2_cv2image(fpath))
        #image = mask_image(image, text='lilll', coordinates=None, transparency=0.7, size=1.2, color=(255, 10, 10), thickness=2)
        image = mask_image(image, text='lilll', coordinates=(2000, 550), transparency=0.7, size=4.2, color=(10, 10, 255), thickness=8)
        cv2image_2_file(image_2_cv2image(image), '../data/material/x.png')

if __name__ == "__main__":
    tag = 'all'
    test(tag)
    #test_proces_dir('muzzy')
    #test_img_fea()

    #image = file_2_cv2image('example.jpg')
    #image = mask_image(image, text='lichuan', coordinates=(20, 40), transparency=0.7, size=5, color=(255, 255, 255), thickness=7)
    #cv2image_2_file(image, 'test.bmp')

    #images = giffile_2_images('data.train.gif')
    #images = np.array(images)
    #cv2image_2_file(images[0,:,:,: 3], '1.bmp')
    pass


