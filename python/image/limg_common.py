#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import os
import io
import base64
import numpy as np
import urllib
import lcommon

def calc_arr_wrapper(func, images, *arg):
    """
    批量处理图像
    """
    arr = [func(image, *arg) for image in images]
    return np.array(arr) 
    

def rgb_2_gray(image):
    """
    输入图像可以是num, height, width, channel或height, width, channel
    gray = r * 0.3 + g * 0.59 + b * 0.11
    等价于: np.asarray(image.convert('L'))
    等价于: cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    """
    if len(image.shape) == 3:
        gray = (image[:, :, 0] * 0.3 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.11).astype('int')
    else:
        gray = (image[:, :, :, 0] * 0.3 + image[:, :, :, 1] * 0.59 + image[:, :, :, 2] * 0.11).astype('int')
    return np.array(gray, dtype='uint')


def gray_2_rgb(image):
    """
    灰度图像矩阵转换为RGB格式,输入图像可以是num, height, width或height, width
    """
    if len(image.shape) == 2:
        rgb = np.zeros([image.shape[0], image.shape[1], 3])
        rgb[:, :, 0] = rgb[:, :, 1] = rgb[:, :, 2] = image
    else:
        rgb = np.zeros([image.shape[0], image.shape[1], image.shape[2], 3])
        rgb[:, :, :, 0] = rgb[:, :, :, 1] = rgb[:, :, :, 2] = image
    return rgb


def gray_2_hist(grayImage):
    if len(grayImage.shape) == 3:
        return  calc_arr_wrapper(gray_2_hist, grayImage)
    grayImage = np.array(grayImage, dtype='int')
    hist = np.zeros((256), np.int)
    for rr in range(grayImage.shape[0]):
        for cc in range(grayImage.shape[1]):
            hist[grayImage[rr, cc]] += 1
    return hist


def equalization_gray_hist(image, min_v=0, max_v=250):
    """
    对灰度图像进行直方图均衡化
    """
    if len(image.shape) == 2:
        hist = gray_2_hist(image)
        hist = [int(v) for v in hist.reshape(-1)]
        thre = int(0.1 * sum(hist))
        n, low = 0, 0
        for i in range(len(hist)):
            n += hist[i]
            if n !=0 and n >= thre:
                low = i
                break
        n, high = 0, len(hist) - 1
        for i in range(len(hist)):
            n += hist[len(hist) - i - 1]
            if n != 0 and n >= thre: 
                high = len(hist) - i - 1
                break
        image = 1.0 * (max_v - min_v) * (image - low) / (0.00000001 + high - low) + min_v
        image[image < 0] = 0
        image[image > 255] = 255
        return image    
    else:
        return np.array([equalization_gray_hist(v, min_v, max_v) for v in image])

def rgb_2_hist(image, l=16):
    if len(image.shape) == 4:
        return  calc_arr_wrapper(rgb_2_hist, image, l)
    hist = np.zeros([l*l*l], np.int)
    hsize = 256/l
    rows, cols, channels = image.shape
    for rr in range(rows):
        for cc in range(cols):
            b, g, r = image[rr, cc]
            index = np.int(b/hsize)*l*l + np.int(g/hsize)*l + np.int(r/hsize)
            hist[np.int(index)] = hist[np.int(index)] + 1
    return hist


def rgb_resize_v0(image, size):
    """
    废弃
    图像是numpy矩阵, shape是col_num * row_num * channel_num, 也可以是多一个图像个数维度
    缩放到某个尺寸size = (heigh, width), 如果height或width为None，则等比例缩放, 如果size是数值, 则所发size倍
    """
    if len(image.shape) == 3:
        height, width, channel = image.shape
        if type(size) == int or type(size) == float:
            dstHeight, dstWidth = max(1, int(height * size)), max(1, int(width * size))
        else:
            dstHeight, dstWidth = size
            if dstWidth is None:
                dstWidth = dstHeight * width / height
            if dstHeight is None:
                dstHeight = dstWidth * height / width
        dstImage = np.zeros((dstHeight, dstWidth, channel), np.uint8)
        for row in range(dstHeight):
            for col in range(dstWidth):
                oldRow = int(row * (height * 1.0 / dstHeight))
                oldCol = int(col * (width * 1.0 / dstWidth))
                dstImage[row, col] = image[oldRow, oldCol]
    else:
        dstImage = np.array([rgb_resize(src, size) for src in image])
    return dstImage


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


def rgb_fill_resize(image, size, is_fill=True):
    """
    图像是numpy矩阵, col_num * row_num * channel_num
    缩放到某个尺寸size = (heigh, width)
    等比例缩放, is_fill表示是否填充空白区域
    """
    height, width = image.shape[: 2]
    dstHeight, dstWidth = size
    changeHeight = int(height * dstWidth / width)
    changeWidth = int(width * dstHeight / height)
    if changeHeight <= dstHeight:
        h, w = changeHeight, dstWidth
    else:
        h, w = dstHeight, changeWidth
    image = rgb_resize(image, (h, w))
    if not is_fill:
        return image
    dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)
    dstImage[0: h, 0: w, :] = image
    return dstImage


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
  

def matrix_2_str(data):
    """
    numpy矩阵序列化为字符串
    """
    arr = [base64.b64encode(data.tostring()), list(data.shape)]
    return lcommon.json_2_str(arr)


def str_2_matrix(string, dtype='float'):
    """
    字符串反序列化为numpy矩阵
    """
    arr = lcommon.str_2_json(string)
    matrix = np.fromstring(base64.b64decode(arr[0]), dtype=dtype).reshape(arr[1])
    return matrix
 

def random_vec(vec, n):
    """
    随机选择nunpy矩阵的n行, 选择后顺序保持不变
    返回抽样后的矩阵, 抽样的行号
    """

    # replace=False表示不放回抽样
    if n > vec.shape[0]:
        return vec, np.arange(0, vec.shape[0])
    nums = np.random.choice(np.arange(vec.shape[0]), size=(n), replace=False)
    nums.sort()
    return vec[nums], nums 


def even_random_idxs(T, v_max_num):
    """
    T是一维度矩阵, 对每一个元素值随机选取v_max_num个元素,取索引值
    """
    v_2_idxs = {}
    for idx in range(T.shape[0]):
        v = T[idx]
        v_2_idxs.setdefault(v, [])
        v_2_idxs[v].append(idx)
    idxs = []
    for v, sub_idxs in v_2_idxs.items():
        random_sub_idxs, _ = random_vec(np.array(sub_idxs), v_max_num)
        idxs += list(random_sub_idxs) 
    idxs = np.array(idxs)
    return idxs

def merge_images(images, T = None, col_num=5, delt=2):
    """
    images矩阵为num, height, width, channel，输入的图像尺寸一样,
    T表示对对应的图像使用某种颜色(可以是独热编码), T为None则使用同一种颜色
    最终行合并5个图像,
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
        -1: [0, 0, 0]
    } 
    arr = [] 
    elem = [] 
    i = 0
    images = np.array(images)
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
    if elem != []:
        for i in range(col_num - len(elem)):
            elem.append(border_image(np.zeros(images[0].shape), -1, delt))
        arr.append(np.hstack(elem))
    arr = np.vstack(arr)
    return arr


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


def region_image(image, max_gray=None, min_gray=None, delt=0):
    mark_h, mark_w, mark_c = image.shape 
    mark_sum = np.sum(image, axis=2) / mark_c 
    if max_gray is not None:
        idxs = np.where(mark_sum <= max_gray)
    else:
        idxs = np.where(mark_sum >= min_gray)
    u, d, l, r = np.min(idxs[0]), np.max(idxs[0]), np.min(idxs[1]), np.max(idxs[1])
    l, r, u, d = max(l - delt, 0), min(r + 1 + delt, mark_w), max(u - delt, 0), min(d + 1 + delt, mark_h)
    return l, r, u, d


def rgb_2_ycrcb(image):
    """
    rgb图像转换为yCrCb图像,同cv2中的cv2.cvtColor(bgr, cv2.COLOR_BGR2YCR_CB)
    https://www.cnblogs.com/geekite/p/5577987.html
    https://blog.csdn.net/yangzhao0001/article/details/65449171
    https://www.cnblogs.com/blue-lg/archive/2011/12/07/2279879.html
    """
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B 
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    Cb = -0.1687 * R - 0.3313 * G + 0.50 * B + 128
    ycrcb = np.zeros(image.shape)
    ycrcb[:, :, 0] = Y
    ycrcb[:, :, 1] = Cr
    ycrcb[:, :, 2] = Cb
    ycrcb = np.rint(ycrcb)
    ycrcb[ycrcb > 255] = 255
    ycrcb[ycrcb < 0] = 0
    return np.array(ycrcb, dtype='int')


def simple_extract_skin(image):

    # 133≤Cr≤173，77≤Cb≤127是肤色区域
    thres = [(78, 246), (129, 169), (92, 126)]
    ycrcb = rgb_2_ycrcb(image)
    col, row, channel = image.shape
    template = np.zeros((col, row))
    for c in range(col):
        for r in range(row):
            Y, Cr,Cb = ycrcb[c, r, :]
            if thres[0][0] <= Y <= thres[0][1] and thres[1][0] <= Cr <= thres[1][1] and thres[2][0] <= Cb <= thres[2][1]:
                template[c, r] = 255
    return template


def sobel_gray_image(img):
    if len(img.shape) == 3:
        img = rgb_2_gray(img) 
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])  
    s_suanziY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])     
    for i in range(r-2):
        for j in range(c-2):
            new_imageX[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziX))
            new_imageY[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziY))
            #new_image[i+1, j+1] = (new_imageX[i+1, j+1]*new_imageX[i+1,j+1] + new_imageY[i+1, j+1]*new_imageY[i+1,j+1])**0.5
            new_image[i+1, j+1] = min(new_imageX[i+1, j+1] * 0.5 + new_imageY[i+1, j+1] * 0.5, 255)
    return np.uint8(new_image) 


def sobel(image):
    gray_image = rgb_2_gray(image)
    sobel_image = sobel_gray_image(gray_image)
    return sobel_image

def test(tag):
    from lcv2_common import cv2image_2_file, file_2_cv2image, cv2image_2_image, image_2_cv2image

    fpath = '../../data/example/lena.jpg'
    if not os.path.exists('output/'):
        os.mkdir('output/')

    if tag == 'rgb_2_ycrcb' or tag == 'all':
        rgb = np.array([[0, 0, 250], [0, 100, 0], [100, 0, 0]], dtype='uint8').reshape(1, 3, 3)
        ycrcb = rgb_2_ycrcb(rgb)
        print 'ycrcb:', ycrcb 


    if tag == 'simple_extract_skin' or tag == 'all':
        image = cv2image_2_image(file_2_cv2image(fpath))
        template = simple_extract_skin(image)
        cv2image_2_file(gray_2_rgb(template), 'output/skin.bmp')

    if tag =='hist' or tag == 'all':
        image = cv2image_2_image(file_2_cv2image(fpath))
        gray = gray_2_rgb(rgb_2_gray(image)) 
        cv2image_2_file(gray, 'output/gray.bmp')
        hist = gray_2_hist(rgb_2_gray(image))
        print 'hist:', hist
        l, r, u, d = region_image(image, max_gray=222, min_gray=None, delt=10)
        print l, r, u, d, image.shape
        image = image[u: d, l: r]
        smallImage = rgb_resize(image, (40, 40), 'random')
        smallImage = rgb_resize(smallImage, (400, 400))
        hist = rgb_2_hist(smallImage, l=4)
        print 'rgb_hist:', hist
        cv2image_2_file(image_2_cv2image(smallImage), 'output/resize.bmp')
        
        image = cv2image_2_image(file_2_cv2image(fpath))
        #from lcv2_common import sobel_gray_image 
        sobel_image = gray_2_rgb(sobel_gray_image(np.array(rgb_2_gray(image), dtype='uint8')))
        cv2image_2_file(image_2_cv2image(sobel_image), 'output/sobel.bmp')
        
 
    if tag == 'equalization_gray_hist' or tag == 'all':
        image = cv2image_2_image(file_2_cv2image(fpath))
        image = equalization_gray_hist(image)
        cv2image_2_file(image_2_cv2image(image), 'output/equalization_gray.bmp')

    if tag == 'even_random_idxs' or tag == 'all':
        vec = np.array([6, 5, 4, 3, 2, 1])
        r_vec, nums = random_vec(vec, 3)
        print r_vec, nums
        T = np.array([5, 6, 5, 5, 6, 4, 4, 4, 4, 4])
        idxs = even_random_idxs(T, v_max_num = 3)
        print idxs, T[idxs]

    if tag == 'region_image' or tag == 'all':
        from lcv2_common import cv2image_2_file, file_2_cv2image
        image = cv2image_2_image(file_2_cv2image(fpath))
        image = border_image(image, (128, 0, 0), delt=20)
        image = border_image(image, (255, 255, 255), delt=-40)
        l, r, u, d = region_image(image, max_gray=240, min_gray=None, delt=5)
        print l, r, u, d, image.shape
        image = image[u: d, l: r]
        cv2image_2_file(image_2_cv2image(image), 'output/region.bmp')
    

if __name__ == "__main__":
    # cv2和numpy中的图片shape都是(height, width), pil中的图片size是(width, height)
    # cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
    tag = 'all'
    test(tag)
