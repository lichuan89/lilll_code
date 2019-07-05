#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import urllib
import os
import numpy as np
from lcv2_common import cv2image_2_file, image_2_cv2image, file_2_cv2image, cv2image_2_image, sobel_gray_image, template_image
from limg_common import gray_2_rgb, rgb_resize
from lcommon import str_2_file, list_files
from lcmd import muti_process_stdin, muti_process

def urls_2_imageFiles(files):
    """
    files是一个url->fpath的dict, 抓取图像url 并存储起来
    """
    ret = {} 
    for url, fpath in files.items():
        try:
            path = '/'.join(fpath.split('/')[: -1])
            if not os.path.exists(path):
                os.makedirs(path)
        except IOError as error:
            print >> sys.stderr, error
        try:
            context = urllib.urlopen(url).read()
            str_2_file(context, fpath)
            ret[url] = fpath
        except IOError as error:
            print >> sys.stderr, error
    return ret

 

def XImage_2_files(X, shape, path, fnames, prefix=''):
    """
    输入X可以是np数组,也可以是list.
    将图像保存起来, X可能是灰度图像, fnames是文件名列表
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except IOError as error:
        print >> sys.stderr, error
    n = len(X) if type(X) == list else X.shape[0]
    for i in range(n):
        x = X[i]
        fname = fnames[i]
        if len(x.shape) == 2:
            x = gray_2_rgb(x)
        if shape is not None:
            x = rgb_resize(x, size=shape[: 2]) 
        cv2image_2_file(image_2_cv2image(x), '%s/%s%s' % (path, prefix, fname))
        #cv2image_2_file(image_2_cv2image(x), '%s/%s%s.bmp' % (path, prefix, fname))


def files_2_Ximage(path_arr, shape=None):
    """
    将图像文件读取出来，存储到矩阵中, 文件名存储到列表中
    """
    X = []
    fnames = []
    for f in path_arr:
        image = file_2_cv2image(f)
        if image is None:
            continue 
        image = cv2image_2_image(image)
        if shape is not None:
            image = rgb_resize(image, size=shape[: 2])
        X.append(image)
        fnames.append(f.split('/')[-1])
    if shape is not None and shape[0] is not None and shape[1] is not None:
        X = np.array(X, dtype='uint')
    #fnames = np.array(fnames)
    return X, fnames 


def quick_transform_image(input_path, output_path, func, size=(100, 100), prefix='', args=None):
    """
    从一个文件夹读入图像,处理后输出到另一个文件夹
    """
    if type(input_path) == list or type(input_path) == dict:
        files = input_path
    else:
        files = list_files(input_path)
    X, fnames = files_2_Ximage(files, size)
    X = [func(x) for x in X] if args is None else [func(x, args) for x in X]
    if size[0] is not None and size[1] is not None:
        X = np.array(X)
    quick_XImage_2_files(X, None, output_path, fnames, prefix)
    return files 

def quick_urls_2_imageFiles():
    """
    从输入流读取url \t fpath，抓取图像url 并存储起来
    """
    def worker(lines, arg):
        files = dict([line.split('\t')[: 2] for line in lines])
        ret = urls_2_imageFiles(files)
        output = []
        for line in lines:
            arr = line.split('\t')
            if arr[0] in ret:
                output.append(line)
        return output 
    muti_process_stdin(worker, [], batch_line_num=21, thread_running_num=7)


def quick_XImage_2_files(X, shape, path, fnames, prefix=''):
    def worker(lines, arg):
        shape, path, prefix = arg
        #X = np.array([line[0] for line in lines])
        X = [line[0] for line in lines]
        fnames = [line[1] for line in lines]
        XImage_2_files(X, shape, path, fnames, prefix)
    cnt = X.shape[0] if type(X) != list else len(X)
    lines = [(X[i], fnames[i]) for i in range(cnt)]
    muti_process(lines, 6, worker, args=[shape, path, prefix], use_share_path=None)


def quick_files_2_Ximage(path_arr, shape):
    def worker(lines, arg):
        shape = arg[0]
        path_arr = lines
        X, fnames = files_2_Ximage(path_arr, shape)
        return [(X, fnames)] 
    lines = muti_process(path_arr, 6, worker, args=[shape], use_share_path=None)
    X = []
    F = []
    for x, fnames in lines:
        X += [v for v in x]
        F += [v for v in fnames]
    X = np.array(X)
    return X, F 


def quick_rgb_2_hist(images, l=16):
    """
    将rgb图像转换成rgb直方图
    """
    from limg_common import rgb_2_hist 
    def worker(lines, arg):
        l = arg[0]
        idxs = [v[0] for v in lines]
        images = np.array([v[1] for v in lines])
        hists = rgb_2_hist(images, l)
        return [(idxs[i], hists[i]) for i in range(len(idxs))] 
    inputs = [(i, images[i]) for i in range(images.shape[0])] 
    lines = muti_process(inputs, 6, worker, args=[l], use_share_path=None)
    idx_2_hist = dict(lines)
    output = [idx_2_hist[i] for i in range(len(lines))]
    output = np.array(output) 
    return output


def template_visual_image(image, template, tag):
    """
    模版的像素值只有0和255
    """
    if tag == 'outline':
        image[image == 0] = 1
        template = sobel_gray_image(template)
        image[:, :, 0][template > 0] = 255 
    elif tag == 'maxcover':
        image[image == 0] = 1
        image[:, :, 0][template >= 128] = 0 
        image[:, :, 1][template >= 128] = 0
        image[:, :, 2][template >= 128] = 0
    elif tag == 'mincover':
        image[image == 0] = 1
        image[:, :, 0][template < 128] = 0 
        image[:, :, 1][template < 128] = 0
        image[:, :, 2][template < 128] = 0
    elif tag == 'bakground':
        image[image == 0] = 1
        image[:, :, 1][template > 128] = 255
        image[:, :, 2][template > 128] = 255
        template = sobel_gray_image(template)
        image[:, :, 0][template > 0] = 255 
    return image


def test(tag):
    path = '../../data/example/images/'
    fpath = '../../data/example/lena.jpg'
    if not os.path.exists('output/'):
        os.mkdir('output/')

    if tag == 'urls_2_imageFiles' or tag == 'all':
        files = {
            'http://ms.bdimg.com/dsp-image/1756536684.jpg' : 'output/urls_2_imageFiles.1.jpg',
            'http://ms.bdimg.com/dsp-image/571671431.jpg' : 'output/urls_2_imageFiles.2.jpg',
        }
        urls_2_imageFiles(files)

    if tag == 'quick_urls_2_imageFiles':
        #echo -e "http://ms.bdimg.com/dsp-image/1756536684.jpg\ttest/1756536684_1.jpg\t1\nhttp://ms.bdimg.com/dsp-image/571671431.jpg\ttest/571671431_0.jpg\t2" | python xxx.py
        quick_urls_2_imageFiles() 

    if tag == 'files_2_Ximage_AND_files_2_Ximage' or tag == 'all':
        files = list_files(path)
        X, fnames = files_2_Ximage(files, [10, 10])
        XImage_2_files(X, [100, 100], 'output/', fnames, 'files_2_Ximage_AND_files_2_Ximage____')

    if tag == 'quick_files_2_Ximage_AND_files_2_Ximage' or tag == 'all':
        files = list_files(path)
        X, fnames = quick_files_2_Ximage(files, [20, 20])
        quick_XImage_2_files(X, [100, 100], 'output/', fnames, 'quick_files_2_Ximage_AND_files_2_Ximage____')

    if tag == 'quick_transform_image' or tag == 'all':
        from limg_common import simple_extract_skin
        input_path, output_path = path, 'output/'
        quick_transform_image(input_path, output_path, func=simple_extract_skin, size=(100, 100), prefix='quick_transform_image____')

    if tag == 'template_visual_image' or tag == 'all':
        from limg_common import simple_extract_skin
        def worker(image):
            template = simple_extract_skin(image)
            image = template_visual_image(image, template, tag='bakground')
            return image
        input_path, output_path = path, 'output/'
        quick_transform_image(input_path, output_path, func=worker, size=None, prefix='template_visual_image____')
        
    if tag =='quick_mask_image' or tag == 'all':
        from lcv2_common import mask_image
        def worker(image):
            image = mask_image(
                    image, text='lichuan89', coordinates=None,
                    transparency=0.7, size=1.2, color=(255, 10, 10), thickness=2)
            return image
        input_path, output_path = path, 'output/'
        quick_transform_image(input_path, output_path, func=worker, size=None, prefix='quick_mask_image____')
        

if __name__ == "__main__":
    tag = 'quick_urls_2_imageFiles'
    test(tag)
