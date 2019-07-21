#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2017/01/01  
    @note   实现python多进程处理输入流的函数
"""

import sys
sys.path.append("../../python/image/")
import lcv2_common
import limg_common 
from limgs_common import *


def limg(us, tags=[]):
    ## process image. cmd: limg__1i__otsu__path1__.bmp
    func_name = tags[0] if len(tags) > 0 else 'image_otsu' 
    path = tags[1] if len(tags) > 1 else 'tmp.crawl'
    suffix = tags[2] if len(tags) > 2 else '.bmp' 
    arr = tags[3: ] 
    if path.find('/') == -1: 
        path = './static/temp/%s/' % path 
    img = file_2_cv2image(us.encode('utf8', 'ignore'))
    if img is None:
        return None
    img = cv2image_2_image(img)
    f = eval(func_name)
    img, obj = f(img, arr)
    if img is None:
        return unicode(obj)
    fname = us.split('/')[-1]
    output_fpath = '%s/%s%s' % (path, fname, suffix)
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except IOError as error:
        print >> sys.stderr, error  
    cv2image_2_file(image_2_cv2image(img), output_fpath)
    if obj is not None:
        return '%s\t%s' % (output_fpath, obj)
    return output_fpath 
    

def gray(image, args=[]):
    ## gray image. cmd: limg__0i__gray__path1__.bmp
    return limg_common.rgb_2_gray(image), None

def resize(image, tags=[]):
    ## resize image. cmd: limg__0i__resize__path1__.2.bmp__200__100, limg__0i__resize__path1__.2.bmp____100__sobel
    width = int(tags[0]) if len(tags) > 0 else 100
    height = int(tags[1]) if len(tags) > 1 else 100
    pattern = int(tags[2]) if len(tags) > 2 else 'random'
    if width == '':
        width = None
    if height == '':
        height = None
    return rgb_resize(image, (height, width), pattern=pattern), None

def otsu(image, args=[]):
    ## otsu image. cmd: limg__0i__otsu__path1__.4.bmp 
    import cv2
    image = image_2_cv2image(image)
    image = np.array(image, dtype='uint8')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return otsu_image, None

def muzzy(image, args=[100]):
    ## calc muzzy of image. cmd: limg__0i__muzzy__path1__.3.bmp__200, limg__0i__muzzy__path1__.3.bmp 
    import cv2
    width = int(args[0])
    #image = limg_common.rgb_2_gray(image)
    image = np.array(image, dtype='uint8')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    q, cache = lcv2_common.check_gray_muzzy(image, width)
    return cache['check_image'], q

def light(image, args=[1.0, 0]):
    ## light image. cmd: limg__0i__muzzy__path1__.3.bmp__2.3____20
    a = float(args[0]) if len(args) > 0 else 1
    b = float(args[1]) if len(args) > 1 else 0
    return lcv2_common.light_image(image, a=a, b=b), None


def rotate(image, args=[60]):
    ## rotate image. cmd: limg__0i__rotate__path1__.3.bmp__60
    angle = float(args[0]) if len(args) > 0 else 1
    return lcv2_common.rotate_image(image, angle=angle, center=None, scale=1.0), None 

def expand(image, args=[]):
    ## random expand image. cmd: limg__0i__expand__path1__.3.bmp
    return lcv2_common.random_expand_image(image), None

def hist(image, args=[0, 250]):
    ## produce equalization image. cmd: limg__0i__hist__path1__.3.bmp
    # image_cmd: #image_hist____h____0
    min_v = float(args[0]) if len(args) > 0 else 0
    max_v = float(args[1]) if len(args) > 1 else 255
    image = limg_common.rgb_2_gray(image)
    return limg_common.equalization_gray_hist(image, min_v, max_v), None

def sobel(image, args=[]):
    ## sobel image. cmd: limg__0i__sobel__path1__.3.bmp
    return limg_common.sobel_gray_image(image), None

def skin(image, args=[]):
    ## extract skin from image. cmd: limg__0i__skin__path1__.3.bmp
    return limg_common.simple_extract_skin(image), None 

def ycrcb(image, args=[]):
    ## change image from rgb to ycrcb. cmd: limg__0i__ycrcb__path1__.3.bmp
    return limg_common.rgb_2_ycrcb(image), None


if __name__ == "__main__":
    s = './static/temp/path1//338456730887822400270513569953190097124.bmp'
    print limg(s, 'limg__0i__gray__path1__.1.bmp'.split('__')[2:]) 
    print limg(s, 'limg__0i__resize__path1__.2.bmp__200__100'.split('__')[2:]) 
    print limg(s, 'limg__0i__muzzy__path1__.3.bmp__100'.split('__')[2:]) 
    print limg(s, 'limg__0i__otsu__path1__.4.bmp'.split('__')[2:]) 
    print limg(s, 'limg__0i__light__path1__.5.bmp__2.3__20'.split('__')[2:]) 
    print limg(s, 'limg__0i__rotate__path1__.6.bmp__60'.split('__')[2:]) 
    print limg(s, 'limg__0i__expand__path1__.7.bmp'.split('__')[2:]) 
    print limg(s, 'limg__0i__hist__path1__.8.bmp'.split('__')[2:]) 
    print limg(s, 'limg__0i__sobel__path1__.9.bmp'.split('__')[2:]) 
    print limg(s, 'limg__0i__skin__path1__.10.bmp'.split('__')[2:]) 
    print limg(s, 'limg__0i__ycrcb__path1__.11.bmp'.split('__')[2:]) 
