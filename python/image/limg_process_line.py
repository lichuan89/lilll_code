#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2017/01/01  
    @note   实现python多进程处理输入流的函数
"""

import sys
import lcv2_common
import limg_common 
from limgs_common import *

def image_light(image, args=[1.0, 0]):
    #image_cmd: #image_light____l____0____2.3____20
    a = float(args[0]) if len(args) > 0 else 1
    b = float(args[1]) if len(args) > 1 else 0
    return lcv2_common.light_image(image, a=a, b=b)


def image_rotate(image, args=[60]):
    #image_cmd: #image_rotate____r____0____60
    angle = float(args[0]) if len(args) > 0 else 1
    return lcv2_common.rotate_image(image, angle=angle, center=None, scale=1.0) 

def image_expand(image, args=[]):
    #image_cmd: #image_expand____r____0
    return lcv2_common.random_expand_image(image)

def image_gray(image, args=[]):
    # image_cmd: #image_gray____g____0
    return limg_common.rgb_2_gray(image)

def image_hist(image, args=[0, 250]):
    # image_cmd: #image_hist____h____0
    min_v = float(args[0]) if len(args) > 0 else 0
    max_v = float(args[1]) if len(args) > 1 else 255
    image = limg_common.rgb_2_gray(image)
    return limg_common.equalization_gray_hist(image, min_v, max_v)

def image_sobel(image, args=[]):
    # image_cmd: #image_sobel____s____0
    return limg_common.sobel_gray_image(image)

def image_skin(image, args=[]):
    # image_cmd: #image_skin____skin____0
    return limg_common.simple_extract_skin(image) 

def image_ycrcb(image, args=[]):
    # image_cmd: #image_ycrcb____y____0
    return limg_common.rgb_2_ycrcb(image)




def process_path(tags=[]):
    func = tags[0]
    output_dir = "tmp_imgs/" if len(tags) <= 1 else tags[1]
    idx = 0 if len(tags) <= 2 else int(tags[2])
    args = None if len(tags) <= 3 else tags[3:] 

    cur_path = "../../go/web_cmd/static/temp/"
    output_path = cur_path + output_dir
    if func == 'crawl':
        quick_urls_2_imageFiles()
    else:
        imgs = {}
        
        for line in sys.stdin:
            if line[-1] == '\n':
                line = line[:-1]
            arr = line.split('\t')
            if idx < len(arr):
                f = arr[idx].split('/')[-1]
                
                imgs[cur_path + arr[idx]] = '%s/%s\t%s' % (output_dir, f, line)
        files = quick_transform_image(imgs, output_path, func=eval(func), size=(None, 100), prefix="", args=args)
        for img, line in imgs.items():
            print line 

if __name__ == "__main__":
    func_arg = sys.argv[1]
    arr = func_arg.split('____')
    process_path(arr)
