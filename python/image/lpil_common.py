#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""


from PIL import Image
import numpy as np
import urllib
import random
import io
import os
import sys

g_open_debug = False 


def image_2_pilimage(image):
    dst = Image.new('RGB', image.size)


def create_gif(image_list, gif_name, duration=0.1):
    import imageio
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return frames


def file_2_pilimage(fpath, is_return_mat=False):
    try:
        if fpath.find('http') != 0:
            image = Image.open(fpath)
        else:
            fpath = fpath.replace("ms.bdimg.com", "su.bcebos.com")
            fpath = fpath.replace("boscdn.bpc.baidu.com", "su.bcebos.com")
            context = urllib.urlopen(fpath).read()
            image = Image.open(io.BytesIO(context))
        if not is_return_mat:
            return image
        pix = image.load()
        width, height = image.size
        mat = np.zeros([height, width, 3])
        for x in range(width):
            for y in range(height):
                if type(pix[x, y]) == int:
                    r = g = b = pix[x, y]
                elif len(pix[x, y]) == 3:
                    r, g, b = pix[x, y]
                elif len(pix[x, y]) == 4:
                    r, g, b, c = pix[x, y]
                mat[y, x, :] = [r, g, b]
        return image, mat
    except IOError as error:
        print >> sys.stderr, error
        return None


def test(tag):
    fpath = '../../data/example/lena.jpg'
    if not os.path.exists('output/'):
        os.mkdir('output/')

    image = np.array( # 2行3列的RGB图像
        [
            [[254,   0,   5], [255,   4,   0], [  0,   0,   2], [ 98, 162, 224],],
            [[252,   2,   3], [  1, 255,   0], [  0,   3, 255], [254, 253, 249],],
        ]
        )
    image = file_2_pilimage(fpath)
    print image.size
    print '1>', np.asarray(image)
    print '2>', np.asarray(image.convert('L')) # r * 0.3 + g * 0.59 + b * 0.11
    image = np.array(image, dtype=float) 
    print '3>', (image[:, :, 0] * 0.3 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.11).astype('int') 
    #参考:  https://blog.csdn.net/hanging_gardens/article/details/79014160


if __name__ == "__main__":
    tag = 'all'
    test(tag)
