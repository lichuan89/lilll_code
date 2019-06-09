#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def show_images(images, is_show=True):
    fig = plt.figure(figsize=(10, 1), dpi=100)
    for i in range(len(images)):
        ax = fig.add_subplot(1,len(images),i + 1)
        ax.matshow(images[i], cmap='binary') 
        ax.axis('off')
    if is_show:
        plt.show()
    plt.close()

def show_2d_images(images, is_show=True):
    cols = 10 
    rows = len(images) / cols + 1
    fig = plt.figure(figsize=(cols, rows), dpi=100)

    for i in range(len(images)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.matshow(images[i], cmap='binary') 
        ax.axis('off')
    if is_show:
        plt.show()
    plt.close()


def show_array(arrs, labels, title='show array', show_fpath=None, stop=None):
    colors = ['r-', 'k-', 'b-']
    colors = (colors * (len(arrs) / len(colors) + 1))[:len(arrs)]
    linewidths = [3, 2, 0.5]
    linewidths = (linewidths * (len(arrs) / len(linewidths) + 1))[:len(arrs)]
    n = min([len(arr) for arr in arrs])
    min_v = 10000000
    max_v = -100000000
    for arr, label, color, linewidth in zip(arrs, labels, colors, linewidths):  
        min_v = min(min(arr), min_v)
        max_v = max(max(arr), max_v)
        num = len(arr)
        idxs = np.linspace(1, n, num=num)
        if stop is None:
            plt.plot(idxs, arr, color, linewidth=linewidth, label=label)
        else:
            cur_stop = int(num * stop / n) 
            plt.plot(idxs[: cur_stop + 1], arr[: cur_stop + 1], color, linewidth=linewidth, label=label)
    plt.xlabel('array_idx')
    plt.ylabel('array_value', fontsize=15)
    plt.title(title)
    plt.legend()
    plt.axis((0,n,min_v,max_v))
    plt.grid()
    if show_fpath is None:
        plt.show()
    else:
        plt.savefig(show_fpath)
    plt.close()

def show_predict_numbers(y_true, y_pred, show_fpath=None):
    # Show confusion table
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=None)  # Get confustion matrix
    # Plot the confusion table
    class_names = ['${:d}$'.format(x) for x in range(0, 10)]  # Digit class names
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Show class labels on each axis
    ax.xaxis.tick_top()
    major_ticks = range(0,10)
    minor_ticks = [x + 0.5 for x in range(0, 10)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    # Set plot labels
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.suptitle('Confusion table', y=1.03, fontsize=15)
    # Show a grid to seperate digits
    ax.grid(b=True, which=u'minor')
    # Color each grid cell according to the number classes predicted
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    # Show the number of samples in each cell
    for x in xrange(conf_matrix.shape[0]):
        for y in xrange(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            color = 'r'
            ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)
             

    if show_fpath is None:
        plt.show()
    else:
        plt.savefig(show_fpath)
    plt.close()


def show_3d_space(surfaces=[], points=[], save_path=None):
    from mpl_toolkits.mplot3d import axes3d
    ax = plt.gca(projection='3d') 
    plt.title('-', fontsize=20)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_zlabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    for xs, ys, zs in surfaces:
        ax.plot_surface(xs, ys, zs, rstride=10, cstride=10, cmap='jet')
    for xs, ys, zs, c in points:
        ax.scatter(xs, ys, zs, c=c, marker='o', label='-')
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path,dpi=200)
    else:
        plt.show()


def collect_train_data_2x1y(begin, end, func):
    gx, gy = np.meshgrid(np.linspace(begin, end, 1000), np.linspace(begin, end, 1000))
    gz = func(gx, gy)

    np.random.seed(seed=1)
    xy = np.random.uniform(begin, end, (300, 2))
    x = xy[:, 0]
    y = xy[:, 1]
    z = func(x, y)
    noise_variance = 0.5 
    noise = np.random.randn(x.shape[0]) * noise_variance
    z = z + noise
    return (gx, gy, gz), (x, y, z)


def test(tag):
    if tag == 'show_3d_space' or tag == 'all':
        def f(x, y):
            return 5 * (x**2) - 10 * y

        (gx, gy, gz), (x, y, z) = collect_train_data_2x1y(-10, 10, f)
        show_3d_space([(gx, gy, gz)], [(x, y, z, 'r')])
    
    if tag == 'show_array' or tag == 'all':
        show_array(arrs=[(12.0, 1.5, 1.0, 0.3), (12.1, 1.6)], labels=['one', 'two'], title='show array', show_fpath='data/test.jpg', stop=2)
        costs_arr = ([2.341391170322213, 2.3315095456917336, 2.3223760257551973, 2.3138531892586922, 2.3058223288076096, 2.2981810718025595, 2.2908408701110003, 2.2837246897755978, 2.2767650241359929, 2.2699022212056716, 2.2630830800965982, 2.2562596815253815, 2.249388433889989, 2.2424293232533206, 2.2353453517551238, 2.228102138955022, 2.2206676498457707, 2.2130120066429813, 2.2051073426886894, 2.1969276681542089], [2.3386467421202086, 2.3288486991762039, 2.3197256115939715, 2.3111475313496292, 2.3030028249264745, 2.2951954487046042, 2.2876423606913252, 2.2802712388193074, 2.2730185399602978, 2.2658278806841077, 2.2586487129624464, 2.2514352716029711, 2.2441457701902516, 2.2367418159872292, 2.2291880041614327, 2.2214516421064703, 2.2135025500245464, 2.2053128882071, 2.1968569771161377, 2.1881111039527377])
        labels = ('cost full training set', 'cost validation set')
        show_array(costs_arr, labels, title='Decrease of cost over backprop iteration', show_fpath='data/test2.jpg', stop=None)
        #show_predict_numbers(np.array([1, 0, 1, 1]), np.array([1, 1, 1, 1]), show_fpath='data/test.jpg')

if __name__ == "__main__":
    tag = 'show_3d_space'
    test(tag) 
