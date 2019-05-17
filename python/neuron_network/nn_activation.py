#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import numpy as np
import copy
from functools import reduce


# 参考:
#   https://blog.csdn.net/cckchina/article/details/79915181
#   https://zhuanlan.zhihu.com/p/37740860

def logistic(z):
    o = copy.deepcopy(z)
    o[o > 0] = 1 / (1 + np.exp(-o[o > 0]))
    o[o <= 0] = np.exp(o[o <= 0]) / (1 + np.exp(o[o <= 0]))
    return o 

def logistic_deriv(y):
    return np.multiply(y, (1 - y)) 

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def softmax_deriv(y):
    return np.multiply(y, (1 - y)) 


def relu(z):
    return np.maximum(z, 0)


def relu_deriv(y):
    o = np.ones(y.shape)
    o[y < 0] = 0 
    return o


def tanh(z):
    return np.tanh(z)


def tanh_deriv(y):
    return 1 - y ** 2


def arctan(z):
    return np.arctan(z)


def arctan_deriv(z):
    return 1 / (1 + z ** 2)

def crossEntropy_cost(y, t): 
    """ 
    交叉熵损失函数
    """
    o = copy.deepcopy(y)
    o[o < 1e-10] = 1e-10
    o[o > 0.99] = 1 - 1e-10 
    return - np.multiply(t, np.log(o)).sum() / o.shape[0]

def crossEntropy_cost_deriv(y, t): 
    """ 
    交叉熵损失函数对预测值y的梯度
    """
    return (y - t) / y / (1 - y) / y.shape[0]


def batch_norm(
            x,
            is_train=True,
            focus_axis=(0, 1),
            alpha = 1,
            beta = 0,
            moving_mean=None,
            moving_var=None,
            moving_decay=0.997,
            epsilon=0.00001
        ):
    """
    x是n个样本,每个样本有多维特征
    mean_axis表示在这些维度上求均值和方差, 一般保留channel维度 
    """
    n = x.shape[0]
    mean = np.mean(x, axis=focus_axis, keepdims=True)
    var = np.var(x, axis=focus_axis, keepdims=True)
    #dim_num = reduce(lambda x, y: x * y, [x.shape[i] for i in focus_axis])
    #var_v2 = np.sum((x - mean)**2, axis=focus_axis, keepdims=True) / dim_num 

    #if n > 1:
    #    var = n / (n - 1.0) * var

    if is_train and  moving_mean is not None and moving_var is not None:
        # 积累均值和方差
        moving_mean = moving_decay * moving_mean + (1 - moving_decay) * mean
        moving_var = moving_decay * moving_var + (1 - moving_decay) * var
    elif is_train:
        # 初始
        moving_mean = mean
        moving_var = var 
    else:
        # 预测
        pass

    normed_x = (x - moving_mean) / np.sqrt(moving_var + epsilon)
    y = normed_x * alpha + beta
    return y, normed_x, moving_mean, moving_var, mean, var
   

def batch_norm_deriv(x, y, grad, focus_axis, alpha, beta, normed_x, mean, var, epsilon=0.00001):
    a_gradient = np.sum(grad * normed_x, axis=focus_axis, keepdims=True) 
    b_gradient = np.sum(grad, axis=focus_axis, keepdims=True) 
    normed_x_gradient = grad * alpha
    dim_num = reduce(lambda x, y: x * y, [x.shape[i] for i in focus_axis])
    var_gradient = np.sum(-1.0 / 2 * normed_x_gradient * (x - mean)/(var + epsilon)**(3.0/2), axis=focus_axis, keepdims=True)
    mean_gradinet = np.sum(-1/np.sqrt(var + epsilon)*normed_x_gradient, axis=focus_axis, keepdims=True)
    g1 = normed_x_gradient/np.sqrt(var+epsilon)
    g2 = 2*(x-mean)*var_gradient/dim_num
    g3 = mean_gradinet/dim_num
    x_gradient = normed_x_gradient / np.sqrt(var+epsilon)+2*(x-mean)*var_gradient/dim_num+mean_gradinet/dim_num
    return x_gradient, a_gradient, b_gradient
