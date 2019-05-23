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
#import collections
from lcommon import str_2_json, json_2_str, file_2_str, str_2_file, clear_dir, list_files 
from nn_activation import logistic
from nn_activation import logistic_deriv
from nn_activation import softmax
from nn_activation import softmax_deriv
from nn_activation import relu 
from nn_activation import relu_deriv
from nn_activation import tanh 
from nn_activation import tanh_deriv 
from nn_activation import arctan 
from nn_activation import arctan_deriv 
from nn_activation import crossEntropy_cost
from nn_activation import crossEntropy_cost_deriv 
from nn_matrix import cross_validation_split
from nn_matrix import matrix_2_string 
from nn_matrix import string_2_matrix 
from nn_matrix import accuracy_score 
from nn_matrix import recall_score 
from nn_matrix import confusion_matrix 
from cnn_common import conv
from cnn_common import conv_gradient
from cnn_common import maxpool
from cnn_common import maxpool_gradient
from cnn_common import avgpool
from cnn_common import avgpool_gradient
from lcommon import log
from lcommon import md5 
from lcmd import muti_process

g_open_test = True # 程序处于测试状态, 固定随机数字
g_open_debug = True # 用于调试, 打印更多日志
g_debug_cache = {}  # 用于调试, 缓存数据


class Layer(object):
    """
    神经网络的层
    """
    
    def get_params(self):
        """
        参数
        """
        
        return []
    
    def get_params_grad(self, X, output_grad):
        """
        损失函数在参数上的梯度。其中output_grad为损失函数在该层输出的梯度。
        """
        return []
    
    def get_output(self, X):
        """
        神经网络这一层的输出
        """
        pass
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """
        损失函数在该层输入的梯度。output_grad为损失函数在该层输出的梯度。
        """
        pass


class LinearLayer(Layer):
    """
    线性回归
    """
    
    def __init__(self, n_in=None, n_out=None, ws_bs=None):
        # n_out个神经元，每个神经元接收n_in个输入特征
        if ws_bs is not None:
            self.from_string(ws_bs)
            return 
        self.W = np.random.randn(n_in, n_out) * 0.1 # 标准正态分布

        if g_open_test: 
            fpath = './tmp.%sx%s.randn.txt' % (n_in, n_out)
            if not os.path.exists(fpath):
                str_2_file(matrix_2_string(self.W), fpath)
            else:
                self.W = string_2_matrix(file_2_str(fpath))

 
        self.b = np.zeros(n_out)
 
 
    def get_params(self):
        #return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
        #                       np.nditer(self.b, op_flags=['readwrite']))
        return [self.W, self.b]
   
    def get_output(self, X):
        # 输出矩阵：sample_num * n_out
        return X.dot(self.W) + self.b

        
    def get_params_grad(self, X, output_grad):
        # 各个样本的对应梯度之和
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        #return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]
        return [JW, Jb]
    
    def get_input_grad(self, Y, output_grad):
        return output_grad.dot(self.W.T)

    def from_string(self, ws_bs):
        self.W, self.b = string_2_matrix(ws_bs[0]), string_2_matrix(ws_bs[1])


    def to_string(self):
        return [matrix_2_string(self.W), matrix_2_string(self.b)]


'''
class BatchNormLayer(layer):
    """
    批标准化
    """
    def __init__(self, n_in=None, focus_axis=None, ws_bs=None, moving_decay=0.997, epsilon=0.00001):
        if ws_bs is not None:
            self.from_string(ws_bs)
            return
        self.insize = n_in
        self.focus_axis = focus_axis 
        self.alpha = 1
        self.beta = 0
        self.moving_mean = None,
        self.moving_var = None,
        self.moving_decay = moving_decay,
        self.epsilon = epsilon

        self.mid = None

    def get_params(self):
        return itertools.chain(np.nditer(self.alpha, op_flags=['readwrite']),
                               np.nditer(self.beta, op_flags=['readwrite']))
    
    def get_output(self, X, is_train=True):
        input_size = [X.shape[0]] + list(self.insize)
        X = X.reshape(input_size)

        y, normed_x, self.moving_mean, self.moving_var, mean, var = batch_norm(
                X,
                is_train,
                focus_axis=self.focus_axis,
                alpha=self.alpha,
                beta=self.beta,
                moving_mean=self.moving_mean,
                moving_var=self.moving_var,
                moving_decay=self.moving_decay,
                epsilon=self.epsilon
            ) 
        self.mid = X, normed_x, mean, var
        y = y.reshape((y.shape[0], -1))
        return y
 
    def get_params_grad(self, X, output_grad):
        Ja, Jb = self.mid
        return [g for g in itertools.chain(np.nditer(Ja), np.nditer(Jb))]


    def get_input_grad(self, Y, output_grad):
        X, normed_x, mean, var = self.mid
        x_gradient, a_gradient, b_gradient = batch_norm_deriv(X, Y, output_grad, self.focus_axis, self.alpha, self.beta, normed_x, mean, var, self.epsilon)
        Jx = x_gradient.reshape((x_gradient.shape[0], -1))
        self.mid = a_gradient, b_gradient
        return Jx 


    def __init__(self, n_in=None, focus_axis=None, ws_bs=None, moving_decay=0.997, epsilon=0.00001):
        self.W = np.fromstring(base64.b64decode(ws_bs[0])).reshape(ws_bs[2])
        self.b = np.fromstring(base64.b64decode(ws_bs[1])).reshape(ws_bs[3])
        self.insize = ws_bs[4]
        self.focus_axis = ws_bs[5]

    def to_string(self):
        return [
                base64.b64encode(self.alpha.tostring()),
                base64.b64encode(self.beta.tostring()),
                base64.b64encode(self.moving_mean.tostring()),
                base64.b64encode(self.moving_var.tostring()),
                list(self.insize),
                list(self.focus_axis)
                self.moving_decay,
                self.epsilon,
            ]
'''

def forward_step(input_samples, layers):
    """
    前向传播，取输入和每一层输出
    """
    activations = [input_samples] 
    X = input_samples
    for layer in layers:
        log('begin to forward_step:', str(layer))
        Y = layer.get_output(X)  
        activations.append(Y)   
        X = activations[-1]  
        log('finish to forward_step')
    return activations  



class LogisticLayer(Layer):
    def get_output(self, X):
        return logistic(X)
    
    def get_input_grad(self, Y, output_grad):
        return np.multiply(logistic_deriv(Y), output_grad)

    def to_string(self):
        return ['LogisticLayer']

class SoftmaxLayer(Layer):
    def get_output(self, X):
        return softmax(X)
    
    def get_input_grad(self, Y, output_grad):
        return np.multiply(softmax_deriv(Y), output_grad)
   
    def to_string(self):
        return ['SoftmaxLayer']


class ReluLayer(Layer):
    def get_output(self, X):
        return relu(X)
    
    def get_input_grad(self, Y, output_grad):
        return np.multiply(relu_deriv(Y), output_grad)
   
    def to_string(self):
        return ['Relu']



class MaxpoolLayer(Layer):
    def __init__(self, n_in=None, ksize=(2, 2), stride=2, args=None):
        if args is not None:
            self.from_string(args)
            return
        self.insize = n_in
        self.ksize = ksize
        self.stride = stride
        self.mid = None

    def get_output_size(self):
        row, col, channel = self.insize
        return int(math.ceil(1.0 * row / self.stride)), int(math.ceil(1.0 * col / self.stride)), channel
 
    def get_output(self, X):
        input_size = [X.shape[0]] + list(self.insize)
        X = X.reshape(input_size)
        y, self.mid = maxpool(X, self.ksize, self.stride)
        y = y.reshape((y.shape[0], -1))
        return y
    
    def get_input_grad(self, Y, output_grad):
        output_size = [Y.shape[0]] + list(self.get_output_size())
        output_grad = output_grad.reshape(output_size)
        dX = maxpool_gradient(output_grad, self.mid, self.ksize, self.stride)
        dX = dX.reshape(dX.shape[0], -1) 
        return dX 
   
    def from_string(self, args):
        self.insize, self.ksize, self.stride, _ = args
        self.mid = None

    def to_string(self):
        return [
            list(self.insize),
            list(self.ksize),
            self.stride,
            'MaxpoolLayer' 
        ]



class AvgpoolLayer(Layer):
    def __init__(self, n_in=None, ksize=(2, 2), stride=2, args=None):
        if args is not None:
            self.from_string(args)
            return
        self.insize = n_in
        self.ksize = ksize
        self.stride = stride
        self.mid = None

    def get_output_size(self):
        row, col, channel = self.insize
        return int(math.ceil(1.0 * row / self.stride)), int(math.ceil(1.0 * col / self.stride)), channel
 
    def get_output(self, X):
        input_size = [X.shape[0]] + list(self.insize)
        X = X.reshape(input_size)
        y, self.mid = avgpool(X, self.ksize, self.stride)
        y = y.reshape((y.shape[0], -1))
        return y
    
    def get_input_grad(self, Y, output_grad):
        output_size = [Y.shape[0]] + list(self.get_output_size())
        output_grad = output_grad.reshape(output_size)
        dX = avgpool_gradient(output_grad, self.mid, self.ksize, self.stride)
        dX = dX.reshape(dX.shape[0], -1) 
        return dX 
   
    def from_string(self, args):
        self.insize, self.ksize, self.stride, _ = args
        self.mid = None

    def to_string(self):
        return [
            list(self.insize),
            list(self.ksize),
            self.stride,
            'AvgpoolLayer' 
        ]


class ConvLayer(Layer):
    """
    卷积
    """
    
    def __init__(self, n_in=None, ksize=None, ws_bs=None):
        if ws_bs is not None:
            self.from_string(ws_bs)
            return
        self.insize = n_in
        self.ksize = ksize
        row, col, channel = n_in
        conv_row, conv_col, channel, conv_channel = ksize

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, n_in) / conv_channel)
        self.W = np.random.standard_normal((conv_row, conv_col, channel, conv_channel)) / weights_scale
        self.b = np.random.standard_normal(conv_channel) / weights_scale
        if g_open_debug:
            print >> sys.stderr, 'W.shape:%s, b.shape:%s' % (self.W.shape, self.b.shape) 


    def get_params(self):
        #return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
        #                       np.nditer(self.b, op_flags=['readwrite']))
        return [self.W, self.b]
 
    def get_output(self, X):
        input_size = [X.shape[0]] + list(self.insize)
        X = X.reshape(input_size)
        y, x_cols = conv(X, self.W, self.b, stride=1)
        y = y.reshape((y.shape[0], -1))
        self.mid = X
        return y
 
    def get_params_grad(self, X, output_grad):
        input_size = [X.shape[0]] + list(self.insize)
        X = X.reshape(input_size)
        output_size = [output_grad.shape[0]] + list(self.insize[: 2]) + [self.ksize[-1]]
        output_grad = output_grad.reshape(output_size)
        Jparam, Jx = conv_gradient(X, output_grad, self.W, self.b, stride=1)
        return Jparam    

    def get_input_grad(self, Y, output_grad):
        X = self.mid
        input_size = [Y.shape[0]] + list(self.insize)
        X = X.reshape(input_size)
        output_size = [Y.shape[0]] + list(self.insize[: 2]) + [self.ksize[-1]]
        output_grad = output_grad.reshape(output_size)
        Jparam, Jx = conv_gradient(X, output_grad, self.W, self.b, stride=1)
        Jx = Jx.reshape((Jx.shape[0], -1))
        return Jx 

    def from_string(self, ws_bs):
        self.W = np.fromstring(base64.b64decode(ws_bs[0])).reshape(ws_bs[2])
        self.b = np.fromstring(base64.b64decode(ws_bs[1])).reshape(ws_bs[3])
        self.insize = ws_bs[4]
        self.ksize = ws_bs[5]

    def to_string(self):
        return [
                base64.b64encode(self.W.tostring()),
                base64.b64encode(self.b.tostring()),
                list(self.W.shape),
                list(self.b.shape),
                list(self.insize),
                list(self.ksize)
            ]

def forward_step(input_samples, layers):
    """
    前向传播，取输入和每一层输出
    """
    activations = [input_samples] 
    X = input_samples
    for layer in layers:
        log('begin to forward_step:', str(layer))
        Y = layer.get_output(X)  
        activations.append(Y)   
        X = activations[-1]  
        log('finish to forward_step')
    return activations  

 
    
def backward_step(activations, targets, layers, cost_grad_func):
    """
    后向传播，取损失函数在每一层的梯度
    """
    #param_grads = collections.deque()  
    param_grads = [] 
    output_grad = None
    for layer in reversed(layers):
        log('begin to backward_step:', str(layer))   
        Y = activations.pop()
        # 交叉熵损失函数, 合并链式梯度公式 
        if output_grad is None \
                and cost_grad_func == crossEntropy_cost_deriv \
                and type(layer) == SoftmaxLayer:
            input_grad = (Y - targets) / Y.shape[0]
        else:
            if output_grad is None:
                output_grad = cost_grad_func(Y, targets)  
            input_grad = layer.get_input_grad(Y, output_grad)
        log('begin to calc cost') 
        X = activations[-1]
        grads = layer.get_params_grad(X, output_grad)
        #param_grads.appendleft(grads)
        param_grads.append(grads)
        output_grad = input_grad
        log('finish to backward_step')
    param_grads.reverse()
    return param_grads
    #return list(param_grads)

class NeuronNetwork(object):
    def __init__(
            self,
            input_feature_num=None, # 输入数据集样本数 
            layer_neuron_nums=None, # 每一层的输出节点数 
            layer_active_funcs=None, # 每一层的激活函数
            cost_func=None, # 损失函数
            cost_grad_func=None, # 损失函数对应的梯度函数
            create_string=None
        ):
        if create_string is not None:
            self.from_string(create_string)
            return
        # 构建神经网络
        self.layers = []
        for (neuron_num, active_func) in zip(layer_neuron_nums, layer_active_funcs):
            if active_func == MaxpoolLayer or active_func == AvgpoolLayer:
                # 构建卷积层: 卷积网络 + 池化
                insize, ksize, poolsize, poolstride = neuron_num 
                self.layers.append(ConvLayer(insize, ksize))
                outsize = insize[0], insize[1], ksize[-1]
                pooler = active_func(outsize ,poolsize, poolstride)
                self.layers.append(pooler)
                size = pooler.get_output_size()
                input_feature_num = size[0] * size[1] * size[2] 
                #row, col, channel = outsize
                #input_feature_num = int(math.ceil(1.0 * row / poolstride)) * int(math.ceil(1.0 * col / poolstride)) * channel
            else: 
                # 构建每一层: 线性网络 + 激活函数
                self.layers.append(LinearLayer(input_feature_num, neuron_num))
                self.layers.append(active_func())
                input_feature_num = neuron_num
        self.cost_func = cost_func
        self.cost_grad_func = cost_grad_func
   
    def from_string(self, string):
        classes = [SoftmaxLayer, LogisticLayer, ReluLayer, MaxpoolLayer, AvgpoolLayer, crossEntropy_cost, crossEntropy_cost_deriv]
        classes = dict([active_func.__name__, active_func] for active_func in classes)
    
        self.layers = []
        objs = str_2_json(string)
        if objs is None:
            print >> sys.stderr, 'failed to load string'
            return False
        for layer_string in objs['layers']:
            if layer_string[-1] in ['MaxpoolLayer', 'AvgpoolLayer']:
                ws, bs, wi, bi, insize, ksize, insize2, ksize2, stride2, active_func_name = layer_string
                self.layers.append(ConvLayer(ws_bs=(ws, bs, wi, bi, insize, ksize)))
                self.layers.append(classes[active_func_name](args=(insize2, ksize2, stride2, active_func_name)))
            else:
                ws, bs, active_func_name = layer_string
                self.layers.append(LinearLayer(ws_bs=(ws, bs))) 
                self.layers.append(classes[active_func_name]())
        self.cost_func = classes[objs['cost_func']]
        self.cost_grad_func = classes[objs['cost_grad_func']]
        return True

    def to_string(self):
        obj = {
            'layers': [],
            'cost_func': self.cost_func.__name__,
            'cost_grad_func': self.cost_grad_func.__name__,
        } 
        for i in range(0, len(self.layers), 2):
            active_func_name = self.layers[i + 1].__class__.__name__
            if active_func_name in ['MaxpoolLayer', 'AvgpoolLayer']:
                ws, bs, wi, bi, insize, ksize = self.layers[i].to_string()
                insize2, ksize2, stride2, active = self.layers[i + 1].to_string()
                obj['layers'].append([ws, bs, wi, bi, insize, ksize, insize2, ksize2, stride2, active])
            else:
                ws, bs = self.layers[i].to_string()
                active, = self.layers[i + 1].to_string()
                obj['layers'].append([ws, bs, active_func_name])
        return json_2_str(obj)
 
    def test_accuracy(self, X_test, T_test):
        from sklearn import metrics
        y_test = self.predict(X_test)
        y_true = np.argmax(T_test, axis=1)
        y_pred = np.argmax(y_test, axis=1)

        #test_accuracy = metrics.accuracy_score(y_true, y_pred) 
        diff = y_true - y_pred
        accuracy_score = float(len(diff[diff == 0])) / len(diff) # 准确率, 代替sklearn.accuracy_score
         
        return accuracy_score 
        #return test_accuracy
    
    def predict(self, X):
        activations = forward_step(X, self.layers) 
        return activations[-1]
    
 
    def train_once(self, X, T, learning_rate):
        activations = forward_step(X, self.layers) # 计算每一层的输出 
        cost = self.cost_func(activations[-1], T) # 计算错误率cost
        if learning_rate is None: 
            return cost
        param_grads = backward_step(activations, T, self.layers, self.cost_grad_func) # 计算参数的梯度
        for layer, layer_backprop_grads in zip(self.layers, param_grads):
            #for param, grad in itertools.izip(layer.get_params(), layer_backprop_grads):
            for param, grad in zip(layer.get_params(), layer_backprop_grads):
                param -= learning_rate * grad # 更新参数
        return cost
        
    def train_random_grad_desc(
            self,
            X_train,
            T_train,
            X_validation,
            T_validation,
            batch_size,
            max_nb_of_iterations, 
            learning_rate
            ):
        if g_open_test:
            log('train_random_grad_desc param is:', \
                    md5(matrix_2_string(X_train)), md5(matrix_2_string(T_train)), \
                    md5(matrix_2_string(X_validation)), md5(matrix_2_string(T_validation)), \
                    batch_size, max_nb_of_iterations, learning_rate)

        nb_of_batches = X_train.shape[0] / batch_size  # 批处理次数
        # 从训练集中分批抽取(X, Y) 
        XT_batches = zip(
            np.array_split(X_train, nb_of_batches, axis=0),  # X samples
            np.array_split(T_train, nb_of_batches, axis=0))  # Y targets
       
        minibatch_costs = []
        training_costs = []
        validation_costs = []

        i = 0
        # 每一轮迭代
        for iteration in range(max_nb_of_iterations):
            # 每一轮迭代中的批量训练
            for X, T in XT_batches:  # For each minibatch sub-iteration
                cost = self.train_once(X, T, learning_rate)
                minibatch_costs.append(cost)
                log('finish to do minibatch train. i:', len(minibatch_costs), ', cost:', cost) 
            cost = self.train_once(X_validation, T_validation, learning_rate=None)
            validation_costs.append(cost)
            if g_open_debug:
                cost = self.train_once(X_train, T_train, learning_rate=None)
                training_costs.append(cost)
                activations = forward_step(X_validation, self.layers)
                Y_validation = activations[-1]
                g_debug_cache.setdefault('Y_validation', [])
                g_debug_cache['Y_validation'].append(Y_validation)
            accuracy = self.test_accuracy(X_validation, T_validation)
            log('finish to train once. i:', i, ', cost:', cost, ', accuracy:', accuracy)
            if len(validation_costs) > 3:
                if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                    break
            i += 1
 
        nb_of_iterations = iteration + 1
        costs_vec = [minibatch_costs, training_costs, validation_costs]
        return (validation_costs[-1], nb_of_iterations, costs_vec)
    
    def train_grad_desc(
            self,
            X_train,
            T_train,
            X_validation,
            T_validation,
            max_nb_of_iterations, 
            learning_rate
            ):
        training_costs = []
        validation_costs = []

        i = 0
        for iteration in range(max_nb_of_iterations):
            cost = self.train_once(X_train, T_train, learning_rate)
            training_costs.append(cost)
            cost = self.train_once(X_validation, T_validation, learning_rate=None)
            validation_costs.append(cost)
            if g_open_debug:
                activations = forward_step(X_validation, self.layers)
                Y_validation = activations[-1]
                g_debug_cache.setdefault('Y_validation', [])
                g_debug_cache['Y_validation'].append(Y_validation)
            i += 1
            accuracy = self.test_accuracy(X_validation, T_validation)
            log('finish to train once. i:', i, ', cost:', cost, ', accuracy:', accuracy)
            if len(validation_costs) > 3:
                if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3] and accuracy > 0.95:
                    break
    
        nb_of_iterations = iteration + 1
        costs_vec = [training_costs, validation_costs]
        return (validation_costs[-1], nb_of_iterations, costs_vec)


def images_2_feature(X_T_list, arg=[(50, 40), ['pixel'], 0]):
    #fea_types = ['pixel'] # 可以取值['pixel', 'hist', 'rgb_hist']
    feature_size, fea_types, expand_num = arg 
    gray_size = feature_size[: 2] 

    from limg_common import rgb_2_hist, rgb_2_gray, gray_2_hist, rgb_resize, equalization_gray_hist
    from lcv2_common import random_expand_image 
    output = []


    for x, t in X_T_list:
        images = [x] # 放入原图
        if t != 0:
            for j in range(expand_num): 
                images.append(random_expand_image(x)) # 随机调节图像 
        for image in images:
            fea = []
            if 'pixel' in fea_types:
                resize_image = rgb_resize(image, gray_size)
                if feature_size[2] == 1:
                    gray_X = rgb_2_gray(resize_image) # 产出灰度像素
                    gray_X = equalization_gray_hist(gray_X, 0, 16)
                    #gray_X = 13 * (gray_X) / 255.0
                    fea.append(gray_X.reshape(-1))
                else:
                    fea.append(resize_image.reshape(-1))
            if 'rgb_hist' in fea_types: 
                rgb_hist = rgb_2_hist(image, l=16) # 产出rgb直方图
                fea.append(rgb_hist.reshape(-1))
            if 'hist' in fea_types:
                hist = gray_2_hist(rgb_2_gray(image)) # 产出灰度直方图
                fea.append(hist.reshape(-1))
            fea = np.hstack(fea)
            output.append([fea, t])

    if g_open_debug and 'pixel' in fea_types: 
        from limgs_common import quick_XImage_2_files
        path = 'data/tmp/'
        clear_dir(path, is_del=False)
        X = np.array([fea_t[0][: gray_size[0] * gray_size[1]].reshape(gray_size[0], gray_size[1]) for fea_t in output], dtype='uint8')
        X = np.array([fea_t[0][: feature_size[0] * feature_size[1] * feature_size[2]].reshape(feature_size[0], feature_size[1], feature_size[2]) for fea_t in output], dtype='uint8')
        T = ['%s_%d.bmp' % (output[i][1], i) for i in range(len(output))]
        quick_XImage_2_files(X[: 10], None, path, T[: 10])
        log('finish to save sample.') 
    log('finish to calc images_2_feature. image_num:', len(output)) 
    return output
                     


def train_test_split(X, T):
    from sklearn import cross_validation
    X_train, X_test, T_train, T_test = cross_validation.train_test_split(
            X, T, test_size=0.4)
    X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
            X_test, T_test, test_size=0.5)
        
    log('finish to split train data. X_train.shape:', X_train.shape, ', X_validation.shape:', X_validation.shape, ', X_test.shape:', X_test.shape)
    
    return X_train, T_train, X_validation, T_validation, X_test, T_test



def collect_images_feature(path, shape=(100, 100, 3), feature_size=(40, 40, 1), fea_types=['pixel'], expand_num=0):
    """
    从文件夹中读入图片, path为文件夹, 文件名如 xxx_n.jpg, 其中n是数字, 表示类别.
    读入的图片标准化为尺寸shape, expand_num表示n非0的图片做些旋转、调节亮度等转换产出的图片数,用以提升模型效果的稳定性.
    返回图片矩阵X尺寸为(个数 x 行数x 列数 x 通道数), T是类别矩阵. 
    """
    from sklearn import datasets, cross_validation, metrics
    from limgs_common import quick_files_2_Ximage, files_2_Ximage 
    from limg_common import rgb_2_gray

    # 1. 载入图像    
    files = list_files(path)
    X, fnames = quick_files_2_Ximage(files, shape) # 读取图像, 统一尺寸
    #X, fnames = files_2_Ximage(files, shape) # 读取图像, 统一尺寸

    # 2. 提取特征
    n = len(X) if type(X) == list else X.shape[0]
    X_fnames_list = [(X[i], fnames[i]) for i in range(n)]
    X_fnames_list = muti_process(X_fnames_list, 6, images_2_feature, args=[feature_size, fea_types, expand_num], use_share_path=None)
    X = np.array([fea_t[0] for fea_t in X_fnames_list])
    T = [int(fea_t[1].split('_')[-1].split('.')[-2]) for fea_t in X_fnames_list] # 从图像文件名读取图像的分类索引
    log('finish to collect fea.') 

    # 3. 独热编码
    T = np.array(T) 
    T = idx_2_oneHotEncoder(T)
    log('finish to encode T. T.shape:', T.shape)
    return X, T


def collect_train_data():
    """
    废弃
    """
    # 返回数字图片的dict, images字段:图片数组，target字段：图片上的数字数组
    # 1797张8*8图片
    from sklearn import datasets, cross_validation, metrics
    from plt_common import show_images 
    digits = datasets.load_digits()
    
    
    # 用one-hot-encoding的10维向量表示10个数字
    T = np.zeros((digits.target.shape[0],10)) 
    T[np.arange(len(T)), digits.target] += 1

    if g_open_debug:
        from limg_common import equalization_gray_hist
        #save_XImage_T(equalization_gray_hist(255 - digits.data, 0, 255), T, shape=(8, 8), path='digits')

    # 把数据集拆分成训练集和测试集
    X_train, X_test, T_train, T_test = cross_validation.train_test_split(
            digits.data, T, test_size=0.4)
    # 把测试集拆分成校验集和最终测试集
    X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
            X_test, T_test, test_size=0.5)
        

    if g_open_debug:
        print '训练集、校验集、测试集的X量级：', X_train.shape, X_validation.shape, X_test.shape
        print '训练集、校验集、测试集的Y量级：', T_train.shape, T_validation.shape, T_test.shape
    
        # 显示数据集
        #for i in range(3):
        #    show_images(digits.images[i * 10: i * 10 + 10])
    
    return X_train, T_train, X_validation, T_validation, X_test, T_test


def small_train(tag, X_train, T_train, X_validation, T_validation, X_test, T_test, num1=20, num2=20, train_method='random_grad_desc', save_file=None):
    from plt_common import show_images, show_array, show_predict_numbers  

    # 构建神经网络
    if tag == 'nn':
        nn = NeuronNetwork(
                X_train.shape[1], # 输入样本特征数 
                [num1, num2, T_train.shape[1]], # 每一层网络输出特征数
                [LogisticLayer, LogisticLayer, SoftmaxLayer], # 每一层网络的激活函数
                crossEntropy_cost, # 损失函数
                crossEntropy_cost_deriv # 损失函数梯度计算
            )
    elif tag == 'cnn':
        nn = NeuronNetwork(
                X_train.shape[1], # 输入样本特征数 
                [num1, num2, T_train.shape[1]],
                [AvgpoolLayer, LogisticLayer, SoftmaxLayer],
                crossEntropy_cost,
                crossEntropy_cost_deriv 
        )

    # 训练
    if train_method == 'random_grad_desc':
        (cost, terations, costs_vec) = nn.train_random_grad_desc(
                X_train,
                T_train,
                X_validation,
                T_validation,
                batch_size=25,
                max_nb_of_iterations=300, 
                learning_rate=0.1
            )
        minibatch_costs, training_costs, validation_costs = costs_vec
        arrs = (minibatch_costs, training_costs, validation_costs)
        labels = ('cost minibatches', 'cost training set', 'cost validation set')
    elif train_method == 'grad_desc':
        (cost, terations, costs_vec) = nn.train_grad_desc(
                X_train,
                T_train,
                X_validation,
                T_validation,
                max_nb_of_iterations=1600, 
                learning_rate=0.1
            )
        training_costs, validation_costs = costs_vec
        arrs = (training_costs, validation_costs)
        labels = ('cost full training set', 'cost validation set')
    show_array(arrs, labels, title='Decrease of cost over backprop iteration', show_fpath='data/cost_func.jpg')
            
    if  g_open_debug:
         g_debug_cache['costs'] = labels, arrs
        

    # 评估准确率 
    test_accuracy = nn.test_accuracy(X_test, T_test)
    print 'test_accuracy:', test_accuracy   

    
    # 预测
    y_test = nn.predict(X_test)
    y_true = np.argmax(T_test, axis=1) 
    y_pred = np.argmax(y_test, axis=1)   
    show_predict_numbers(y_true, y_pred, show_fpath='data/check.jpg')    
    if save_file is not None:
        str_2_file(nn.to_string(), save_file)

def small_test(load_file, X_test, T_test=None):
    from plt_common import show_images, show_array, show_predict_numbers  
    s = file_2_str(load_file)
    nn = NeuronNetwork(create_string=s)
    # 预测
    y_test = nn.predict(X_test)
    if T_test is not None:
        y_true = np.argmax(T_test, axis=1) 
        y_pred = np.argmax(y_test, axis=1)   
        show_predict_numbers(y_true, y_pred, show_fpath='data/check.jpg')    
        test_accuracy = nn.test_accuracy(X_test, T_test)
        print 'test_accuracy:', test_accuracy 
        #cv2image_2_file(image_2_cv2image(gray_2_rgb(X_test[0])), 'test1.bmp')
    return y_test

def test():
    test_nums = [1]
    if 0 in test_nums:
        layer = LinearLayer(3, 4)
        strings = layer.to_string()
        print strings 
        layer.from_string(strings)
    if 1 in test_nums:
        nn = NeuronNetwork(
                2, 
                [3, 4, 5],
                [LogisticLayer, LogisticLayer, SoftmaxLayer],
                crossEntropy_cost,
                crossEntropy_cost_deriv 
            )
        s = nn.to_string()
        nn = NeuronNetwork(create_string=s)

def oneHotEncoder_2_idx(T):
    return np.argmax(T, axis=1)

def idx_2_oneHotEncoder(T, tag_num=None):
    if tag_num is None:
        tag_num = np.max(T) + 1
    Y = np.zeros((T.shape[0], tag_num))
    Y[np.arange(len(Y)), T] += 1
    #for i in range(len(T)):
    #    Y[i, T[i]] = 1
    return Y

 
def show_nn_iterations(X, Ys, T, label_arr, costs_arr, path='data/'):
    """
    X是图像数据集, 矩阵为 n, c, r, channel或者n, c, r
    Ys是预测结果的每次迭代的集合
    T是真实的结果
    """ 
    sample_width = 8 #每种小图像的宽度归一化
    v_max_num = 30 # 每种小图像的个数限制
    col_num = 30 # 小图像一行显示的个数
    delt = -1 # 小图像的边框像素宽度
    iter_delt = 50 # 每隔xx次迭代，显示的迭代次数减半
    gif_fpath = 'data.train'
    is_equalization = False 
    tmp_fpath = 'data/tmp.images_2_gif.bmp'

    from limg_common import equalization_gray_hist, even_random_idxs, gray_2_rgb, rgb_resize, rgb_resize, merge_images, border_image 
    from lcv2_common import images_2_giffile, file_2_cv2image
    from plt_common import show_images, show_array, show_predict_numbers  

    #  直方图均衡化
    if is_equalization: # 必须是灰度图像才有这个操作
        X = equalization_gray_hist(255 - X, 0, 255)
       
    # 按分类随机等量抽样
    sample_idxs = even_random_idxs(T, v_max_num) # 每个分类小图像随机选择v_max_num个 
    X = X[sample_idxs]
    Ys = Ys[:, sample_idxs]
    T = T[sample_idxs]

    if len(X.shape) == 3:
        X = gray_2_rgb(X) # 转换为n个rgb图像
    #X = rgb_resize(X, (None, sample_width))

    iter_images = np.array([merge_images(X, Y, col_num=col_num,  delt=delt) for Y in Ys])
    #iter_idxs = np.array([i for i in range(0, len(Ys), 2) if i % (int(i / iter_delt) + 1) == 0 or i >= len(Ys) - 30])
    iter_idxs = np.array([i for i in range(0, len(Ys)) if i % 30 == 0 or i >= len(Ys) - 20])
    iter_images = iter_images[iter_idxs]
    images_2_giffile(iter_images, path + gif_fpath + '.image.gif', tmp_fpath=tmp_fpath, duration=0.2)

    iter_images = []
    for i in iter_idxs:
        show_array(costs_arr, label_arr, title='Decrease of cost over backprop iteration', show_fpath=tmp_fpath, stop=i)
        iter_images.append(file_2_cv2image(tmp_fpath))
    iter_images = np.array(iter_images) 
    images_2_giffile(iter_images, path + gif_fpath + '.curve.gif', tmp_fpath=tmp_fpath, duration=0.2) 

    iter_images = []
    for i in iter_idxs:
        show_predict_numbers(T, Ys[i], show_fpath=tmp_fpath) 
        iter_images.append(file_2_cv2image(tmp_fpath))   
    iter_images = np.array(iter_images) 
    images_2_giffile(iter_images, path + gif_fpath + '.table.gif', tmp_fpath=tmp_fpath, duration=0.2) 


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


def main():

    # 参数
    tag = 'cnn'
    tag = 'nn'
    select = 'predict'
    select = 'train'
    read_shape = (100, 100, 3)

    train_method='grad_desc'
    train_method='random_grad_desc'
    
    expand_num = 2 # 非0样本扩充图片的数量
    fea_types = ['pixel'] # 可以取值['pixel', 'hist', 'rgb_hist']
    

    # 手写数字集
    path = '../../data/example/digits/'
    expand_num = 0
    read_shape = (8, 8, 3) 
    feature_size = (8, 8, 1)


    # 二维码商标图
    #path = '../data/trade_mark/'
    #read_shape = (100, 100, 3)
    #feature_size = (40, 40, 1)


    # 例子
    #path = '../data/example/'
    #read_shape = (100, 100, 3)
    #feature_size = (40, 40, 1)


    #X, T = collect_images_feature(path=path, shape=read_shape, feature_size=feature_size, fea_types=fea_types, expand_num=expand_num)
    #X_train, T_train, X_validation, T_validation, X_test, T_test = train_test_split(X, T)

    #import fea 
    #X_train, T_train, X_validation, T_validation, X_test, T_test = fea.collect_train_data()



    from sklearn import datasets, cross_validation, metrics
    digits = datasets.load_digits()
    T = np.zeros((digits.target.shape[0],10)) 
    T[np.arange(len(T)), digits.target] += 1
    random_path = "./tmp.random_nums.txt" if g_open_test else None
    X_train, X_validation, X_test, T_train, T_validation, T_test = cross_validation_split(digits.data, T, random_path)

    if select == 'train':
        if tag == 'cnn':
            tag = 'cnn'
            num1 = (feature_size, (3, 3, 1, 2), (2, 2), 2)
            num2 = 15
        if tag == 'nn':
            tag = 'nn'
            num1 = 20
            num2 = 20

        # 三层神经网络
        small_train(tag, X_train, T_train, X_validation, T_validation, X_test, T_test, num1, num2, train_method=train_method, save_file='tmp.simple_nn.txt')
    if select == 'predict':
        small_test('tmp.simple_nn.txt', X_test, T_test)
    if g_debug_cache != {} and 'Y_validation' in g_debug_cache:
        cache = g_debug_cache['Y_validation']
        from limg_common import gray_2_rgb, equalization_gray_hist
        shape = (-1, read_shape[0], read_shape[1])
        X_show = X_validation.reshape(shape) # 转成图像矩阵
        T_show = np.argmax(T_validation, axis=1) # 单值模式
        Y_shows = np.array([np.argmax(v, axis=1) for v in cache])
        labels_arr, costs_arr = g_debug_cache['costs']
        show_nn_iterations(X_show, Y_shows, T_show, labels_arr, costs_arr, path='data/')

        fpath1 = 'data/data.train.image.gif'
        fpath2 = 'data/data.train.curve.gif'
        fpath3 = 'data/data.train.table.gif'
        fpath12 = 'data/data.train.curve_image.gif'
        fpath23 = 'data/data.train.curve_table.gif'
        #show_nn_iterations_gif(fpath2, fpath1, fpath12)
        #show_nn_iterations_gif(fpath2, fpath3, fpath23)


if __name__ == "__main__":
    log('begin to run program')
    main()
    log('finish to run program')
