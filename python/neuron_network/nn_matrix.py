#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import base64
import numpy as np
import random
import os
from lcommon import str_2_json, json_2_str, file_2_str, str_2_file

def cross_validation_split(X, T, fpath="./tmp.random_nums.txt"):
    """
    将数据集拆分为60%训练集、20%验证集、20%测试集
    """
    n = len(X)
    nums = range(n)
    random.shuffle(nums)
    if fpath is not None:
        if not os.path.exists(fpath):
            str_2_file(json_2_str(nums), fpath)
        else:
            nums = str_2_json(file_2_str(fpath))
    train_nums = nums[: int(0.6 * n)]
    validation_nums = nums[int(0.6 * n): int(0.8 * n)]
    test_nums = nums[int(0.8 * n): n]
    X_train = np.array([X[i] for i in train_nums])
    T_train = np.array([T[i] for i in train_nums])
    X_validation = np.array([X[i] for i in validation_nums])
    T_validation = np.array([T[i] for i in validation_nums])
    X_test = np.array([X[i] for i in test_nums])
    T_test = np.array([T[i] for i in test_nums])
    return X_train, X_validation, X_test, T_train, T_validation, T_test

def matrix_2_string(matrix):
    return json_2_str([base64.b64encode(matrix.tostring()), list(matrix.shape)])


def string_2_matrix(string):
    matrix_shape = str_2_json(string)
    return np.fromstring(base64.b64decode(matrix_shape[0])).reshape(matrix_shape[1])


def accuracy_score(y_true, y_pred):    
    diff = y_true == y_pred 
    score = float(len(diff[diff == True])) / len(diff)
    # 准确率, 代替sklearn.metrics.accuracy_score
    # 等价于sklearn.metrics.recall_score(y_true, y_pred, average='weighted')
    # 等价于sklearn.metrics.recall_score(y_true, y_pred, average='micro')
    return score

def recall_score(y_true, y_pred):
    categorys = {}
    for c in y_true:
        categorys.setdefault(c, 0)
        categorys[c] += 1
 
    macro_score = 0
    category_2_score = {} # 各个类的召回率, 以及各个类样本量的占比
    for c, n in categorys.items():
        diff = y_pred[y_true == c] == c
        score = float(len(diff[diff == True])) / len(diff)
        macro_score += score
        category_2_score[c] = {'score' : score, 'num': float(len(diff)) / len(y_true)}
    # 各类均匀分布的情况下总召回率
    macro_score = macro_score / len(categorys) # sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    return macro_score, category_2_score

def confusion_matrix(y_true, y_pred):
    categorys = list(set([v for v in y_true]))
    m = {} 
    for c1 in y_true:
        for c2 in y_pred:
            m.setdefault((c1, c2), 0)
            m[(c1, c2)] += 1
    return m, categorys

# https://blog.csdn.net/CherDW/article/details/55813071
def test(tag):
    if tag == 'score' or tag == 'all':
        from sklearn import metrics
        y_true = np.array([1, 2, 2, 3, 3, 2, 3, 3, 10, 10])
        y_pred = np.array([1, 1, 2, 3, 3, 1, 1, 1, 10, 1])
        print '-->accuracy_score:', metrics.accuracy_score(y_true, y_pred)
        print '-->micro.recall_score:', metrics.recall_score(y_true, y_pred, average='micro') 
        print '-->weighted.recall_score:', metrics.recall_score(y_true, y_pred, average='weighted') 
        print '-->macro.recall_score:', metrics.recall_score(y_true, y_pred, average='macro')
        print '-->accuracy_score:', accuracy_score(y_true, y_pred)
        print '-->recall_score:', recall_score(y_true, y_pred)
        print '-->confusion_matrix:', metrics.confusion_matrix(y_true, y_pred, labels=list(set([v for v in y_true])))
        print '-->confusion_matrix:',  confusion_matrix(y_true, y_pred)
    elif tag == 'file_matrix' or tag == 'all':
        W = np.random.randn(64, 20) * 0.1
        str_2_file(matrix_2_string(W), '64x20.randn.txt')
        W1 = string_2_matrix(file_2_str('64x20.randn.txt'))
        W = np.random.randn(20, 20) * 0.1
        str_2_file(matrix_2_string(W), '20x20.randn.txt')
        W = np.random.randn(20, 10) * 0.1
        str_2_file(matrix_2_string(W), '20x10.randn.txt')


if __name__ == "__main__":
    test('score')
