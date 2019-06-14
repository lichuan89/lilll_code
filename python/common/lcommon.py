#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2017/01/01  
    @note   实现通用工具
"""
import os 
import sys
import struct
import hashlib
import logging
import json 
import hashlib
import datetime
import re
import fileinput
import urllib
import urllib2
import threading
import urlparse
import time 
import fcntl 


g_open_log = True 
g_current_time = None


g_top_host_postfix = (
            '.co.kr', '.co.in', '.co.jp', '.com.ag', '.com.br', '.com.bz', '.com.cn',
            '.com.co', '.com.es', '.com.hk', '.com.mx', '.com.tw', '.co.nz',
            '.co.uk', '.edu.cn', '.firm.in', '.gen.in', '.idv.tw', '.ind.in',
            '.me.uk', '.net.ag', '.net.br', '.net.bz', '.net.cn', '.net.co',
            '.net.in', '.net.nz', '.nom.co', '.nom.es', '.org.ag', '.org.cn',
            '.org.es', '.org.in', '.org.nz', '.org.tw', '.org.uk', '.ag', '.top',
            '.am', '.asia', '.at', '.be', '.biz', '.bz', '.ca', '.cc', '.cn',
            '.co', '.com', '.de', '.es', '.eu', '.fm', '.fr', '.gs', '.hk',
            '.in', '.info', '.io', '.it', '.jobs', '.jp', '.la', '.me',
            '.mobi', '.ms', '.mx', '.net', '.nl', '.nu', '.org', '.se',
            '.tc', '.tk', '.tv', '.tw', '.us', '.vg', '.ws', '.xxx')


def log(*arg):
    """
    write_log
    """
    global g_open_log
    global g_current_time
    if not g_open_log:
        return
    delt_time = time.time() - g_current_time if g_current_time is not None else 0
    g_current_time = time.time()
    prefix = '[log] [%s] [%.3f]' % (time.strftime("%Y-%m-%d %H:%M:%S"), delt_time)
    coding = 'utf8'
    if type(arg) == type((1, 2)):
        arg = [v if v is not None else str(None) for v in arg]
        if type(arg) == type((1, 2)):
            arg = list(arg)
        arg = [arg if type(arg) not in set([type([]), type({})]) else json_2_str(arg)]
        li = [elem.encode(coding, 'ignore') for elem in arg]
        print >> sys.stderr, prefix, ' '.join(li)
    else:
        if arg is None:
            arg = str(None)
        print >> sys.stderr, prefix, arg.encode(coding, 'ignore')


def str_2_json(uni):
    """
    unicode字符串转json 
    """
    obj = {}
    try:
        obj = json.loads(uni)
    except Exception as e:
        print >> sys.stderr, e
        return None
    return obj


def json_2_str(obj):
    """
    json转unicode字符串
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        return None 


def file_2_str(fpath, decode=None):
    """
    从文件读取字符串
    """
    with open(fpath, 'rb') as f:
        context = f.read()
        if decode is not None:
            context = context.decode(decode, 'ignore')
        return context

def str_2_file(string, fpath, encode=None):
    """
    将字符串写入文件
    """
    with open(fpath, 'wb') as f:
        if encode is not None:
            string = string.encode(encode, 'ignore')
        f.write(string)


def clear_dir(path, is_del=True):
    """
    创建文件夹,或者删除文件夹下的直接子文件
    """
    if not os.path.exists(path):
        os.makedirs(path)
    elif is_del:
        arr = os.listdir(path)
        for v in arr:
            f = '%s/%s' % (path, v) 
            if os.path.isfile(f):
                os.remove(f)

def list_files(path):
    arr = os.listdir(path)
    return ['%s/%s' % (path, v) for v in arr if os.path.isfile('%s/%s' % (path, v)) and v.find('.') != 0]


def read_dir(path, decode='utf8'):
    arr = os.listdir(path)
    output = []
    for v in arr:
        f = '%s/%s' % (path, v)
        if os.path.isfile(f):
            context = file_2_str(f)
            if decode is not None:
                context = context.decode(decode, 'ignore')
            output.append(context)
    return output 


def md5(ustring):
    m1 = hashlib.md5()
    m1.update(ustring.encode("utf-8"))
    token = m1.hexdigest()
    return token


def get_domains(url):
    """
    such as: 
    https://list.jd.com/list.html?cat=9987,653,655
    http://item.jd.com/100001009384.html

    # 返回domain列表，最后一个domain是最宽泛的 
    """
    arr = url.split("/")
    if len(arr) >= 3:
        url = arr[2].split('?')[0]
    regx = r'([^\.]+[.])?([^\.]+)(' + \
            '|'.join([h.replace('.', r'\.') for h in g_top_host_postfix]) + ')\\b'
    pattern = re.compile(regx, re.IGNORECASE)
    m = pattern.findall(url)
    if m:
        arr = [] 
        for elem in m:
            arr.append(''.join(elem))
            if elem[0] != '' and len(elem) > 1: 
                arr.append(''.join(elem[1:]))
        return arr 
    else:   
        return [domain]


def expand_json(obj, is_simple_key=True, path=[]):
    def merge_dict(o1, o2):
        return dict([v for v in o1.items()] + [v for v in o2.items()])

    kvs = {}
    kns = {}
    path_key = '--'.join(path)
    if type(obj) == dict:
        for k, v in obj.items():
            o1, o2 = expand_json(v, is_simple_key, path + [k])
            kvs = merge_dict(kvs, o1)
            kns = merge_dict(kns, o2)
        kns['%s--dict' % path_key] = len(obj)
    elif type(obj) == list:
        for i in range(len(obj)):
            o1, o2 = expand_json(obj[i], is_simple_key, path + [unicode(i)])
            kvs = merge_dict(kvs, o1)
            kns = merge_dict(kns, o2)
        kns['%s--list' % path_key] = len(obj)
    else:
        if is_simple_key:
            kvs[path[-1] if path != [] else ''] = obj
        else:
            kvs[path_key] = obj
        

    return kvs, kns 


def test(opt):
    if opt == "file_opt" or opt == "all":
        clear_dir('tmp.muti_process')
        str_2_file('abcd\nefg', 'tmp.muti_process/1')
        str_2_file('abcd\nefg', 'tmp.muti_process/2')
        print read_dir('tmp.muti_process')
        clear_dir('tmp.muti_process')
    if opt == 'get_domains' or opt == 'all':
        print get_domains('https://list.jd.com/list.html?cat=9987,653,655')

    if opt == 'expand_json' or opt == 'all':
        obj = [
            'v1',
            {'k2': 'v2', 'k3': 'v3', 'k4': [5, 6]},
            7
        ]
        print obj
        print expand_json(obj)
        print expand_json(obj, is_simple_key=False)

if __name__ == "__main__":
    opt = "all"
    test(opt)
