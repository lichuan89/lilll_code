#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2017/01/01  
    @note   实现python多进程处理输入流的函数
"""

import traceback
import base64
import re
import sys
import random 
import time
import datetime
import urllib
from lcommon import str_2_json
from lcommon import json_2_str
from lcommon import smart_str_list 
from lcommon import expand_pair 
from lcommon import json_2_kvs 
from lcommon import log 
from lcommon import crawl 
from lcommon import get_domains 

g_rule = {
        '<TAB>' : '\t',
        '<SPACE>': ' ',
        '<VERT>': '|',
        '<UNDER>': '_',
        '<2UNDER>': '__',
    }

g_data = {}

def cat(us, tags=[]):
    ## print lines. cmd: cat, cat__r
    return us
    
def b64(us, tags=['utf8']):
    ## base64 one col. cmd: bs64, b64__3i, bs64__3i__utf8 
    code = tags[0]
    return base64.b64encode(us.encode(code, 'ignore'))

def unb64(us, tags=['utf8']):
    ## unbase64 one col. cmd: unb64, unb64__3i, unb64__3i__utf8
    code = tags[0]
    return base64.b64decode(us).decode(code, 'ignore') 

def quote(us, tags=['utf8']):
    ## quote one col. cmd: quote, quote__3i, quote__3i__utf8 
    code = tags[0]
    return urllib.quote(us.encode(code, 'ignore')) 

def unquote(us, tags=['utf8']):
    ## unquote one col. cmd: unquote, unquote__3i, unquote__3i__utf8
    code = tags[0]
    return urllib.unquote(us.encode('utf8', 'ignore')).decode(code, 'ignore')

def domain(us, tags=[]):
    ## extract domain form on col. cmd: domain, domain__3i
    domains = get_domains(us)
    return domains[-1] if domains != [] and domains[0] != us else ''

def split(us, tags=['\t', None]):
    ## split on col with a sep. cmd: split__3i__<UNDER>, split__3i__<UNDER>__4
    sep = tags[0]
    cnt = int(tags[1]) if len(tags) > 1 else None
    output = us.split(sep)
    if cnt is not None and len(output) < cnt:
        output += ['' for i in range(cnt - len(output))]
    return '\t'.join(output)

def join(us, tags=['\t', 0, -1]):
    ## join cols. cmd: join__r__<UNDER>__0__1 
    sep = tags[0] if len(tags) > 0 else '\t'
    begin = int(tags[1]) if len(tags) > 1 else 0
    end = int(tags[2]) if len(tags) > 2 else -1

    output = us.split('\t')
    s = sep.join(output[begin: end + 1 if end != -1 else len(output)])
    output.append(s)
    return '\t'.join(output) 


def match(us, tags):
    ## match string. cmd: match__1a__hello, match__a__hello, match__af__hello, match__ak__hello 
    v = tags[0]
    if us.find(v) == -1:
        return None
    return v 

def replace(us, tags):
    ## replace from string1 to string2. cmd: replace__0r__hello__world
    rule = tags[0]
    val = tags[1]
    return us.replace(rule, val) 

def rematch(us, tags):
    ## match with regex. cmd: rematch__1a__[0-9]+, rematch__1a__[0-9]+__<VERT>
    rule = tags[0]
    sep = tags[1] if len(tags) > 1 else '____'
    m = re.findall(rule, us)
    if m != []: 
        return sep.join(m)
    else:
        return None

def rereplace(us, tags):
    ## replace with regex. cmd: rereplace__3i__[0-9]+__hello 
    rule = tags[0]
    val = tags[1]
    rule = re.compile(rule)
    us = rule.sub(val, us)
    return us

def cols(us, tags):
    ## select cols, cmd: cols__r__3__2
    idxs = [int(i) for i in tags]
    arr = us.split('\t')
    output = []
    for i in idxs:
        v = arr[i] if i < len(arr) else ''
        output.append(v)
    return '\t'.join(output)


def colIdx(us, tags=[]):
    ## add index for echo col. cmd: colIdx__r__0 
    begin = tags[0] if len(tags) > 0 else ''
    if 'col_head' in g_data:
        return us

    count = len(us.split('\t'))
    if re.search('^[0-9-]+$', begin) is None:
        output = [begin] + [unicode(i) for i in range(1, count)]
    else:
        begin = int(begin)
        output = [unicode(i) for i in range(begin, begin + count)]
    g_data['col_head'] = '\t'.join(output)
    return '\t'.join(output) + '\n' + us 
    

def addHead(us, tags):
    ## add string before first line. cmd: addHead__r__hello__world__!
    arr = tags

    g_data.setdefault('row_idx', -1)
    g_data['row_idx'] += 1 
    
    if g_data['row_idx'] == 0:
        return '\t'.join(arr) + '\n' + us
    return us 

def delHead(us, tags):
    ## del string at the head. cmd: delHead__r__2 
    idx = int(tags[0]) if len(tags) > 0 else 1

    g_data.setdefault('row_idx', -1)
    g_data['row_idx'] += 1 

    if g_data['row_idx'] < idx:
        return None 
    return us

def rand(us, tags=[0.5]):
    ## prodce a random prob use filter. cmd: rand__a__0.6, rand__ak__0.6
    thre = float(tags[0])
    prob = random.random()
    if prob < thre:
        return unicode(prob) 
    else:
        return None

def const(us, tags):
    ## add a col with const value. const__3i__川流不息
    const = tags[0]
    return const 
    

def jsonKvs(us, tags):
    ## extract values from json with keys. cmd: jsonKvs__1r__key1__key2
    keys = tags
    obj = str_2_json(us) 
    kvs = json_2_kvs(obj, keys)
    output = [unicode(kvs[k]) if k in kvs else '' for k in keys]
    return '\t'.join(output)

def jsonUnpair(us, tags):
    ## format json from {"column": "name", "value": "duck"} to {"name": "duck"}. cmd: jsonUnpair__1r__column__value 
    keys = tags[0].split(',')
    vals = tags[1].split(',')
    obj = str_2_json(us) 
    o = expand_pair(obj, keys, vals)
    if o is None:
        o = obj
    return json_2_str(o)  

def html(us, tags=['utf8']):
    ## crawl static html. cmd: html__1i__utf8
    decode = tags[0] if len(tags) > 0 else 'utf8'
    url = us.encode('utf8', 'ignore')
    data = crawl(url, decode)
    data = data.replace('\n', '\r') if data is not None else ''
    return data 

def rowIdx(us, tags=['']):
    ## add index for echo row. cmd: rowIdx__0b__0 
    begin = tags[0] if len(tags) > 0 else ''
    g_data.setdefault('row_idx', begin)

    idx = g_data['row_idx']
    if type(idx) == int or re.search('^[0-9-]+$', idx) is not None:
        g_data['row_idx'] = int(g_data['row_idx']) + 1
    else:
        g_data['row_idx'] = 1
    return unicode(idx)
    

def process_lines(process_col, arr, lines):
    """
    lines 为unicode的行数组
    一般情况下, arg1为: 处理的字段列号(默认all),r(替换原字段),a(追加到行尾),i(插入到原字段后),px(在后面参数中x都替换为|)
    如果有参数使用正则, 则正则格式为: 分隔符 regex 分隔符 替换|的字符串
    """
    # 第一个参数开头的数字表示处理的列序号
    i = 0
    s = arr[0] if arr != [] else 'r'
    while i < len(s) and (s[i] == '-' or '0' <= s[i] <= '9'):
        i += 1
    idx = s[: i] # 为空表示处理整行
    idx = int(idx) if idx != '' else ''
    s = s[i: ]

    # 第一个参数p后面的字符串,表示等价于|
    i = s.find('p')
    if i != -1:
        pipe = s[i + 1: ]
        s = s[: i]
        arr = [v.replace(pipe, '|') for v in arr]

    # 模式分为r(替换原字段),a(追加到行尾),i(插入到原字段后)
    mod = s

    if 'g' in mod:
        for i in range(1, len(arr)):
            for k, v in g_rule.items():
                arr[i] = arr[i].replace(k, v) 

    col_arg = arr[1: ]
    log('notice', 'process_lines. idx:{0}, mod:{1}, col_arg:{2}'.format(idx, mod, col_arg))
    output = []
    for line in lines:
        if line[-1] == '\n':
            line = line[:-1]
        #line = line.decode('utf8', 'ignore')
        arr = line.split('\t')
        if idx == '':
            arr = [line]
            index = 0
        else:
            index = idx
        if index < 0:
            index = len(arr) + index
        if index < len(arr):
            s = arr[index]
            v = process_col(s) if col_arg == [] else process_col(s, col_arg)

            if 'k' not in mod:
                if 'f' in mod:
                    if v is None:
                        output.append(line) #.encode('utf8', 'ignore')
                    continue
                if v is None:
                    continue
            else:
                if v is None:
                    v = ''
            if 'r' in mod:
                arr[index] = v
            elif 'i' in mod:
                arr = arr[: index + 1] + [v] + arr[index + 1: ]
            elif 'a' in mod:
                arr.append(v)
            elif 'b' in mod:
                arr = arr[: index - 1] + [v] + arr[index - 1: ] if index != 0 else [v] + arr
        #print '\t'.join(arr).encode('utf8', 'ignore')
        output.append('\t'.join(arr))
    return output

def quick_process_lines(process_col, arr):
    mod = arr[0]
    i = mod.find('p')
    if i != -1:
        mod = mod[: i]
    # 多进程
    if 'm' in mod:
        def worker(lines, args):
            process_col = args[0]
            arr = args[1: ]
            output = process_lines(process_col, arr, lines)
            return output
        from lcmd import muti_process_stdin
        muti_process_stdin(worker, [process_col] + arr, batch_line_num=30, thread_running_num=7, use_share_path=None)
    else:
        inputs = []
        batch_num = 300
        for line in sys.stdin:
            if line[-1] == '\n':
                line = line[:-1]
            line = line.decode('utf8', 'ignore')
            inputs.append(line)
            if len(inputs) > batch_num:
                outputs = process_lines(process_col, arr, inputs)
                for line in outputs:
                    print line.encode('utf8', 'ignore')
                inputs = []
        outputs = process_lines(process_col, arr, inputs)
        for line in outputs:
            print line.encode('utf8', 'ignore')


if __name__ == "__main__":
    func_arg = sys.argv[1]
    arg_2_func(func_arg)
