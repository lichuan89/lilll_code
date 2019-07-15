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
import time
import datetime
import urllib
from lcommon import str_2_json
from lcommon import json_2_str
from lcommon import smart_str_list 
from lcommon import log 

g_rule = {
    '<tab>': '\t',
    '<space>' : ' ',
    '<stand>' : '|',
}

def cat(us, tags=[]):
    return us
    

def b64(us, tags=['utf8']):
    code = tags[0]
    return base64.b64encode(us.encode(code, 'ignore'))

def unb64(us, tags=['utf8']):
    code = tags[0]
    return base64.b64decode(us).decode(code, 'ignore') 

def quote(us, tags=['utf8']):
    code = tags[0]
    return urllib.quote(us.encode(code, 'ignore')) 

def unquote(us, tags=['utf8']):
    code = tags[0]
    return urllib.unquote(us).decode(code, 'ignore')

def domain(us, tags=[]):
    domains = get_domains(url)
    return domains[-1] if domains != [] and domains[0] != url else ''

def split(us, tags=['\t', None]):
    sep = tags[0]
    cnt = tags[1] if len(tags) > 1 else None
    output = us.split(sep)
    if cnt is not None and len(output) < cnt:
        output += ['' for i in range(cnt - len(output))]
    return '\t'.join(output)


def match(us, tags):
    v = tags[0]
    if us.find(v) == -1:
        return None
    return v 

def replace(us, tags):
    rule = tags[0]
    val = tags[1]
    return us.replace(rule, val) 

def rematch(us, tags):
    rule = tags[0]
    sep = tags[1] if len(tags) > 1 else '____'
    m = re.findall(rule, us)
    if m != []: 
        return sep.join(m)
    else:
        return None

def rereplace(us, tags):
    rule = tags[0]
    val = tags[1]
    rule = re.compile(rule)
    us = rule.sub(val, us)
    return us

def select_col(us, tags): # 模式选全字段
    idxs = [int(i) for i in tags]
    arr = us.split('\t')
    output = []
    for i in idxs:
        v = arr[i] if i < len(arr) else ''
        output.append(v)
    return '\t'.join(output)


def add_const(us, tags):
    const = tags[0]
    pos = tags[1] if len(tags) > 1 else 'after'
    return '%s\t%s' % (us, const) if pos == 'after' else '%s\t%s' % (const, us)
    

def process_lines(process_col, arr, lines=None):
    """
    一般情况下, arg1为: 处理的字段列号(默认all),r(替换原字段),a(追加到行尾),i(插入到原字段后),px(在后面参数中x都替换为|)
    如果有参数使用正则, 则正则格式为: 分隔符 regex 分隔符 替换|的字符串
    """
    # 第一个参数开头的数字表示处理的列序号
    i = 0
    s = arr[0]
    while s[i] == '-' or '0' <= s[i] <= '9':
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
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        line = line.decode('utf8', 'ignore')
        arr = line.split('\t')
        if idx == '':
            arr = [line]
            idx = 0
        s = arr[idx]
        v = process_col(s) if col_arg == [] else process_col(s, col_arg)
        if 'f' in mod:
            if v is None:
                print line.encode('utf8', 'ignore')
            continue
        if v is None:
            continue
        if 'r' in mod:
            arr[idx] = v
        elif 'i' in mod:
            arr = arr[: idx + 1] + [v] + arr[idx + 1: ]
        elif 'a' in mod:
            arr.append(v)
        print '\t'.join(arr).encode('utf8', 'ignore')


def arg_2_func(string):
    """
    例如: arg_2_func("____select_idx____0____1")
    """
    # 将字符串转换为参数list
    arr = smart_str_list(string, use_end_ch=False)
    arr = [v.decode('utf8', 'ignore') for v in arr]
    #try:
    if True:
        if arr == []:
            log('notice', 'arg_2_func with no func. {0}'.format(arr))
            return None
        elif len(arr) == 1: # ____select_idx --> select_idx() 
            log('notice', 'arg_2_func with no arg. {0}'.format(func))
            func = arr[0]
            output = eval(func)()
        elif arr[1] != '': # ____select_idx____0a____xx --> process_lines(select_idx, [0a, xx])
            func = arr[0] 
            log('notice', 'arg_2_func with mod arg. {0} {1} {2}'.format('process_lines', func, arr[1: ]))
            output = process_lines(eval(func), arr[1:])
        else:
            func = arr[0]
            arg = arr[2: ] # ____select_idx________xx --> select_idx([xx]) 
            log('notice', 'arg_2_func with no mod arg. {0} {1}'.format(func, arr[2: ]))
            output = eval(func)(arg)
    #except Exception as e: 
    #    print >> sys.stderr, "failed to process.", e, sys._getframe().f_lineno
    #    return None 
    return output


if __name__ == "__main__":
    func_arg = sys.argv[1]
    arg_2_func(func_arg)
