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
import random
import urllib
from lcommon import str_2_json
from lcommon import json_2_str
from lcommon import str_2_file
from lcommon import get_domains 
from lprocess_line import *


def process_field(tags, func):
    idx = int(tags[0])
    tag = tags[1] if len(tags) >= 2 else 'append'
    for line in sys.stdin:
        line = line[: -1].decode('utf8', 'ignore')
        arr = line.split('\t')
        v = func(arr[idx])
        if tag == 'replace':
            arr[idx] = v
        else:
            arr.append(v)
        print '\t'.join(arr).encode('utf8', 'ignore')

     
def quote_field(tags):
    process_field(tags, lambda v: urllib.quote(v))

def unquote_field(tags):
    process_field(tags, lambda v: urllib.unquote(v))

def domain_field(tags=[0, 'append']):
    def get_domain(url):
        domains = get_domains(url)
        return domains[-1] if domains != [] else ''
    process_field(tags, func=get_domain)


def sum_field(tags):
    idxs = [int(v) for v in tags]
    dic = {}
    for line in sys.stdin:
        line = line[: -1].decode('utf8', 'ignore')
        arr = line.split('\t')
        key = '\t'.join([arr[idx] for idx in range(len(arr)) if idx not in set(idxs)])
        dic.setdefault(key, {})
        for idx in idxs:
            dic[key].setdefault(idx, 0)
            dic[key][idx] += float(arr[idx])
        dic[key].setdefault('n', 0)
        dic[key]['n'] += 1
    for key, data in dic.items():
        arr = [key, unicode(data['n'])]
        arr += [unicode(data[idx]) for idx in idxs]
        print '\t'.join(arr).encode('utf8', 'ignore')


def rsearch(tag):
    rule = tag[0].decode('utf8', 'ignore')
    idx = tag[1] if len(tag) >= 2 else 'all'
    for line in sys.stdin:
        line = line[:-1].decode('utf8', 'ignore')
        arr = line.split('\t')
        v = line if idx == 'all' else arr[int(idx)]
        if re.search(rule, v) is not None: 
            print '\t'.join(arr).encode('utf8', 'ignore')

def rrsearch(tag):
    rule = tag[0].decode('utf8', 'ignore')
    idx = tag[1] if len(tag) >= 2 else 'all'
    for line in sys.stdin:
        line = line[:-1].decode('utf8', 'ignore')
        arr = line.split('\t')
        v = line if idx == 'all' else arr[int(idx)] 
        if re.search(rule, v) is None: 
            print '\t'.join(arr).encode('utf8', 'ignore')

def save(tags):
    fpath = tags[0]
    if fpath.find('/') == -1:
        fpath = 'static/temp/%s' % fpath

    lines = []
    for line in sys.stdin:
        line = line[:-1]
        print line 
        lines.append(line)
    str_2_file('\n'.join(lines), fpath)


def random_line(arg=[0.01, 0]):
    prob_thre, min_thre = arg
    prob_thre, min_thre = float(prob_thre), float(min_thre)
    cache = []
    cnt = 0
    for line in sys.stdin:
        line = line[:-1]
        prob = random.random()
        if prob < float(prob_thre):
            cnt += 1
            print line
        elif len(cache) + cnt < min_thre:
            cache.append(line)
    if cnt < min_thre:
        for line in cache:
            print line


def random_line_limit_num(num):
    num = float(num[0])
    lines = []
    for line in sys.stdin:
        line = line[:-1]
        lines.append(line)
    count = len(lines)
    if count != 0:
        thre = num / count
    else:
        thre = 0
    if thre >= 1:
        thre = 1
    for line in lines:
        prob = random.random()
        if prob < thre:
            print line


def select_idx(tags):
    idxs = [int(idx) for idx in tags]
    for line in sys.stdin:
        line = line[:-1].decode('utf8')
        arr = line.split('\t')
        output = [arr[idx] for idx in idxs]
        print '\t'.join(output).encode('utf8', 'ignore')


def select_field(tags):
    keys = {} 
    for line in sys.stdin:
        line = line[:-1].decode('utf8')
        arr = line.split('\t')
        if keys == {}:
            for i in range(len(arr)):
                keys[arr[i]] = i
        output = [arr[keys[k]] for k in tags]
        print '\t'.join(output).encode('utf8', 'ignore')


def transpose():
    arrs = [] 
    for line in sys.stdin:
        line = line[:-1]
        arr = line.split('\t')
        arrs.append(arr)
    for i in range(len(arrs[0])):
        arr = [v[i] for v in arrs]
        print '\t'.join(arr)

def add_empty_head(tags=[1]):
    n = int(tags[0]) 
    for i in range(n):
        print '' 
    for line in sys.stdin:
        line = line[:-1]
        print line 

def add_head(tags=[]):
    line = '\t'.join(tags)
    print line
    for line in sys.stdin:
        line = line[:-1]
        print line 

def del_head(tags):
    n = int(tags[0]) 
    i = 0 
    for line in sys.stdin:
        i += 1
        if i <= n:
            continue
        line = line[:-1]
        print line 

def print_html_table(tags=['']):
    """
    第一行表示各列的类型,包括image/url/text,可以只填前n列，后面几列默认为text;
    第二行表示各列的列名,该行为空则不展示;
    剩余行为数据内容;
    参数: TK -> 第一行类型第二行字段名; T -> 第一行类型; K -> 第一行类型名
    """
    tag = tags[0]

    print '''
        <link type="text/css" rel="styleSheet"  href="../../static/css/lvisual.css" />
        <script src="../../static/js/echarts-all.js"></script>
        <script src="../../static/js/lvisual.js"></script>
        <script src="../../static/js/lcommon.js"></script>

    '''
    sys.stdout.write('<table class="simpletable">')
    i = 0 if 'K' in tag else 1
    types = [] if 'T' in tag else ['text']
    for line in sys.stdin:
        line = line[:-1]
        arr = line.split('\t')
        if types == []:
            types = arr
            continue
        pre = "<td>"
        suf = "</td>"
        if i == 0 and line.strip() == '':
            i += 1
            continue
        if i == 0:
            pre = "<th>"
            suf = "</th>"
        for j in range(len(arr)):
            v = arr[j]
            if j < len(types) and v.find('http') == 0:
                if types[j] == 'image':
                    arr[j] = '<img width=%d src="%s"/>' % (100, v)
                elif types[j] == 'url':
                    txt = '...%s' % (v[-16:]) if len(v) > 16 else v
                    arr[j] = '<a href="%s" target="_blank">%s</a>' % (v, txt) 

        output = ["%s%s%s" % (pre, v, suf) for v in arr]
        sys.stdout.write('<tr>%s</tr>' % '\t'.join(output))
        i += 1
    sys.stdout.write('</table>')
    

def print_html_field(tags=['TK', 5, 150, 0, 0]):
    """
    第一行表示各列的类型,包括image/url/text,可以只填前n列，后面几列默认为text;
    第二行表示各列的列名,该行为空则不展示;
    剩余行为数据内容;
    参数:
    每一行为一个元素，col_num个元素展现成一行,每个元素展现的宽度为col_width.每个元素的主key是第id_idx个字段, use_boder表示是否展现边框
    """
    tag, col_num, col_width, id_idx, use_boder = 'TK', 5, 150, 1, 0
    if len(tags) >= 1:
        tag = tags[0]
    if len(tags) >= 2:
        col_num = int(tags[1])
    if len(tags) >= 3:
        col_width = int(tags[2])
    if len(tags) >= 4:
        id_idx = int(tags[3])
    if len(tags) >= 5:
        use_boder = int(tags[4])


    print '''
        <link type="text/css" rel="styleSheet"  href="../../static/css/lvisual.css" />
        <script src="../../static/js/echarts-all.js"></script>
        <script src="../../static/js/lvisual.js"></script>
        <script src="../../static/js/lcommon.js"></script>

    '''
    lines = {}   
    keys = [] if 'K' in tag else ['...'] 
    types = [] if 'T' in tag else ['text']
    j = 0
    num = 0
    for line in sys.stdin:
        line = line[:-1]
        arr = line.split('\t')
        if types == []:
            types = arr
            continue
        if keys == []:
            keys = arr
            continue
        if len(keys) < len(arr):
            keys += ['.' for _ in range(len(arr) - len(keys))]
        if len(types) < len(arr):
            types += ['text' for _ in range(len(arr) - len(types))]
        num += 1 
        lines.setdefault(j, [])  
        if len(lines[j]) >= col_num:
            j += 1 
            lines.setdefault(j, [])  
        lines[j].append(arr)
    n = j + 1
    td_boder = ' class="border" ' if use_boder == 1 else 0 
    print '<table class="hiddentable selecting_cell_table" value="%d">' % num 
    for i in range(0, n):  
        arrs = lines[i]
        for j in range(len(keys)):
            td = ['<td %s><span class="none" style="width:%spx" >%s</span></td>' % (td_boder, 50, keys[j])]
            for k in range(len(arrs)):
                vid = arrs[k][id_idx]  
                v = arrs[k][j]
                elem = v 
                if types[j] == 'image':
                    elem = '<img class="red selecting_cell" value="%s" width=%d src="%s" onclick="select_cell(this, \'selecting_cell_label\')"/>' % (vid, col_width, v)
                elif types[j] == 'url':
                    txt = '...%s' % (v[-16:]) if len(v) > 16 else v
                    elem = '<a class="selecting_cell" value="%s" style="display:block;width:%spx" href="%s" target="_blank" onclick="select_cell(this, \'selecting_cell_label\')">%s</a>' % (vid, col_width, v, txt) 
                else:
                    elem = '<span class="none selecting_cell" value="%s" style="width:%spx" onclick="select_cell(this, \'selecting_cell_label\')" >%s</span>' % (vid, col_width, v)
                td.append('<td %s >%s</td>' % (td_boder, elem)) 
            print '<tr>%s</tr>' %  ''.join(td)
    print '</table>'
    print '<hr><label id = "selecting_cell_label" style="width: 1000px; height: 100px;"></label>'


if __name__ == "__main__":
    func_arg = sys.argv[1]
    arr = func_arg.split('____')
    try:
        if len(arr) == 1:
            func = arr[0]
            eval(func)()
        else:
            func = arr[0]
            arg = [v.decode('utf8', 'ignore') for v in arr[1:]]
            eval(func)(arg)
    except Exception as e: 
        print "error:", e
        cat()
