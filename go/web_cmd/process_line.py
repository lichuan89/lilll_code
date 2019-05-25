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
from lprocess_line import *


def quote_url(tag):
    idx = int(tag[0])
    is_quote = tag[1]
    for line in sys.stdin:
        line = line[:-1]
        arr = line.split('\t')
        arr[idx] = urllib.quote(arr[idx]) if is_quote == 'quote' else urllib.unquote(arr[idx])
        print '\t'.join(arr)


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


def select_fields(tags):
    idxs = [int(idx) for idx in tags]
    for line in sys.stdin:
        line = line[:-1].decode('utf8')
        arr = line.split('\t')
        output = [arr[idx] for idx in idxs]
        print '\t'.join(output).encode('utf8', 'ignore')

def add_first_line(tags):
    line = '\t'.join(tags)
    print line
    for line in sys.stdin:
        line = line[:-1]
        print line 

def print_html_table():
    print '<table class="simpletable">'
    i = 0
    for line in sys.stdin:
        line = line[:-1]
        arr = line.split('\t')
        pre = "<td>"
        suf = "</td>"
        if i == 0:
            pre = "<th>"
            suf = "</th>"
        output = ["%s%s%s" % (pre, v, suf) for v in arr]
        print "<tr>%s</tr>" % '\t'.join(output)
        i += 1
    print '</table>'
    
def print_html_attr(tags):
    types = dict([(i, tags[i]) for i in range(0, len(tags))])
    print '<div>'
    j = 0
    for line in sys.stdin:
        line = line[:-1]
        line = line.decode('utf8', 'ignore')
        fields = line.split('\t')
        keys = [fields[i] for i, t in types.items() if t == 'key']
        imgs = [fields[i] for i, t in types.items() if t == 'image']
        urls = [fields[i] for i, t in types.items() if t == 'url']
        texts = [fields[i] for i, t in types.items() if t == 'text']
        key = keys[0] if len(keys) > 0 else j
        j += 1
        key_html = '<label>%s<input type="checkbox" name="show_analysis_data_id" value="%s" onclick="change()"/></label>' % (key, key)

        url_html = '|'.join(['<a href="%s">%s</a>' % (url, url[:15]) for url in urls])

        text_html = '|'.join(['<span width=100>%s</span>' % text for text in texts])

        img_html = []
        for v in imgs: 
            red = ' style="border:3px solid #ff0000" ' if v.find("select_color=1") == -1 else ' style="border:3px solid #000000" '
            img_html.append('<img src="%s" width=150  %s />' % (v, red))
        img_html = '\t'.join(img_html)
        output = [key_html, img_html, url_html, text_html, '<hr>'] 
        print '\t'.join(output).encode('utf8', 'ignore')
    print '<hr><label id = "show_analysis_data_label" style="width: 1000px; height: 100px;"></label>'
    print '</div>'


def print_html_image(tag=[5, 150]):
    """  
    mapper_prod_stream
    """  
    col_num, width = int(tag[0]), int(tag[1])
    lines = {}   
    j = 0  
    for line in sys.stdin:
        line = line[:-1]
        image, name, idx = line.split('\t')

        lines.setdefault(j, [])  
        if len(lines[j]) >= col_num:
            j += 1 
        lines.setdefault(j, [])  
        lines[j].append((image, name, idx))
    print '''  
        <style type="text/css">
        table td{ height:10px; width:%d; font-size:5px}
        </style>
        ''' % width
    print '<table border="0.1" border-spacing:0px  cellspacing="0" cellpadding="0">'
    for i in range(0, j + 1):  
        arr = lines[i]
        th = ['<th><img src="%s" width=%d style="border:3px solid #ff0000" /></th>' % (v[0], width) for v in arr]   
        td = ['<td><label><input type="checkbox" name="show_analysis_data_id" value="%s" onclick="change()"/>%s</label></td>' % (v[2], v[1]) for v in arr]   
        print '<tr>%s</tr> <tr>%s</tr>\n' % (''.join(th), ''.join(td))
    print '</table>'
    print '<hr><label id = "show_analysis_data_label" style="width: 1000px; height: 100px;"></label>'


if __name__ == "__main__":
    func_arg = sys.argv[1]
    arr = func_arg.split('____')
    try:
        if len(arr) == 1:
            func = arr[0]
            eval(func)()
        else:
            func = arr[0]
            arg = arr[1:]
            eval(func)(arg)
    except Exception as e: 
        cat()
