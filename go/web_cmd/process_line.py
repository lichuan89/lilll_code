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
from lcommon import str_2_json
from lcommon import json_2_str
from lprocess_line import *

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
    
def print_html_image(tags):
    types = dict([(i, tags[i]) for i in range(0, len(tags))])
    print '''
    <script type="text/javascript">
        function change(){
            var show_analysis_data_label = document.getElementById("show_analysis_data_label");
            show_analysis_data_label.innerHTML = '';
            var c_u_ids = document.getElementsByName("show_analysis_data_id");
            var str = '选中prodId: ';
            hit = 0
            for (var i = 0; i < c_u_ids.length; i++) {
                if (c_u_ids[i].checked) {
                    str += c_u_ids[i].value;
                    str += "    ";
                    hit += 1;
                }
            }
            p = hit / c_u_ids.length
            p = '' + p
            hit = '' + hit
            n = '' + c_u_ids.length
            str = '总数: ' + n + '<br>选中: ' + hit + '<br>命中率: ' + p + '<br>' + str
            show_analysis_data_label.innerHTML = str;
        }
    </script>
    <div>
    '''
    i = 0
    for line in sys.stdin:
        line = line[:-1]
        line = line.decode('utf8', 'ignore')
        fields = line.split('\t')
        keys = [fields[i] for i, t in types.items() if t == 'key']
        imgs = [fields[i] for i, t in types.items() if t == 'image']
        urls = [fields[i] for i, t in types.items() if t == 'url']
        texts = [fields[i] for i, t in types.items() if t == 'text']
        key = keys[0] if len(keys) > 0 else i
        i += 1
        key_html = '<label>%s<input type="checkbox" name="show_analysis_data_id" value="%s" onclick="change()"/></label>' % (key, key)

        url_html = '|'.join(['<a href="%s">%s</a>' % (url, url[:15]) for url in urls])

        text_html = '|'.join(['<span width=100>%s</span>' % text for text in texts])

        img_html = []
        for v in imgs: 
            red = ' style="border:3px solid #ff0000" ' if v.find("select_color=1") != -1 else ' style="border:3px solid #000000" '
            img_html.append('<img src="%s" width=150  %s />' % (v, red))
        img_html = '\t'.join(img_html)
        output = [key_html, img_html, url_html, text_html, '<hr>'] 
        print '\t'.join(output).encode('utf8', 'ignore')
    print '<hr><label id = "show_analysis_data_label" style="width: 1000px; height: 100px;"></label>'
    print '</div>'


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
