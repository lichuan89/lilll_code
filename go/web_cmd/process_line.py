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
