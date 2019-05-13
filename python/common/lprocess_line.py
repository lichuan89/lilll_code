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


def cat():
    for line in sys.stdin:
        print line[: -1]

def match_str(tags):
    for line in sys.stdin:
        line = line[:-1]
        for tag in tags:
            if line.find(tag) != -1:
                print line 
                break


if __name__ == "__main__":
    func_arg = sys.argv[1]
    arr = func_arg.split('____')
    if len(arr) == 1:
        func = arr[0]
        eval(func)()
    else:
        func = arr[0]
        arg = arr[1:]
        eval(func)(arg)
