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
from lcommon import expand_json
from lcommon import expand_pair
from lcommon import unexpand_pair
from lcommon import unexpand_json 
from lcommon import file_2_str 
from lprocess_line import *


def help():
    data_fpath = './static/example/example_data.txt' 
    data = file_2_str(data_fpath)
    print '1. 例子数据:\n%s\n%s' % (data_fpath, data)
    print ''
    print '''2. 例子指令:
        shell指令\t:awk -F"\t" '{if($8~/^[0-9.]+$/) {s+=$8;n+=1}}END{print "平均价格:"s/n;}'
        自定义sh指令\t#restart
        选择列\tselect_idx____0____6____1____7
        随机排序\tshuffle|:head -6
        纯文本表格\thtml_table
        图文表格\tselect_idx____0____6____1____2____3____4____7|add_head____image____url____txt____txt|html_table____KT
        图文卡片\tselect_idx____0____6____1____2____3____4____7|add_head____image____url____txt____txt|html_card____KT____4
        曲线\t:head -4|select_idx____1____9____10____11____12____13____14____15|chart_curve
        饼图\t:head -5|select_idx____1____9____10____11____12____13____14____15|del_head____1|add_head____sale____1____2____3____4____5____6____7|transpose|chart_pie
        树图\tselect_idx____3____4____5|del_head____1|chart_tree____field
        运行管道命令\t$
'''

    print ''
    print '3. py指令全集:'
    files = ['lprocess_line.py', 'process_line.py']
    for f in files:
        data = file_2_str(f)    
        fs = re.findall('def ([a-zA-Z0-9_]+[(].*?[)])', data)
        print '\t' + '\n\t'.join(fs) 
    
    print ''
    print '4. sh指令全集:'
    files = ['cmd.sh']
    for f in files:
        data = file_2_str(f)    
        fs = re.findall('if [[][[] [$]cmd == "(.*?)" []][]]; then', data)
        print '\t' + '\n\t'.join(fs) 
    
    print ''
    print '4. 图片处理指令:'
    print '    xxxxxxxxxxxxxx'

    print '5. 常用复杂命令:'
    print '''
        #image_ls____c|#image_muzzy____m____0|#image_sobel____s____-1|#image_otsu____o____0|mirror|#image_fmt____5|html_table____T 
    '''

def process_field(tags, func):
    """
    tags = [序号(a表示全行), 扩展/替换, 解码(none表示不操作), 编码后处理(none表示不操作), 最终编码]
    """
    idx = int(tags[0]) if tags[0] != 'a' else tags[0]
    tag = tags[1] if len(tags) >= 2 else 'append'
    decode = tags[2] if len(tags) >= 3 else 'utf8'
    encode = tags[3] if len(tags) >= 4 else 'none'
    last_encode = tags[4] if len(tags) >= 5 else decode
    for line in sys.stdin:
        line = line[: -1]
        if decode != 'none': 
            line = line.decode(decode, 'ignore')
        if idx == 'a':
            arr = [line]
            v = func(line)
        else:
            arr = line.split('\t')
            v = arr[idx]
            if encode != 'none':
                v = v.encode(encode, 'ignore')
            v = func(v)
            if last_encode != 'none':
                v = v.encode(last_encode, 'ignore')
        if tag == 'replace':
            arr[idx if idx != 'a' else 0] = v
        else:
            arr.append(v)
        output = '\t'.join(arr)
        print output

def shuffle():
    lines = []
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        lines.append(line)
    random.shuffle(lines)
    for line in lines:
        print line


def parse_json(tags):
    idx = int(tags[0])
    keys = tags[1].split('^')
    pair_key = tags[2] if len(tags) >= 4 else ''
    pair_val = tags[3] if len(tags) >= 4 else ''
    is_simple_key = (tags[4] == '1') if len(tags) >= 5 else True
    i = 0 
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        line = line.decode('utf8', 'ignore')
        arr = line.split('\t')
        obj = str_2_json(arr[idx])
        if obj is None:
            continue
        if pair_key != '':
            obj = expand_pair(obj, pair_key, pair_val)
        kvs, kns = expand_json(obj, sep="____", is_simple_key=is_simple_key)
        vs = [kvs[k] if k in kvs else '' for k in keys]
        output = arr[: idx] + vs + arr[idx + 1:]
        print '\t'.join(output).encode('utf8', 'ignore')


def base64_field(tags=['a', 'replace', 'none', 'none', 'none']):
    process_field(tags, lambda v: base64.b64encode(v))
    

def unbase64_field(tags=['a', 'replace', 'none']):
    process_field(tags, lambda v: base64.b64decode(v))

def encode_field(tags=['a', 'replace', 'none']):
    process_field(tags, lambda v: v)
 
def quote_field(tags):
    process_field(tags, lambda v: urllib.quote(v))


def unquote_field(tags):
    process_field(tags, lambda v: urllib.unquote(v))


def gbk_field(tags=['a', 'replace', 'utf8', 'none', 'gbk']):
    process_field(tags, lambda v: v)

def utf8_field(tags=['a', 'replace', 'gbk', 'none', 'utf8']):
    process_field(tags, lambda v: v)

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


def split_field(tags=['\t']):
    sep = tags[0]
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[: -1]
        line = line.decode('utf8', 'ignore')
        arr = re.split(sep, line) 
        print '\t'.join(arr).encode('utf8', 'ignore')  


def split_line(tags=['\t']):
    sep = tags[0]
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[: -1]
        line = line.decode('utf8', 'ignore')
        arr = re.split(sep, line) 
        for v in arr:
            print v.encode('utf8', 'ignore')  


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


def add_index(tags=[0]):
    begin = int(tags[0])
    for line in sys.stdin:
        line = line[:-1]
        arr = line.split('\t')
        output = [str(begin)] + arr
        begin += 1
        print '\t'.join(output)

def add_head_index(tags=[0]):
    begin = int(tags[0])
    is_begin = True
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[: -1]
            if is_begin:
                cnt = len(line.split('\t'))
                arr = [str(v) for v in range(begin, begin + cnt)]
                print '\t'.join(arr)
                is_begin = False
        print line 


def swap_row(tags):
    lines = []
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[: -1]
        lines.append(line)

    for i in range(0, len(tags), 2):
        j, k = int(tags[i]), int(tags[i + 1])
        lines[j], lines[k] = lines[k], lines[j]
    for line in lines:
        print line        


def swap_col(tags):
    lines = []
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[: -1]
        arr = line.split('\t')
        for i in range(0, len(tags), 2):
            j, k = int(tags[i]), int(tags[i + 1])
            arr[j], arr[k] = arr[k], arr[j]
        print '\t'.join(arr) 


def add_const(tags):
    const = tags[0]
    idx = int(tags[1]) if len(tags) > 1 else 0
    for line in sys.stdin:
        line = line[:-1].decode('utf8')
        arr = line.split('\t')
        output = arr[: idx] + [const] + arr[idx: ]
        print '\t'.join(output).encode('utf8')
    
        

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
        if line[-1] == '\n':
            line = line[:-1]
        arr = line.split('\t')
        arrs.append(arr)
    for i in range(len(arrs[0])):
        arr = [v[i] for v in arrs]
        print '\t'.join(arr)

def mirror():
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        arr = line.split('\t')
        print '\t'.join(reversed(arr))

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


def html_table(tags=['']):
    """
    第一行表示各列的类型,包括image/url/text,可以只填前n列，后面几列默认为text;
    第二行表示各列的列名,该行为空则不展示;
    剩余行为数据内容;
    参数: TK -> 第一行类型第二行字段名; T -> 第一行类型; K -> 第一行类型名
    """
    tag = tags[0]
    red_idx = int(tags[1]) if len(tags) > 1 else None 
    print '''
        <link type="text/css" rel="styleSheet"  href="../../static/css/lvisual.css" />
        <script src="../../static/js/echarts-all.js"></script>
        <script src="../../static/js/cmd.js"></script>
        <script src="../../static/js/lcommon.js"></script>

    '''
    sys.stdout.write('<table class="simpletable">')
    i = 0 if 'K' in tag else 1
    types = [] if 'T' in tag else ['text']
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        arr = line.split('\t')
        if types == []:
            types = arr
            continue
        pre = '<td style="word-wrap:break-word; white-space:normal; word-break:break-all;min-width:70px">'
        suf = "</td>"
        if i == 0 and line.strip() == '':
            i += 1
            continue
        if i == 0:
            pre = "<th>"
            suf = "</th>"
        if pre == "<th>" or len(arr) <= red_idx or (red_idx is None or arr[red_idx] == '0' or arr[red_idx] == '' or arr[red_idx] == '-'):
            tr = '<tr>'
        else:
            pre = '<td style="background:#ffce9f;word-wrap:break-word; white-space:normal; word-break:break-all;">'
            tr = '<tr style="border:solid red">' 
        for j in range(len(arr)):
            v = arr[j]
            if j < len(types) and (v.find('http') == 0 or v.find('/') != -1):
                if types[j] == 'image':
                    arr[j] = '<img width=%d src="%s"/>' % (100, v)
                elif types[j] == 'url':
                    txt = '...%s' % (v[-16:]) if len(v) > 16 else v
                    arr[j] = '<a href="%s" target="_blank">%s</a>' % (v, txt) 


        output = ["%s%s%s" % (pre, v, suf) for v in arr]
        sys.stdout.write('%s%s</tr>' % (tr, '\t'.join(output)))
        i += 1
    sys.stdout.write('</table>')
    

def html_card(tags=['TK', 5, 150, 0, 0]):
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
        <script src="../../static/js/cmd.js"></script>
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
                vid = arrs[k][id_idx] if id_idx < len(arrs[k]) else 0 
                v = arrs[k][j]
                elem = v 
                if types[j] == 'image':
                    elem = '<img class="red selecting_cell" value="%s" width=%d src="%s" onclick="select_cell(this, \'selecting_cell_label\')"/>' % (vid, col_width, v)
                elif types[j] == 'url':
                    txt = '...%s' % (v[-16:]) if len(v) > 16 else v
                    elem = '<a class="selecting_cell" value="%s" style="display:block;width:%spx" href="%s" target="_blank" onclick="select_cell(this, \'selecting_cell_label\')">%s</a>' % (vid, col_width, v, txt) 
                else:
                    elem = '<span class="none selecting_cell" value="%s" style="width:%spx;word-wrap:break-word; white-space:normal; word-break:break-all;" onclick="select_cell(this, \'selecting_cell_label\')" >%s</span>' % (vid, col_width, v)
                td.append('<td %s >%s</td>' % (td_boder, elem)) 
            print '<tr>%s</tr>' %  ''.join(td)
    print '</table>'
    print '<hr><label id = "selecting_cell_label" style="width: 1000px; height: 100px;"></label>'


def chart_curve():
    lines = []
    for line in sys.stdin:
        line = line[:-1].decode('utf8')
        arr = line.split('\t')
        lines.append(arr)

    option = {
            "title":{
                "text":"title0"
            },
            "tooltip":{
                "trigger":"axis"
            },
            "legend":{
                "data":[
                    "1",
                    "2"
                ]
            },
            "toolbox":{
                "show":1,
                "feature":{
                    "mark":{
                        "show":1
                    },
                    "dataView":{
                        "show":1,
                        "readOnly":0
                    },
                    "magicType":{
                        "show":1,
                        "type":[
                            "line",
                            "bar",
                            "stack",
                            "tiled"
                        ]
                    },
                    "restore":{
                        "show":1
                    },
                    "saveAsImage":{
                        "show":1
                    }
                }
            },
            "calculable":1,
            "xAxis":[
                {
                    "type":"category",
                    "boundaryGap":0,
                    "data":[
                        "A",
                        "B",
                        "C"
                    ]
                }
            ],
            "yAxis":[
                {
                    "type":"value"
                }
            ],
            "series":[
                {
                    "name":"1",
                    "type":"line",
                    "smooth":1,
                    "itemStyle":{
                        "normal":{
                            "areaStyle":{
                                "type":"default"
                            }
                        }
                    },
                    "data":[
                        "20.1",
                        "30.2",
                        "50"
                    ]
                },
                {
                    "name":"2",
                    "type":"line",
                    "smooth":1,
                    "itemStyle":{
                        "normal":{
                            "areaStyle":{
                                "type":"default"
                            }
                        }
                    },
                    "data":[
                        "60.3",
                        "10",
                        "100"
                    ]
                }
            ]
        }
    option = {
            "title":{
                "text":"aaa"
            },
            "tooltip":{
                "trigger":"axis"
            },
            "legend":{
                "data":[
                    "1",
                    "2"
                ]
            },
            "toolbox":{
                "show":1,
                "feature":{
                    "mark":{
                        "show":1
                    },
                    "dataView":{
                        "show":1,
                        "readOnly":0
                    },
                    "magicType":{
                        "show":1,
                        "type":[
                            "line",
                            "bar",
                            "stack",
                            "tiled"
                        ]
                    },
                    "restore":{
                        "show":1
                    },
                    "saveAsImage":{
                        "show":1
                    }
                }
            },
            "calculable":1,
            "xAxis":[
                {
                    "type":"category",
                    "boundaryGap":0,
                    "data":[
                        "A",
                        "B",
                        "c"
                    ]
                }
            ],
            "yAxis":[
                {
                    "type":"value"
                }
            ],
            "series":[
                {
                    "name":"1",
                    "type":"line",
                    "smooth":1,
                    "itemStyle":{
                        "normal":{
                            "areaStyle":{
                                "type":"default"
                            }
                        }
                    },
                    "data":[
                        "20.1",
                        "30.2",
                        "100"
                    ]
                },
                {
                    "name":"2",
                    "type":"line",
                    "smooth":1,
                    "itemStyle":{
                        "normal":{
                            "areaStyle":{
                                "type":"default"
                            }
                        }
                    },
                    "data":[
                        "60.3",
                        "30",
                        "50"
                    ]
                }
            ]
        }
    title = lines[0][0]
    row_names = [arr[0] for arr in lines[1: ]]
    col_names = lines[0][1: ]
    option['title']['text'] = title 
    option['legend']['data'] = row_names
    option['xAxis'][0]['data'] = col_names
    series = [
        {
            "name": lines[i][0],
            "type":"line",
            "smooth":1,
            "itemStyle": {
                "normal":{
                    "areaStyle":{
                        "type":"default"
                    }
                }
            },
            "data":lines[i][1: ]
        } for i in range(1, len(lines))
    ]
    option['series'] = series 
    print json_2_str(option).encode('utf8', 'ignore') 
    print_chart_js()

def chart_scatter():
    lines = []
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        line = line.decode('utf8')
        arr = line.split('\t')
        lines.append(arr)

    option = {
            "title":{
                "text":u"男性女性身高体重分布",
                "subtext":"-"
            },
            "tooltip":{
                "trigger":"axis",
                "showDelay":0,
                "axisPointer":{
                    "show":1,
                    "type":"cross",
                    "lineStyle":{
                        "type":"dashed",
                        "width":1
                    }
                }
            },
            "legend":{
                "data":[
                    u"女性",
                    u"男性",
                    u"中性"
                ]
            },
            "toolbox":{
                "show":1,
                "feature":{
                    "mark":{
                        "show":1
                    },
                    "dataZoom":{
                        "show":1
                    },
                    "dataView":{
                        "show":1,
                        "readOnly":0
                    },
                    "restore":{
                        "show":1
                    },
                    "saveAsImage":{
                        "show":1
                    }
                }
            },
            "xAxis":[
                {
                    "type":"value",
                    "scale":1,
                    "axisLabel":{
                        "formatter":"{value} "
                    }
                }
            ],
            "yAxis":[
                {
                    "type":"value",
                    "scale":1,
                    "axisLabel":{
                        "formatter":"{value} "
                    }
                }
            ],
            "series":[
                {
                    "name":u"女性",
                    "type":"scatter",
                    "data":[
                        [
                            161.2,
                            51.6
                        ],
                        [
                            167.5,
                            59
                        ],
                        [
                            159.5,
                            49.2
                        ]
                    ],
                    "markPoint":{
                        "data":[
                            {
                                "type":"max",
                                "name":u"最大值"
                            },
                            {
                                "type":"min",
                                "name":u"最小值"
                            }
                        ]
                    },
                    "markLine":{
                        "data":[
                            {
                                "type":"average",
                                "name":u"平均值"
                            }
                        ]
                    }
                },
                {
                    "name":u"中性",
                    "type":"scatter",
                    "data":[
                        [
                            161.2,
                            51.6
                        ],
                        [
                            167.5,
                            59
                        ],
                        [
                            159.5,
                            49.2
                        ]
                    ],
                    "markPoint":{
                        "data":[
                            {
                                "type":"max",
                                "name":u"最大值"
                            },
                            {
                                "type":"min",
                                "name":u"最小值"
                            }
                        ]
                    },
                    "markLine":{
                        "data":[
                            {
                                "type":"average",
                                "name":u"平均值"
                            }
                        ]
                    }
                },
                {
                    "name":u"男性",
                    "type":"scatter",
                    "data":[
                        [
                            174,
                            65.6
                        ],
                        [
                            175.3,
                            71.8
                        ],
                        [
                            193.5,
                            80.7
                        ],
                        [
                            180.3,
                            83.2
                        ]
                    ],
                    "markPoint":{
                        "data":[
                            {
                                "type":"max",
                                "name":u"最大值"
                            },
                            {
                                "type":"min",
                                "name":u"最小值"
                            }
                        ]
                    },
                    "markLine":{
                        "data":[
                            {
                                "type":"average",
                                "name":u"平均值"
                            }
                        ]
                    }
                }
            ]
        }
    categorys = [lines[i][0] for i in range(0, len(lines), 2)]
    option['title']['text'] = '-' 
    option['legend']['data'] = categorys 
    series = [[[lines[i][j], lines[i + 1][j]] for j in range(1, len(lines[i]))] for i in range(0, len(lines), 2)]
    option['series'] = [
            {
                "data":series[i],
                "type":"scatter",
                "name":categorys[i],
                "markPoint":{
                    "data":[
                        {
                            "type":"max",
                            "name":u"最大值"
                        },
                        {
                            "type":"min",
                            "name":u"最小值"
                        }
                    ]
                },
                "markLine":{
                    "data":[
                        {
                            "type":"average",
                            "name":u"平均值"
                        }
                    ]
                }
            } for i in range(len(categorys))]


 
    print json_2_str(option).encode('utf8', 'ignore') 
    print_chart_js()


def chart_pie():
    lines = []
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        line = line.decode('utf8')
        arr = line.split('\t')
        lines.append(arr)

    option = {
            "timeline":{
                "data":[
                    "2015",
                    "2016"
                ]
            },
            "options":[
                {
                    "series":[
                        {
                            "radius":"50%",
                            "type":"pie",
                            "name":"title0",
                            "data":[
                                {
                                    "name":"name1",
                                    "value":20.1
                                },
                                {
                                    "name":"name2",
                                    "value":30.2
                                }
                            ],
                            "center":[
                                "50%",
                                "45%"
                            ]
                        }
                    ],
                    "toolbox":{
                        "feature":{
                            "magicType":{
                                "type":[
                                    "pie",
                                    "funnel"
                                ],
                                "option":{
                                    "funnel":{
                                        "funnelAlign":"left",
                                        "x":"25%",
                                        "max":1700,
                                        "width":"50%"
                                    }
                                },
                                "show":1
                            },
                            "restore":{
                                "show":1
                            },
                            "dataView":{
                                "readOnly":0,
                                "show":1
                            },
                            "saveAsImage":{
                                "show":1
                            },
                            "mark":{
                                "show":1
                            }
                        },
                        "show":1
                    },
                    "legend":{
                        "data":[
                            "name1",
                            "name2"
                        ]
                    },
                    "tooltip":{
                        "trigger":"item",
                        "formatter":"{a} <br/>{b} : {c} ({d}%)"
                    },
                    "title":{
                        "text":"title0",
                        "subtext":""
                    }
                }
            ]
        }
    title = lines[0][0]
    col_names = [arr[0] for arr in lines[1: ]] 
    option['timeline']['data'] = col_names
    option['options'][0]['title']['text'] = title
    series = [[{"name": lines[0][col], "value": arr[col]} for col in range(1, len(arr))] for arr in lines[1: ]]
    series = [
        {
            "name": title,
            "type":"pie",
            "center":[
                "50%",
                "45%"
            ],
            "radius":"50%",
            "data": serie,
        } for serie in series
    ]
    option['options'][0]['legend']['data'] = lines[0][1:] 
    option['options'][0]['series'] = [series[0]]
    for serie in series[1: ]:
        option['options'].append({'series': [serie]}) #['series'] = series[1: ]
    print json_2_str(option).encode('utf8', 'ignore')
    print_chart_js()



def chart_tree(tags=["json"]):
    # https://echarts.baidu.com/examples/editor.html?c=tree-basic&theme=light
    tag = tags[0] # json or field

    lines = []
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        line = line.decode('utf8')
        arr = line.split('\t')
        lines.append(arr)

    data = {
        "name":"flare",
        "children":[
            {
                "name":"analytics",
                "children":[
                    {
                        "collapsed": 0,
                        "name":"cluster",
                        "children":[
                            {
                                "name":"AgglomerativeCluster",
                                "value":3938
                            },
                            {
                                "name":"CommunityStructure",
                                "value":3812
                            },
                            {
                                "name":"HierarchicalCluster",
                                "value":6714
                            },
                            {
                                "name":"MergeEdge",
                                "value":743
                            }
                        ]
                    }
                ]
            }
        ]
    }
    obj = {}
    if tag == 'field':
        for arr in lines:
            cnt = len(arr)
            for i in range(len(arr)):
                if arr[cnt - i - 1].strip() != '':
                    cnt = cnt - i
                    break
            #obj['\t'.join(arr[: -1])] = arr[-1]
            obj['\t'.join(arr[: cnt])] = '' 
        obj = unexpand_json(obj, sep='\t')
    elif tag == 'json':
        s = ' '.join([' '.join(arr) for arr in lines])
        obj = str_2_json(s)
    def expand_val(obj):
        if type(obj) == list:
            obj = [expand_val(v) for v in obj]
        elif type(obj) == dict:
            for k, v in obj.items():
                obj[k] = expand_val(v)
        elif obj != '':
            obj = {unicode(obj): 0}
        return obj
    obj = expand_val(obj)
    data = unexpand_pair(obj, pair_key='name', pair_val='value', children_val='children', unit_dict={'collapsed': 0}) 
    if len(data) == 1: 
        data = data[0]
    else:
        data = {"name": "", "children": data, "collapsed": 0}

    option = {
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove"
        },
        "series": [
            {
                "type": "tree",

                "data": [data],

                "top": "1%",
                "left": "7%",
               "bottom": "1%",
                "right": "20%",

                "symbolSize": 7,

                "label": {
                    "normal": {
                        "position": "left",
                        "verticalAlign": "middle",
                        "align": "right",
                        "fontSize": 9
                    }
                },

                "leaves": {
                    "label": {
                        "normal": {
                            "position": "right",
                            "verticalAlign": "middle",
                            "align": "left"
                        }
                    }
                },

                "expandAndCollapse": True,
                "animationDuration": 550,
                "animationDurationUpdate": 750
            }
        ]
    }
    print json_2_str(option).encode('utf8', 'ignore')
    print_chart_js()
    print '<script src="../../static/js/echarts.4.2.1.js"></script>'


def print_chart_js():
    html = '''
        <script src="../../static/js/echarts-all.2.2.7.js"></script>
        <script>
            window.onload = function() {
                var dom = document.getElementById('body');
                var option = JSON.parse(dom.innerHTML.split("\\n")[0]);
                console.log("json:", option)
                dom.innerHTML = ""; 
        
                var chart_dom =document.createElement("div");
                chart_dom.setAttribute('style', 'width: 800px;height:400px;');
        
                dom.appendChild(chart_dom)
                var myChart = echarts.init(chart_dom);
                myChart.setOption(option);
            };
        </script>
        <body id="body">
        </body>
    '''
    print html 

if __name__ == "__main__":
    func_arg = sys.argv[1].strip()
    arr = func_arg.split('____')
    if True: #try:
        if len(arr) == 1:
            func = arr[0]
            eval(func)()
        else:
            func = arr[0]
            arg = [v.decode('utf8', 'ignore') for v in arr[1:]]
            eval(func)(arg)
    #except Exception as e: 
    #    cat()
    #    print >> sys.stderr, "failed to parse. err:", e 
