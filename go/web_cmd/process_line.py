#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2017/01/01  
    @note   实现python多进程处理输入流的函数
"""

import sys
sys.path.append("../../python/common/")
import traceback
import base64
import re
import time
import datetime
import random
import urllib
from lcommon import log 
from lcommon import str_2_json
from lcommon import json_2_str
from lcommon import str_2_file
from lcommon import get_domains 
from lcommon import expand_json
from lcommon import expand_pair
from lcommon import unexpand_pair
from lcommon import find_map 
from lcommon import unexpand_json 
from lcommon import file_2_str 
from lcommon import smart_str_list
from lprocess_line import *
from limg_process_line import limg 
import lprocess_line


def history(tags=['100']):
    ## show history cmds. cmd: history, history__50
    n = int(tags[0]) if len(tags) > 0 else 30
    fpath = 'static/temp/cmd.history.txt'
    data = file_2_str(fpath)
    data = list(reversed(data.split('\n')))[: n + 1]
    for line in data:
        arr = line.split('\t')
        if len(arr) < 3:
            continue
        cmd, fpath, context = arr[: 3] 
        print cmd
        print fpath
        print context
        print  


def example(tags=[]):
    data_fpath = './static/example/example_data.txt' 
    data = file_2_str(data_fpath)
    print '1. 例子数据:\n%s\n%s' % (data_fpath, data)
    print ''
    print '''2. 例子指令:
        shell指令\t:awk -F"\\t" '{if($8~/^[0-9.]+$/) {s+=$8;n+=1}}END{print "平均价格:"s/n;}'
        自定义sh指令\t#restart
        输入数据中加入指令\taddHead__r__cols<2UNDER>r<2UNDER>0<2UNDER>1|addHead__r__#!lcmd
        选择列\tcols__r__0__1
        随机排序\tshuffle
        纯文本表格\thtml_table
        图文表格\tcols__r__0__6__1__2__3__4__7__-1|rowIdx__0b__序号|addHead__r__txt__image__url__txt__txt|html_table__KT__-1
        图文卡片\tcols__r__0__6__1__2__3__4__7__-1|rowIdx__0b__序号|addHead__r__txt__image__url__txt__txt|html_card__KT__5__150__4
        曲线\t:head -4|cols__r__1__9__10__11__12__13__14__15|chart_curve
        饼图\t:head -5|cols__r__1__9__10__11__12__13__14__15|delHead__r__1|colIdx__r__sales|transpose|chart_pie
        树图\tcols__r__3__4__5|chart_tree__field
        处理图像\tselect_idx____0____1|#image_crawl____c|select_idx____0____2|#image_light____l____-2____2.3____20|#image_ycrcb____y____-2|#image_skin____skin____-2|#image_sobel____s____-2|#image_expand____e____-2|#image_rotate____r____-2____60|#image_muzzy____m____-2|mirror|#image_fmt____10|add_head____id____name____ori____light____ycrcb____skin____sobel____expand____rorate____muzzy____muzzy_score|swap_row____0____1|html_table____KT
        上一次命令的输出作为输入\t|
        上上次命令的输出作为输入\t||
        多进程选择列\tcols__rm__0__1
        运行管道命令\t$
        抓图像/网页放入文件夹\tcols__r__0__1|:grep http|curl__0i__path1__.bmp
        处理图像(例如缩放图像)\tcols__r__0__1|:grep http|curl__0i__path1__.bmp|limg__1i__resize__path2__.bmp__300__100

'''


def help(tags=[]):

    print "py指令集:"
    files = ['../../python/common/lprocess_line.py', 'process_line.py']
    for f in files:
        data = file_2_str(f)    
        fs = re.findall('def ([a-zA-Z0-9_]+[(].*?[)]:[^:]+##[^\n]+)', data)
        fs = ['\t' + i.replace('\n', '\t').strip() for i in fs]
        print '\n\n'.join(fs) 
    
    print ''
    print 'image指令集:'
    files = ['limg_process_line.py']
    for f in files:
        data = file_2_str(f)    
        fs = re.findall('def ([a-zA-Z0-9_]+[(].*?[)]:[^:]+##[^\n]+)', data)
        fs = ['\t' + i.replace('\n', '\t').strip() for i in fs]
        print '\n\n'.join(fs) 

    print ''
    print 'sh指令集:'
    files = ['cmd.sh']
    for f in files:
        data = file_2_str(f)    
        fs = re.findall('(##[^#]+?##[^\n]+)', data)
        fs = ['\t' + i.replace('\n', '\t').strip() for i in fs]
        print '\n\n'.join(fs) 
    
    print ''
    print '简写指令集:'
    f = 'cmd_map.list' 
    data = file_2_str(f)
    for s in data.split('\n'):
        print '\t' + s

def shuffle(tags=[]):
    ## shuffle lines. cmd: shuffle, shuffle__4
    n = int(tags[0]) if len(tags) > 0 else None
    lines = []
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        lines.append(line)
    random.shuffle(lines)
    i = 0
    for line in lines:
        print line
        i += 1
        if n is not None and i >= n:
            break


def expand_fields(tags):
    idx = int(tags[0])
    sep = tags[1] if len(tags) <= 2 else '____' 
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        line = line.decode('utf8')
        arr = line.split('\t')
        if idx < len(arr): 
            fields = arr[idx].split(sep)
            for i in range(len(fields)):
                arr[idx] = '%s\t%s' % (i, fields[i])
                print '\t'.join(arr).encode('utf8', 'ignore')
        else:
            print line

def expand_pairs(tags):
    # 将 k1 sep v1 sep k2 sep v2 ... 分解到多行
    idx = int(tags[0])
    pair_sep = tags[1] if len(tags) > 1 and tags[1] != '' else '____'
    inner_sep = tags[2] if len(tags) > 2 and tags[2] != '' else '____'   
    for line in sys.stdin:
        if line[-1] == '\n':
            line = line[:-1]
        arr = line.split('\t')
        if len(arr) > idx + 1:
            print line
            continue
        if pair_sep == inner_sep:
            kvs = re.split(pair_sep, arr[idx])
            for i in range(0, len(kvs), 2):
                arr[idx] = '\t'.join([kvs[i], kvs[i + 1]])
                print '\t'.join(arr)
        else:
            pairs = re.split(pair_sep, arr[idx]) 
            for kv in pairs:
                k, v = re.split(inner_sep, kv)
                arr[idx] = '\t'.join([k, v])
                print '\t'.join(arr)


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
        if line[-1] == '\n':
            line = line[:-1]
        line = line.decode('utf8')
        arr = line.split('\t')
        output = [arr[idx] if idx < len(arr) else '' for idx in idxs]
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


def transpose(tags=[]):
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


def html_table(tags=['']):
    ## show cols with table. cmd: html_table, html_table__K, html_table__T, html_table__KT, html_table__KT__2
    """
    第一行表示各列的类型,包括image/url/text,可以只填前n列，后面几列默认为text;
    第二行表示各列的列名,该行为空则不展示;
    剩余行为数据内容;
    参数: TK -> 第一行类型第二行字段名; T -> 第一行类型; K -> 第一行类型名
    """
    tag = tags[0] if len(tags) > 0 else ''
    red_idx = int(tags[1]) if len(tags) > 1 else None
    colors = {
        '1': 'pink',
        '2': '#C0C0C0',
        '3': '#00FFCC',
        '4': '#CCCC66',
        '5': '#9999CC',
    } 
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
            color = colors[arr[red_idx]] if arr[red_idx] in colors else '#ffce9f'
            pre = '<td style="background:%s;word-wrap:break-word; white-space:normal; word-break:break-all;">' % color
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
    ## show cols with cards. cmd:html_card, html_card__KT__5__150__4__1
    """
    第一行表示各列的类型,包括image/url/text,可以只填前n列，后面几列默认为text;
    第二行表示各列的列名,该行为空则不展示;
    剩余行为数据内容;
    参数:
    每一行为一个元素，col_num个元素展现成一行,每个元素展现的宽度为col_width.每个元素的主key是第id_idx个字段, use_boder表示是否展现边框
    """
    tag, col_num, col_width, id_idx, use_boder = '', 5, 150, 1, 0
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


def chart_curve(tags=[]):
    ## show data with curve chart. cmd: chart_curve
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


def chart_pie(tags=[]):
    ## show dta with pies. cmd: chart_pie
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
    ## show data with tree. cmd: chart_tree__json, chart_tree__field
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
            o = {}
            #for i in range(len(obj)):
            #    o['.list_' + unicode(i)] = expand_val(obj[i])
            #obj = o
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



def arg_2_func(string):
    """
    从lprocess_line.py复制过来的函数 
    """
    # 将字符串转换为参数list
    arr = smart_str_list(string, use_end_ch=False)
    arr = [v.decode('utf8', 'ignore') for v in arr]
    
    sym_map = {
        '<TAB>' : '\t',
        '<SPACE>': ' ',
        '<VERT>': '|',
        '<UNDER>': '_',
        '<2UNDER>': '__',
        '<ENTER>': '\n',
    }
    for i in range(len(arr)):
        v = arr[i]
        for sym, val in sym_map.items():
            v = v.replace(sym, val)
        arr[i] = v

    col_funcs = set(dir(lprocess_line) + ['limg'])
    #try:
    if True:
        if arr == []:
            log('notice', 'arg_2_func with no func. {0}'.format(arr))
            return None

        func = arr[0]
        args = arr[1: ]
        if func in col_funcs:
            log('notice', 'arg_2_func with mod arg. {0} {1} {2}'.format('process_lines', func, args))
            output = quick_process_lines(eval(func), arr[1:])
        else:
            log('notice', 'arg_2_func with mod arg. {0} {1}'.format(func, args))
            output = eval(func)(args)

    #except Exception as e: 
    #    print >> sys.stderr, "failed to process.", e, sys._getframe().f_lineno
    #    return None 
    return output



if __name__ == "__main__":
    # 格式为: 分隔符 函数名 分隔符 模式参数 分隔符 参数2 分隔符 参数3 ... 
    # 模式参数: [num]raifp[str])
    func_arg = sys.argv[1]
    func_arg = '__' + func_arg
    arg_2_func(func_arg)
