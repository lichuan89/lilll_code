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
import copy 
import hashlib
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
        #uni = uni.replace('\\\\"', '\\"')
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

def json_2_fmt_str(obj, indent=4):
    """  
    json to str
    """
    try:    
        return json.dumps(obj, sort_keys=True, indent=indent, ensure_ascii=False)
    except Exception as e:
        return ''

def file_2_str(fpath, decode=None):
    """
    从文件读取字符串
    """
    with open(fpath, 'rb') as f:
        context = f.read()
        if decode is not None:
            context = context.decode(decode, 'ignore')
        return context

def file_2_dict(fpath, sep='\t', decode=None):
    """
    从文件读取字符串
    """
    h = {}
    with open(fpath, 'rb') as f:
        for line in f.readlines():
            if line[-1] == '\n':
                line = line[: -1]
            if decode is not None:
                line = line.decode(decode, 'ignore')
            arr = line.split(sep)
            if arr == []:
                continue
            elif len(arr) < 2:
                arr.append('')
            h[arr[0]] = sep.join(arr[1: ])
    return h

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
        url = url.split(':')[0]
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
        return [url]


def expand_json(obj, sep='----', is_simple_key=True, is_merge_basic_list=False, path=[]):
    """
    将json展开成一个扁平的dict, is_simple_key=False, 则使用全路径;
    is_simple_key=True, 则将冲突的val合并.
    """
    def merge_dict(o1, o2):
        o = {}
        for k, v in (o1.items() + o2.items()):
            o.setdefault(k, [])
            o[k].append(v)
        for k, v in o.items():
            o[k] = sep.join([unicode(i) for i in v])
        return o 

    kvs = {}
    kns = {}
    path_key = sep.join(path)
    if is_simple_key:
        path_key = path[-1] if path != [] else ''
    if type(obj) == dict:
        kns[path_key] = json_2_str(obj)
        for k, v in obj.items():
            o1, o2 = expand_json(v, sep, is_simple_key, is_merge_basic_list, path + [k])
            kvs = merge_dict(kvs, o1)
            kns = merge_dict(kns, o2)
    elif type(obj) == list and is_merge_basic_list and True not in [(type(i) == list or type(i) == dict) for i in obj]:
        kvs[path_key] = sep.join([unicode(v) for v in obj]) 
    elif type(obj) == list:
        kns[path_key] = json_2_str(obj)
        for i in range(len(obj)):
            o1, o2 = expand_json(obj[i], sep, is_simple_key, is_merge_basic_list, path + [unicode(i)])
            kvs = merge_dict(kvs, o1)
            kns = merge_dict(kns, o2)
    else:
        kvs[path_key] = obj

    return kvs, kns 


def unexpand_pair(obj, pair_key, pair_val, children_val=None, unit_dict={}):

    if type(obj) == list:
        items = [(str(i), obj[i]) for i in range(len(obj))]
    elif type(obj) == dict:
        items = obj.items()
    else:
        return obj

    if children_val is None:
        o = {}
        for k, v in items:
            o[k] = unexpand_pair(v, pair_key, pair_val, children_val, unit_dict)
    else:
        o = []
        for k, v in items:
            val = copy.deepcopy(unit_dict)
            val[pair_key] = k
            if type(v) == dict or type(v) == list:
                val[children_val] = unexpand_pair(v, pair_key, pair_val, children_val, unit_dict)
            else:
                val[pair_val] = v
            o.append(val)
    return o  


def expand_pair(obj, pair_keys, pair_vals):
    """
    将json内格式为{pair_key:v1, pair_val: v2}的结构写成{v1:v2}
    """ 
    if type(pair_keys) != list:
        pair_keys = [pair_keys]
    if type(pair_vals) != list:
        pair_vals = [pair_vals]
    if type(obj) == dict:
        pair_key = None
        for k in pair_keys:
            if k in obj:
                pair_key = k
        pair_val = None
        for k in pair_vals:
            if k in obj:
                pair_val = k
        if pair_key is not None and pair_val is not None:
            obj[obj[pair_key]] = expand_pair(obj[pair_val], pair_keys, pair_vals)
            del obj[pair_key]
            del obj[pair_val] 
        else:
            for k, v in obj.items():
                obj[k] = expand_pair(v, pair_keys, pair_vals)
    elif type(obj) == list:
        count = 0 
        for v in obj: 
            pair_key = [name for name in pair_keys if type(v) == dict and name in v]
            pair_val = [name for name in pair_vals if type(v) == dict and name in v]
            if pair_key != [] and pair_val != []:
                count +=1
                
        for i in range(len(obj)):
            obj[i] = expand_pair(obj[i], pair_keys, pair_vals)

        if count == len(obj):
            o = {}
            for item in obj:
                for k, v in item.items():
                    o[k] = v
            obj = o
    return obj 


def numdict_2_list(obj):
    if type(obj) == dict:
        cnt = 0
        while cnt in obj or str(cnt) in obj:
            cnt += 1
        if len(obj) == cnt:
            arr = []
            for i in range(cnt):
                arr.append(obj[i] if i in obj else obj[str(i)])
            obj = [numdict_2_list(v) for v in arr]
        else:
            for k, v in obj.items():
                obj[k] = numdict_2_list(v) 
    elif type(obj) == list:
        obj = [numdict_2_list(v) for v in obj]
    return obj


def list_2_numdict(obj):
    if type(obj) == dict:
        for k, v in obj.items():
            obj[k] = list_2_numdict(v)
    elif type(obj) == list:
        obj = dict([(str(i), obj[i]) for i in range(len(obj))]) 
    return obj

def unexpand_json(obj, sep='----'):
    o = {}
    for lk, v in obj.items():
        ks = lk.split(sep)
        m = o
        for i in range(len(ks)):
            if i != len(ks) - 1:
                m.setdefault(ks[i], {})
                m = m[ks[i]]
            else:
                m[ks[i]] = v
    o = numdict_2_list(o)
    return o
        
def find_map(obj, reg):
    output = {}
    for k in obj:
        if re.search(reg, k) is not None:
            output[k] = obj[k]
    return output

def crawl(url, decode=None, post_data=None, header_map={}, is_encode_post_data=True):
    """
    抓取网页, 如果post_data为{}则用get方法抓取网页
    """ 
    if post_data is not None:    
        if is_encode_post_data:
            post_data = urllib.urlencode(post_data)
        else:
            post_data = json.dumps(post_data)
    try:
        headers = {
                'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1;' + \
                ' en-US; rv: 1.9.1.6) Gecko/20091201 Firefox/3.5.6', # 针对服务器限制
                'Referer': 'http://douban.com' # 针对防盗链限制
                }
        for key in header_map:
            headers[key] = header_map[key]
        if post_data is None:
            req = urllib2.Request(url, headers=headers)
        else:
            req = urllib2.Request(url, data = post_data, headers=headers)
        res = urllib2.urlopen(req, timeout = 10)
        code = res.getcode()
        if code < 200 or code >= 300:
            return None
        t = res.headers.dict['content-type']


        context = res.read()

        m = re.search('charset=(.*?)(?:;|$)', t)
        if decode is not None:
            if m is not None:
                decode = m.groups()[0]
            context = context.decode(decode, 'ignore')
        return context
    except Exception as e:
        if isinstance(e, urllib2.HTTPError):
            log('http error: {0}'.format(e.code))
        elif isinstance(e, urllib2.URLError): # and isinstance(e.reason, socket.timeout):
            log('url error: socket timeout {0}'.format(e.__str__()))
        else:
            log('misc error: ' + e.__str__())
        return None


def md5(input):
    """
    md5
    """
    md5sign = hashlib.md5()
    md5sign.update(input)
    str = md5sign.digest()
    data = struct.unpack("IIII", str)
    md5value = data[0] << 96 | data[1] << 64 | data[2] << 32 | data[3] 
    return md5value

def find_begin_reduplication(string):
    if string == '':
        return ''

    word = string[0]
    for i in range(1, len(string)):
        if string[i] == string[0]:
            word = string[: i + 1]
        else:
            break
    return word


def smart_str_list(string, use_end_ch=False):
    """
    格式为 sep1 
                sep2 v1
                sep2 v2
                spe2 v3 
                sep2 ... 
            sep1
    """
    if use_end_ch:
        end_ch = find_begin_reduplication(string)
        string = string[len(end_ch):]
        end = string.find(end_ch)
        string = string[: end]
    if string == '':
        return []
    sep = find_begin_reduplication(string)
    arr = string.split(sep)
    return arr[1: ]

def smart_str_dict(string, use_end_ch=False, use_order=False):
    """
    格式为 sep1
                sep2
                    k1 sep3 v11 spe3 v12
                sep2
                sep2
                    k2 sep3 v21
                sep2
            sep1
                
    """
    if use_end_ch:
        end_ch = find_begin_reduplication(string)
        string = string[len(end_ch):]
        end = string.find(end_ch)
        string = string[: end]
    if string == '':
        return {}
    sep_ch = find_begin_reduplication(string)
    end = string.find(sep_ch)
    string = string[len(sep_ch): ]
    if string == '':
        return {}
    inner_sep_ch = find_begin_reduplication(string)
    end = string.find(inner_sep_ch)
    string = string[len(inner_sep_ch): ]
    arr = string.split(sep_ch)
    kvs = []
    for item in arr:
        kv = item.split(inner_sep_ch)
        k = kv[0]
        v = kv[1: ]
        if len(v) <= 1:
            v = ''.join(v)
        kvs.append([k, v])
    if use_order:
        return kvs
    else:
        return dict(kvs)


def json_2_kvs(obj, keys):
    output = {}
    if type(obj) == dict:
        for k, v in obj.items():
            if k in keys:
               output[k] = v if type(v) is not list and type(v) is not dict else json_2_str(v)
            else:
                o = json_2_kvs(v, keys)
                for i in o:
                    output[i] = o[i]
    elif type(obj) == list:
        for v in obj:
            o = json_2_kvs(v, keys) 
            for i in o:
                output[i] = o[i]
    return output

 
def test_val():
    print find_begin_reduplication("##12iii#####") == "##" 
    print find_begin_reduplication("a##12iii#####") == "a" 
    print smart_str_list("##|a|b|c##", True) == ['a', 'b', 'c']
    print smart_str_list("#||a||b||c#", True) == ['a', 'b', 'c']
    print smart_str_list("||a||b||c", False) == ['a', 'b', 'c']
    print smart_str_dict("##;:a:b;c:d:e;f##", use_end_ch=True, use_order=False) == {'a': 'b', 'c': ['d', 'e'], 'f': ''}
    print smart_str_dict("##;:a:b;c:d:e;f##", use_end_ch=True, use_order=True) == [['a', 'b'], ['c', ['d', 'e']], ['f', '']]
    print json_2_kvs({"a":["b", "c"], "d": 123}, ["a"]) == {'a': '["b", "c"]'}
    print json_2_kvs({"a":["b", "c"], "d": 123}, ["a", "d"]) == {'a': '["b", "c"]', 'd': 123} 
    print file_2_dict('lcommon.py', sep=' ')

def test(opt):
    test_val()
    return
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
            {'k2': ['v2'], 'k3': 'v3', 'k4': [5, 6]},
            {'k2': ['v10', {}]} 
        ]
        print 'ori:', obj
        print 'expand_json 1:', expand_json(obj, '____')
        print 'expand_json 2:', expand_json(obj, '____', False)
        print 'expand_json 3:', expand_json(obj, '____', True, True)
        o = expand_json(obj, '____', False, False)
        print 'ori:', o[0]
        print 'unexpand_json:', unexpand_json(o[0], '____')
        obj ={ 
            'k1': {'key': 1, 'val': 2},
            'k2': [{'key': 11, 'val': 22}, {'key': 10, 'val': 20}],
        }
        print 'ori: ', obj
        print 'expand_pair: ', expand_pair(obj, 'key', 'val')
    
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
        print "ori:", data 
        obj = expand_pair(data, ["name"], ["value", "children"])
        print "expand_pair:", obj
        obj = unexpand_pair(obj, pair_key='name', pair_val='value', children_val='children', unit_dict={'collapsed': 0})
        print "unexpand_pair:", obj

    if opt == 'numdict_2_list' or opt == 'all':   
        obj = {'0': '00', '1': '11', '2': [{4: 1}, {0:5}]}
        print obj 
        print numdict_2_list(obj)

    if opt == 'crawl' or opt == 'all':
        ret = crawl('http://fxhh.jd.com/detail.html?id=173238717')
        print ret


if __name__ == "__main__":
    opt = "all"
    test(opt)
