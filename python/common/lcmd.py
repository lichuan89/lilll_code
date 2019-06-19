#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2017/01/01  
    @note   实现python多进程处理输入流的函数
"""
import os 
import sys
import multiprocessing
from lcommon import str_2_file
from lcommon import read_dir
from lcommon import clear_dir 


class Processor(multiprocessing.Process):
    """
    升级进程类, 使用multiprocessing.Manager或临时文件将数据传递给主进程
    """
    def __init__(self, worker, lines, args):
        multiprocessing.Process.__init__(self)
        self.worker = worker
        self.lines = lines
        self.args = args[: -1]
        self.share = args[-1]
    
    def run(self):
        """ 
        run 
        """
        result = self.worker(self.lines, self.args)
        id = os.getpid()
        if type(self.share) == unicode or type(self.share) == type(''):
            str_2_file('\n'.join(result).encode('utf8', 'ignore'), '%s/tmp.%s' % (self.share, id))
        else:
            self.share.setdefault(id, result)


def muti_process(lines, thread_running_num, worker, args, use_share_path=None): 
    """ 
    开启多进程处理数据 
    """
    if use_share_path is None:
        # 使用共享内存
        manager = multiprocessing.Manager()
        contexts = manager.dict()
    else:
        # 先并发输入文件,再统一搜集输出
        clear_dir(use_share_path)
        contexts = use_share_path 
    threadpool = []
    batch_arr = {}

    # 为多进程分配输入数据
    for i in range(len(lines)):
        k = i % thread_running_num 
        batch_arr.setdefault(k, [])
        batch_arr[k].append(lines[i])
    
    for idx in batch_arr:
        th = Processor(worker, batch_arr[idx], args + [contexts])
        threadpool.append(th)

    # 执行多进程
    idx = 0 
    threads = []
    for th in threadpool:
        th.start()
        
    for th in threadpool:
        th.join()

    # 合并结果数据
    lines = []
    if use_share_path is not None:
        lines = read_dir(use_share_path)
        clear_dir(use_share_path)
    else: 
        for k, v in contexts.items():
            if v is None:
                continue
            for line in v:
                lines.append(line)
    return lines


def muti_process_stdin(worker, args, batch_line_num, thread_running_num, use_share_path=None):
    """
    多进程处理标准输入流, 批处理，并输出到标准输出流
    每一批处理batch_line_num行数据，每一批开启thread_running_num个线程 
    worker格式如:
    def worker(lines, args): return ['%s:%s' % (args[0], line) for line in lines]
    """
    idx = 0 
    batch = []
    for line in sys.stdin:
        line = line[: -1].decode('utf8', 'ignore')
        batch.append(line)
        if len(batch) >= batch_line_num:
            output_lines = muti_process(batch, thread_running_num, worker, args, use_share_path)
            if output_lines != []:
                print '\n'.join(output_lines).encode('utf8', 'ignore')
                sys.stdout.flush()
            batch = []
    if batch != []:
        output_lines = muti_process(batch, thread_running_num, worker, args, use_share_path)
        if output_lines != []:
            print '\n'.join(output_lines).encode('utf8', 'ignore')

def test():
    def worker(lines, args): return ['%s:%d:%s' % (args[0], os.getpid(), line) for line in lines]
    #muti_process_stdin(worker, ['prefix'], batch_line_num=30, thread_running_num=7, use_share_path='tmp.muti_process/')
    muti_process_stdin(worker, ['prefix'], batch_line_num=30, thread_running_num=7, use_share_path=None)


if __name__ == "__main__":
    test()
