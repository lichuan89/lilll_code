#!/bin/bash
# note: 微服务，接收请求的命令和参数，返回结果
# author: lichuan89@126.com
# date:   2019/01/01


cmd="$1"
output_fpath="$2"

cmd=$( echo "$cmd" | awk -F"|" '{for(i=1;i<=NF;i++) s = s "python process_line.py "$i" | ";}END{s = s" cat";  print s}')

echo "$cmd > $output_fpath" > $output_fpath.sh
/bin/bash $output_fpath.sh
#python process_line.py "$cmd" > $output_fpath
