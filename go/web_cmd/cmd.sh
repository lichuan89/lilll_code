#!/bin/bash
# note: 微服务，接收请求的命令和参数，返回结果
# author: lichuan89@126.com
# date:   2019/01/01


cmd="$1"
param="$2"

if [[ $cmd == "cat" ]]; then
    echo "[$param]"
fi

