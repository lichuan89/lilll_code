#!/bin/bash

param="$1"
cmd=`echo "$param" | awk -F"____" '{print $1}'`
arg1=`echo "$param" | awk -F"____" '{print $2}'`
arg2=`echo "$param" | awk -F"____" '{print $3}'`

if [[ $cmd == "stop" ]]; then
    ps aux| grep cmd_server | awk '{print $2;}' | xargs kill -9; nohup go run cmd_server.go  & 
fi

if [[ $cmd == "restart" ]]; then
    ps aux| grep cmd_server | awk '{print $2;}' | xargs kill -9; nohup go run cmd_server.go  & 
fi

if [[ $cmd == "clear" ]]; then
    rm -r static/temp/* 
fi
