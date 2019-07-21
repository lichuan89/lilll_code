#!/bin/bash

param="$1"
cmd=`echo "$param" | awk -F"__" '{print $1}'`
arg1=`echo "$param" | awk -F"__" '{print $2}'`
arg2=`echo "$param" | awk -F"__" '{print $3}'`

args=""
for i in "$@"; do
    if [[ $args == "" ]]; then
        args=$i
    else
        args=$args"__"$i
    fi  
done


# for server
if [[ $cmd == "hello" ]]; then
    ## hello
    ##   echo "Hello World".
    echo "Hello World"
    exit 0
fi

if [[ $cmd == "stop" ]]; then
    ## stop 
    ##   stop server.
    ps aux| grep cmd_server | awk '{print $2;}' | xargs kill -9
    exit 0
fi

if [[ $cmd == "restart" ]]; then
    ## restart
    ##   restart server
    ps aux| grep cmd_server | awk '{print $2;}' | xargs kill -9; nohup go run cmd_server.go  & 
    exit 0
fi

if [[ $cmd == "clear" ]]; then
    ## clear
    ##   clear temp cache 
    rm -r static/temp/*
    exit 0
fi

# for image
if [[ $cmd == "img_head" ]]; then
    ## img_addhead
    ##   add head for col type. cmd: img_addhead
   awk -F"\t" '{
        for(i=1; i<=NF; i++) {
            j = index($i, ".jpg");
            if (j == 0) j = index($i, ".bmp");
            if (j != 0) count[i] += 1;
        }
        if (col < NF) {
            col = NF;
        }
        row += 1;
        lines[row] = $0;
    } END{
        head = "";
        for (i = 1; i <= NF; i++) {
            if (count[i] >= 4 || (row < 10 && count[i] >= 1)) tag = "image"; else tag="text";
            if (head == "") head = tag; else head = head "\t" tag;
        }
        print head;
        for (i = 1; i <= row; i++) print lines[i];
    }'

    exit 0
fi

if [[ $cmd == "img_ls" ]]; then
    ## img_ls 
    ##   list image_fpath in floder. cmd: #img_ls__path1 
    ls static/temp/$arg1/ | awk -F"\t" -v dir="./static/temp/$arg1/" '{print dir $1;}' | grep "$arg1/"
    exit 0
fi

if [[ $cmd == "img_id" ]]; then
    ## img_id
    ## list image floder. cmd: #img_id
    ls -l static/temp/ | awk '{if($1~/d/) { cmd="ls static/temp/"$NF;  s=""; while(cmd | getline v) s=s"\t"v; print $NF"\t"s;}}'
    exit 0
fi
