#!/bin/bash

param="$1"
cmd=`echo "$param" | awk -F"____" '{print $1}'`
arg1=`echo "$param" | awk -F"____" '{print $2}'`
arg2=`echo "$param" | awk -F"____" '{print $3}'`

args=""
for i in "$@"; do
    if [[ $args == "" ]]; then
        args=$i
    else
        args=$args"____"$i
    fi  
done




if [[ $cmd == "stop" ]]; then
    ps aux| grep cmd_server | awk '{print $2;}' | xargs kill -9; nohup go run cmd_server.go  & 
    exit 0
fi

if [[ $cmd == "restart" ]]; then
    ps aux| grep cmd_server | awk '{print $2;}' | xargs kill -9; nohup go run cmd_server.go  & 
    exit 0
fi

if [[ $cmd == "clear" ]]; then
    rm -r static/temp/* 
    exit 0
fi

# for image

if [[ $cmd == "image_save_show" ]]; then
    mkdir static/data/cmd/ || echo "" > /dev/null
    awk '{
        gsub("static/temp/", "../../static/temp/");
        print $0;
    }' > static/data/cmd/$arg1
    cat static/data/cmd/$arg1 
fi

if [[ $cmd == "html_image_show" ]]; then
    # #html_image_show____c____r____l
    cnt=`echo "$args" | awk -F"____" '{print NF - 1;}'`
    ls static/temp/$arg1/ | awk -F"\t" -v fs="$args" '{
            l=split(fs, arr, "____");
            s=arr[2]"/"$1;
            for (i=3;i<=l;i++){
                s=s"\t"arr[i]"/"$1;
            } 
            print s;
    }' | sh cmd.sh image_fmt____$cnt | python process_line.py html_table____T
    exit 0
fi

if [[ $cmd == "image_ls" ]]; then
    # 枚举指定文件件下文件
    ls static/temp/$arg1/ | awk -F"\t" -v dir="$arg1/" '{print dir $1;}' | grep "$arg1/"
    exit 0
fi

if [[ $cmd == "image_fmt" ]]; then
    # 为前几个字段补充路径
    # 前$arg1个字段是图片
    awk -F"\t" -v idx=$arg1 'BEGIN{
            s="text"; 
            for(i=1;i<=idx;i++)s=s"\timage";
            print s;
        }{
            pre="../../static/temp/"; s=NR; 
            for(i=1;i<=idx;i++) {
                if ($i~/^http/ || !($i~/\//)) s = s "\t"$i;
                else s=s"\t"pre $i;
            } 
            for(i=idx+1;i<=NF;i++)s=s"\t"$i; print s;
        }'
    exit 0
fi

if [[ $cmd == "image_crawl" ]]; then
    #echo -e "http://ms.bdimg.com/dsp-image/1756536684.jpg\ta\nhttp://ms.bdimg.com/dsp-image/571671431.jpg\ttest" | sh cmd.sh crawl____jd
    # 输入: 图片url, ...
    # 输出: 图片url, 本地图片url, ...
    cd ../../python/image/
    awk -F"\t" -v dir="$arg1" '{
            l = split($1, arr, "/");
            f0 = dir "/" NR"__"arr[l];
            f1 = "static/temp/"f0;
            f2 = "../../go/web_cmd/" f1

            $1 = $1"\t"f2"\t"f0;
            OFS = "\t";
            print $0; 
    }' | python limg_process_line.py crawl | awk -F"\t" '{s=$3"\t"$1; for(i=4;i<=NF;i++) s=s"\t"$i; print s;}' 
    cd - > /dev/null
    exit 0
fi

    #sh cmd.sh simple_extract_skin____path5
    # 输入: 本地图片url, ...
    # 输出: 本地产出图片url, 本地图片url, ... 
    cd ../../python/image/
        python limg_process_line.py $args
        exit 0
    exit 0
    cd - > /dev/null
