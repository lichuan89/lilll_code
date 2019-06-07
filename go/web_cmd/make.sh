#!/bin/bash
cp ../../python/common/*py ./
mkdir -p static/temp/
go run cmd_server.go 

echo '''
(1) 格式:
a. 管道式命令
b. 分隔符(为空字符串表示tab)
c. http/ftp文件
d. 数据内容(c和d二选一,为空字符串表示不选)

(2) 例子
:sort -k3,3n|select_idx____0____1____2|base64_field|unbase64_field|:awk -F"\t" '{s+=$3;}END{print "平均年龄:" s/NR;}'
print_html_table____K

|

name|sex|age|height
Tom|male|12|1.5
Lili|femal|15|1.6
Trent|male|11|1.4
cat|animal|1|0.3
