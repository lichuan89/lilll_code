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
a. 例子1
cat|print_html_chart
|

广告费|周一|周二|周三|周四|周五|周六|周日
移动|300|200|100|150|200|100|50
电脑|400|250|300|350|100|100|80



b. 例子2
select_fields____0____1____3____5____7|match_str____广告费____移动|print_html_table
|

广告费|周一|周二|周三|周四|周五|周六|周日
移动|300|200|100|150|200|100|50
电脑|400|250|300|350|100|100|80


c. 例子3
select_fields____0____1____3____5____7|match_str____广告费____移动|print_html_table
|
http://localhost:8000//static/temp/select_fields____0____1____3____5____7__print_html_table_20190515191223_20109600.input.txt
这一行无效


d. 例子4
print_html_image____key____image____text____text____url
|

1|http://img10.360buyimg.com/n7/jfs/t1/7015/30/2122/109896/5bd1598dE5bbab285/9b2b9b89d823218e.jpg|海信（Hisense）HZ55E5A 55英寸 超高清|3368|https://item.jd.com/100000384561.html
2|http://img11.360buyimg.com/n7/jfs/t1/8073/22/3604/363230/5bd75206E77c5e1ff/3e98ee776e76f4e8.jpg|创维（SKYWORTH）65H5 65英寸4K超高|4001|https://item.jd.com/100000384561.html

'''
