#!/bin/bash
cp ../../python/common/*py ./
go run cmd_server.go 

'''
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
'''
