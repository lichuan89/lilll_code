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
print_html_attr____key____image____text____text____url
|

1|http://img10.360buyimg.com/n7/jfs/t1/7015/30/2122/109896/5bd1598dE5bbab285/9b2b9b89d823218e.jpg|海信（Hisense）HZ55E5A 55英寸 超高清|3368|https://item.jd.com/100000384561.html
2|http://img11.360buyimg.com/n7/jfs/t1/8073/22/3604/363230/5bd75206E77c5e1ff/3e98ee776e76f4e8.jpg|创维（SKYWORTH）65H5 65英寸4K超高|4001|https://item.jd.com/100000384561.html

e. 例子5
print_html_image____3____100
|

https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1558681936380&di=65e0883b2bed6e52daa037ae4969e5c6&imgtype=0&src=http%3A%2F%2Fc.hiphotos.baidu.com%2Fzhidao%2Fpic%2Fitem%2Ffaf2b2119313b07e20a919850cd7912397dd8c23.jpg|图1|id1
https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=1859084109,1308425889&fm=26&gp=0.jpg|图2|id2
https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=3465693520,216264677&fm=26&gp=0.jpg|图3|id3
https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=2472113467,807541940&fm=26&gp=0.jpg|图4|id4
https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=685739431,3691001328&fm=26&gp=0.jpg|图5|id5
https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=1439219251,3571615474&fm=26&gp=0.jpg|图6|id6
'''
