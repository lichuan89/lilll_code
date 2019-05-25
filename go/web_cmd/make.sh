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
<1> 表格信息
print_html_attr____image____text____url____text____text
|

图片|名称|链接|价格|评论数|销量
http://localhost:8000/static/example/image/mobile.jpg|vivo iQOO 水滴全面屏超广角 高通骁龙855|https://item.jd.com/47024922529.html|3298|22000|3000
http://localhost:8000/static/example/image/coat.jpg|鸭鸭服饰（yaya）连衣裙小个子穿搭|https://item.jd.com/48109694042.html|168|1000|1446
http://localhost:8000/static/example/image/desk.jpg|极简都市双肩包背包电脑包15.6英寸|https://item.jd.com/100003019857.html|179|100|306
http://localhost:8000/static/example/image/book.jpg|失控——全人类的终命运和结局凯文.凯利|https://item.jd.com/38296718575.html|66|1038|2391
http://localhost:8000/static/example/image/fish.jpg|马达加斯加去脏带鱼切段 500g/袋 20-25块 |https://item.jd.com/2239275.html|29|1038|2391
http://localhost:8000/static/example/image/cat.jpg|美短标斑 短毛猫 可上门 |https://item.jd.com/44164858434.html|2000|1038|2391
http://localhost:8000/static/example/image/rice.jpg|崇明岛 2018新大米 真空包装大米 米大王6号 10Kg|https://item.jd.com/4260590.html|108.8|1038|2391

其他命令:
rsearch____^http____all|print_html_attr____image____text____url____text____text
select_fields____0____1____2|rrsearch____图片.*名称|print_html_image____4____200
select_fields____1____3____4____5|print_html_table


<2> 趋势信息
cat|print_html_chart
|

广告费|周一|周二|周三|周四|周五|周六|周日
移动|300|200|100|150|200|100|50
电脑|400|250|300|350|100|100|80

'''
