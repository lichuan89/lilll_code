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

:head -6|select_idx____0____1____2____4____5|print_html_table____K
|

图片|名称|标题|链接|价格|评论数|周一销量|周二销量|周三销量|周四销量|周五销量|周六销量|周日销量
http://localhost:8000/static/example/image/mobile.jpg|vivoiQOO|vivo iQOO 水滴全面屏超广角 高通骁龙855|https://item.jd.com/47024922529.html|3298|22000|300|200|100|150|200|100|50
http://localhost:8000/static/example/image/coat.jpg|yaya连衣裙|鸭鸭服饰（yaya）连衣裙小个子穿搭|https://item.jd.com/48109694042.html|168|1000|400|250|300|350|100|100|80
http://localhost:8000/static/example/image/desk.jpg|魅族双肩包|极简都市双肩包背包电脑包15.6英寸|https://item.jd.com/100003019857.html|179|100|500|400|300|200|100|200|400
http://localhost:8000/static/example/image/book.jpg|《失控》|失控——全人类的终命运和结局凯文.凯利|https://item.jd.com/38296718575.html|66|1038|50|200|400|450|300|200|100
http://localhost:8000/static/example/image/fish.jpg|马达加斯加带鱼|马达加斯加去脏带鱼切段 500g/袋 20-25块 |https://item.jd.com/2239275.html|29|1038|400|350|300|200|300|500|200
http://localhost:8000/static/example/image/cat.jpg|短毛猫|美短标斑 短毛猫 可上门 |https://item.jd.com/44164858434.html|2000|1038|400|350|300|200|300|500|200
http://localhost:8000/static/example/image/rice.jpg|米大王6号|崇明岛 2018新大米 真空包装大米 米大王6号 10Kg|https://item.jd.com/4260590.html|108.8|1038|400|350|500|200|300|500|200


:head -6

select_idx____0____1____2____3____4____5|print_html_table____K


select_idx____0____1____2____3____4____5|add_head____image____text____text____url|print_html_field____KT

:head -5|select_idx____1____6____7____8____9____10____11____12|print_curve

:head -5|select_idx____1____6____7____8____9____10____11____12|del_head____1|add_head____sale____1____2____3____4____5____6____7|transpose|print_pie
'''
