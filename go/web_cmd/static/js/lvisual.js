
function set_pie_option(data){
    var title = data[0][0]
    var len = data[0].length
    var x =data[0].splice(1, len) // 类别 
    var tag = []
    var ys = []
    for (var i = 1; i < data.length; i++) {
        tag[i - 1] = data[i][0] // 时间轴
        categorys = []
        for (var j = 1; j < data[i].length; j++) {
            categorys[j - 1] = {value: parseFloat(data[i][j]),  name:x[j - 1]}
        }
        ys[i - 1] = {
                name: title, 
                type:'pie',
                center: ['50%', '45%'],
                radius: '50%',
                data:  categorys
        }
    }

    // 指定图表的配置项和数据
    var option = {
        timeline : {
            data : tag, 
        },
        options : [
            {
                title : {
                    text: title,
                    subtext: ''
                },
                tooltip : {
                    trigger: 'item',
                    formatter: "{a} <br/>{b} : {c} ({d}%)"
                },
                legend: {
                    data: x
                },
                toolbox: {
                    show : true,
                    feature : {
                        mark : {show: true},
                        dataView : {show: true, readOnly: false},
                        magicType : {
                            show: true, 
                            type: ['pie', 'funnel'],
                            option: {
                                funnel: {
                                    x: '25%',
                                    width: '50%',
                                    funnelAlign: 'left',
                                    max: 1700
                                }
                            }
                        },
                        restore : {show: true},
                        saveAsImage : {show: true}
                    }
                },
                series : [ ys[0]]
            },
            //{
            //    series : ys.splice(1, ys.length) 
            //}
        ]
    };
    for (var i = 1; i < ys.length; i++) {
        option['options'][i] = {series: [ys[i]]}
    }
    console.log('set pie chart. json:', JSON.stringify(option)) 
    return option; 
}

function set_chart_option(data){
    var title = data[0][0]
    var len = data[0].length
    var x =data[0].splice(1, len)
    var tag = []
    var ys = []
    for (var i = 1; i < data.length; i++) {
        tag[i - 1] = data[i][0]
        ys[i - 1] = {
                name: data[i][0], 
                type:'line',
                smooth:true,
                itemStyle: {normal: {areaStyle: {type: 'default'}}},
                data:data[i].splice(1, len)
            }
    }

    // 指定图表的配置项和数据
    var option = {
        "title" : {
            text: title, 
            //subtext: '纯属虚构'
        },
        tooltip : {
            trigger: 'axis'
        },
        legend: {
            data: tag, 
        },
        toolbox: {
            show : true,
            feature : {
                mark : {show: true},
                dataView : {show: true, readOnly: false},
                magicType : {show: true, type: ['line', 'bar', 'stack', 'tiled']},
                restore : {show: true},
                saveAsImage : {show: true}
            }
        },
        calculable : true,
        xAxis : [
            {
                type : 'category',
                boundaryGap : false,
                data : x,
            }
        ],
        yAxis : [
            {
                type : 'value'
            }
        ],
        series : ys
    };
    return option; 
}

function show_chart(arg, domid){
    // 基于准备好的dom，初始化echarts实例
    var myChart = echarts.init(document.getElementById(domid));

    // 使用刚指定的配置项和数据显示图表。
    console.log(arg)
    myChart.setOption(arg);
}

function str_2_chart(str, tag, domid){
    //var str = "某楼盘销售\t周一\t周二\t周三\n意向\t1320\t1132\t601\n预购\t30\t182\t434\n成交\t10\t12\t21"
    var data = str_2_arr(str)
    var option = ""
    if (tag == "curve") {
        option = set_chart_option(data)
    } else if (tag == "pie") {
        option = set_pie_option(data)
    }
    show_chart(option, domid)
} 


function select_cell(cur, showid){
    console.log("select cell. current:", cur.className);
    if (cur.className.indexOf("selecting_cell") != -1) {
        cur.setAttribute("backup", cur.className)
        cur.className = "green selected_cell";
    } else {
        cur.className = cur.getAttribute("backup"); 
    }
    var no_elems = document.getElementsByClassName("selecting_cell")
    var yes_elems = document.getElementsByClassName("selected_cell")
    var show_elem = document.getElementById(showid);
    arr = []
    for (var i = 0; i < yes_elems.length; i++) {
        arr[i] = yes_elems[i].getAttribute("value");
    }
    hit_num = yes_elems.length
    all_num = document.getElementsByClassName("selecting_cell_table")[0].getAttribute("value")
    str = "";
    str += "hit_num:"  + hit_num + "<br>";
    str += "all_num:" + all_num + "<br>";
    str += "rate:" + hit_num / all_num + "<br>";
    str += "hit:<br>" + arr.join("<br>");

    show_elem.innerHTML = str;
}

function change(selectid, showid){
    var selecting_cell_label = document.getElementById(showid);
    selecting_cell_label.innerHTML = '';
    var c_u_ids = document.getElementsByName(selectid);
    var str = '选中id: ';
    hit = 0
    for (var i = 0; i < c_u_ids.length; i++) {
        if (c_u_ids[i].checked) {
            str += c_u_ids[i].value;
            str += "    ";
            hit += 1;
        }
    }
    p = hit / c_u_ids.length
    p = '' + p
    hit = '' + hit
    n = '' + c_u_ids.length
    str = '总数: ' + n + '<br>选中: ' + hit + '<br>命中率: ' + p + '<br>' + str
    selecting_cell_label.innerHTML = str;
}
