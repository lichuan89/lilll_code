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

function str_2_chart(str, domid){
    //var str = "某楼盘销售\t周一\t周二\t周三\n意向\t1320\t1132\t601\n预购\t30\t182\t434\n成交\t10\t12\t21"
    var data = str_2_arr(str)
    var option = set_chart_option(data)
    show_chart(option, domid)
} 


function change(){
    var show_analysis_data_label = document.getElementById("show_analysis_data_label");
    show_analysis_data_label.innerHTML = '';
    var c_u_ids = document.getElementsByName("show_analysis_data_id");
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
    show_analysis_data_label.innerHTML = str;
}
