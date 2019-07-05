
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

function show_output(url, req_data, res_data) {
    var oTxt = document.getElementById("output_area");
    oTxt.value = res_data
 
    var oDiv = document.getElementById("output_html");
    var arr = res_data.split("\n")
    var input_url = arr[0]
    var output_url = arr[1]
    var log_url = arr[2]
    var context = arr.slice(3).join("\n")

    if (context.indexOf("#!lcmd") == 0) {
        var context = arr.slice(4).join("\n")
        document.getElementById("input_area").value = context;
        OnButton();
    }

    var html = '<a href="' + input_url + '" target="_blank">输入链接</a>'
    html += ' --> '
    html += '<a href="' + output_url + '" target="_blank">输出链接</a> | '
    html += '<a href="' + log_url + '" target="_blank">日志链接</a><br>'
    var cmd = req_data.split("\n")[0]
    console.log('print chart. ', cmd, context)
    if (cmd.indexOf("chart_") != -1) {
        context = context.split("\n")[0];
        var option = JSON.parse(context);
        var domid = "output_html";
        var dom = document.getElementById(domid);
        dom.innerHTML = "";
        var file_dom = document.createElement("div");
        file_dom.innerHTML = html;
        dom.appendChild(file_dom);

        var chart_dom =document.createElement("div");
        chart_dom.setAttribute('style', 'width: 800px;height:400px;');
        dom.appendChild(chart_dom);

        var myChart = echarts.init(chart_dom);
        console.log("it will show chart:", option);
        myChart.setOption(option);
    } else {
        if (cmd.indexOf("html_") == -1) { 
            //context = context.replace(/\n/g, "<br>")
            context = "<pre>" + context + "</pre>" 
        }
        html += context
        oDiv.innerHTML= html;
    } 
}
