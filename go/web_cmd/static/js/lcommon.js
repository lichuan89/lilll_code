// String函数
String.prototype.trim=function(){
　　    return this.replace(/(^\s*)|(\s*$)/g, "");
　　 }

String.prototype['format'] = function () {
    const e = arguments;
    return !!this && this.replace(/\{(\d+)\}/g, function (t, r) {
        return e[r] ? e[r] : t;
    });
}

function str_2_arr(str) {
    var arr = []
    var lines = str.split("\n")
    for(var i = 0; i < lines.length; i++) {
        if (lines[i] == "") {
            continue
        }
        var cols = lines[i].split("\t")
        arr[i] = cols 
    }
    return arr
}


// ajax: 无需重新加载页面, 异步请求服务器(HTTP)获取数据来, 并局部更新网页。
// 呈现的效果是局部刷新，不影响用户操作。
function ajax_request(url, req_data, callback) {
    console.log("ajax request. url:%s, req:%s", url, req_data)
    // 局部请求http
    var xhr = new XMLHttpRequest();
    xhr.open("post", url, true);
    xhr.setRequestHeader("Content-Type","application/json");
    xhr.send(req_data)
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4) { // 读取完成
            if (xhr.status == 200) {
                console.log("ajax response. url:%s, res:%s, res:%s", url, req_data, xhr.responseText)
                callback(url, req_data, xhr.responseText)
            }
        }
    }
}

function show_output(url, req_data, res_data) {
    var oTxt = document.getElementById("output_area");
    oTxt.value = res_data
 
    var oDiv = document.getElementById("output_html");
    var arr = res_data.split("\n")
    var input_url = arr[0]
    var output_url = arr[1]
    var context = arr.slice(2).join("\n")
    var html = '<a href="' + input_url + '" target="_blank">输入链接</a>'
    html += ' --> '
    html += '<a href="' + output_url + '" target="_blank">输出链接</a><br>'
    var cmd = req_data.split("\n")[0]
    console.log('print chart. type:', cmd)
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
    } else if (cmd.indexOf("print_curve") != -1) {
        str_2_chart(context, "curve", "output_html")
    } else if (cmd.indexOf("print_pie") != -1) {
        str_2_chart(context, "pie", "output_html")
    } else if (cmd.indexOf("print_scatter") != -1) {
        str_2_chart(context, "scatter", "output_html")
    } else {
        if (cmd.indexOf("print_html_") == -1 && cmd.indexOf("html_") == 0) { 
            context = context.replace(/\n/g, "<br>")
        }
        html += context
        oDiv.innerHTML= html;
    } 
}
