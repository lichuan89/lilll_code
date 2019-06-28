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

