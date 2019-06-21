
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
