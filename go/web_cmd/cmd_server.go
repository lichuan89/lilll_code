// note: 微服务，接收请求的命令和参数，返回结果 
// author: lichuan89@126.com
// date:   2019/01/01

package main

import (
    "regexp"
    "encoding/base64"
    "bytes"
    "errors"
    "fmt"
    "encoding/json"
    "html/template"
    "os/exec"
    "io"
    "io/ioutil"
    "net/http"
    "net"
    "runtime"
    "strings"
    "bufio"
    "time"
    "os"
    "log"
    "strconv"
)

const (
    IP string = "127.0.0.1"
    HTTP_PORT  string = "8000"
)

func u2s(from string) string{
    sUnicodev := strings.Split(from, "\\u")
    var context string
    for _, v := range sUnicodev {
         if len(v) < 1 { 
             continue
         }
         temp, err := strconv.ParseInt(v, 16, 32)
         if err != nil {
             panic(err)
         }
         context += fmt.Sprintf("%c", temp)
     }
     return context
}

func GetIP() string {
    addrs, err := net.InterfaceAddrs()
    if err != nil{
        return ""
    }   
    for _, value := range addrs{
        if ipnet, ok := value.(*net.IPNet); ok && !ipnet.IP.IsLoopback(){
            if ipnet.IP.To4() != nil{
                return ipnet.IP.String()
            }   
        }   
    }   
    return ""
}

func GetNowTime() string {
    // 返回当前时刻，格式为[年 月 日 时 分 秒]，如 20190515070850
    nowTime := time.Now()
    t := nowTime.String()
    timeStr := t[:19]
    timeStr = strings.Replace(timeStr, "-", "", -1) 
    timeStr = strings.Replace(timeStr, ":", "", -1) 
    timeStr = strings.Replace(timeStr, " ", "", -1) 
    return timeStr
}

func GetGID() uint64 {
       b := make([]byte, 64)
       b = b[:runtime.Stack(b, false)]
       b = bytes.TrimPrefix(b, []byte("goroutine "))
       b = b[:bytes.IndexByte(b, ' ')]
       n, _ := strconv.ParseUint(string(b), 10, 64)
       return n
}

func Str_2_file(content string, fileName string, is_append bool) {
    // https://blog.csdn.net/qq_34021712/article/details/86433918
    mod := os.O_CREATE|os.O_WRONLY|os.O_TRUNC
    if is_append {
        mod = os.O_CREATE|os.O_APPEND|os.O_RDWR
    }
    //f, err := os.Create(fileName)
    f, err := os.OpenFile(fileName, mod, 0660)
    if err != nil {
        fmt.Println(err)
        return
    }
    l, err := f.WriteString(content)
    if err != nil {
        fmt.Println(err)
        f.Close()
        return
    }
    fmt.Println(l, "bytes written successfully")
    err = f.Close()
    if err != nil {
        fmt.Println(err)
        return
    }
}
func Byte_2_file(content []byte, fileName string) {
    // https://blog.csdn.net/qq_34021712/article/details/86433918
    f, err := os.Create(fileName)
    if err != nil {
        fmt.Println(err)
        return
    }
    //l, err := f.WriteString(content)
    l, err := f.Write([]byte(content))
    if err != nil {
        fmt.Println(err)
        f.Close()
        return
    }
    fmt.Println(l, "bytes written successfully")
    err = f.Close()
    if err != nil {
        fmt.Println(err)
        return
    }
}


func File_2_str(fileName string, row_num int) string {
    //创建文件,给与创、写权限，421:读写操
    filePointer, err := os.OpenFile(fileName, os.O_RDONLY|os.O_CREATE, 0751)
    if err != nil {
        log.Println("文件创建失败！错误：", err)
        return ""
    }
    //创建成功挂起关闭文件流,在函数结束前执行
    defer filePointer.Close()
 
    reader := bufio.NewReader(filePointer)
    lines := []string {}
    row := 0
    var readString string
    for {
        readString, err = reader.ReadString('\n')
        if err == nil {
            //fmt.Print(readString)
            lines = append(lines, readString)
        } else if err == io.EOF {
            //fmt.Print(readString)
            lines = append(lines, readString)
            break
        } else {
            log.Println("读取失败，错误：", err)
            break
        }
        row += 1
        if row_num != -1 && row >= row_num {
            break
        }
    }
    return strings.Join(lines, "")
}

func File_2_dict(fileName string) map[string] string {
    str := File_2_str(fileName, -1)
    lines := strings.Split(str, "\n")
    hash := map[string] string {}
    for _, line := range lines{
        fields := strings.Split(line, "\t")
        if len(fields) < 2 {
            continue
        }
        hash[fields[0]] = fields[1]
    }
    return hash
}

func json_2_str(obj interface{}) string {
    json_byte, err := json.MarshalIndent(obj, "", " ")
    if err != nil{
        return ""
    }   
    return string(json_byte)
}

func str_2_json(str string) interface{} {
    if str[: 1] == "[" {
        obj := []interface{}{}
        json.Unmarshal([]byte(str), &obj)
        return obj 
    
    }
    obj := map[string]interface{}{}
    json.Unmarshal([]byte(str), &obj)
    return obj 
}


func init() {
    runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
    fs := http.FileServer(http.Dir("static/"))
    http.Handle("/static/", http.StripPrefix("/static/", fs))

    http.HandleFunc("/markdown", MarkDown)
    http.HandleFunc("/cmd", Cmd)
    http.HandleFunc("/card", Card)
    http.HandleFunc("/cmd_ajax", CmdAjax)
    http.HandleFunc("/echo_ajax", EchoAjax)
    err := http.ListenAndServe(IP + ":" + HTTP_PORT, nil)
    if err != nil {
        fmt.Println("服务失败 /// ", err)
    }
}

func WriteTemplateToHttpResponse(res http.ResponseWriter, t *template.Template) error {
    if t == nil || res == nil {
        return errors.New("WriteTemplateToHttpResponse: t must not be nil.")
    }
    var buf bytes.Buffer
    err := t.Execute(&buf, nil)
    if err != nil {
        return err
    }
    res.Header().Set("Content-Type", "text/html; charset=utf-8")
    _, err = res.Write(buf.Bytes())
    return err
}

func EchoHtml(res *http.ResponseWriter, req *http.Request, fpath string) {
    t, err := template.ParseFiles(fpath)
    if err != nil {
        fmt.Println(err)
        return
    }
    err = WriteTemplateToHttpResponse(*res, t)
    if err != nil {
        fmt.Println(err)
        return
    }
}

func Cmd(res http.ResponseWriter, req *http.Request) {
    EchoHtml(&res, req, "static/html/cmd.html")
}

func Card(res http.ResponseWriter, req *http.Request) {
    EchoHtml(&res, req, "static/html/card.html")
}


func MarkDown(res http.ResponseWriter, req *http.Request) {
    t, err := template.ParseFiles("static/html/markdown.html")
    if err != nil {
        fmt.Println(err)
        return
    }
    err = WriteTemplateToHttpResponse(res, t)
    if err != nil {
        fmt.Println(err)
        return
    }
}


func exec_shell(s string) (string, error){ 
    cmd := exec.Command("/bin/bash", "-c", s)
    var out bytes.Buffer
    cmd.Stdout = &out
    err := cmd.Run()
    return out.String(), err 
}

func parse_cmd(cmd string, log_fpath string) string {
    arr := strings.Split(cmd, "|")
    cmds := [] string{}

    // 使用命令替换
    tmp_arr := []string {}
    cmd_map := File_2_dict("cmd_map.list")
    for _, cmd := range(arr){
        split_cmd := strings.Split(cmd, "__")
        split_cmd_key := split_cmd[0]
        v, ok := cmd_map[split_cmd_key]
        if ok {
            subs := strings.Split(v, "|")
            if len(subs) > 0{
                split_cmd[0] = subs[len(subs) - 1]
                subs[len(subs) - 1] = strings.Join(split_cmd, "__") 
            }
            for _, sub := range(subs) {
                tmp_arr = append(tmp_arr, sub)
            }
        } else {
            tmp_arr = append(tmp_arr, cmd)
        }
    }
    fmt.Printf("replace cmd. form:[%s], to:[%s]\n", arr, tmp_arr)
    arr = tmp_arr

    // 使用输出数据栈中的数据作为这一次输入
    idx := -1
    for _, ch := range(cmd) {
        if ch == '|' {
            idx += 1
        } else {
            break 
        }
    }
    if cmd == "" || len(cmd) == idx + 1{
        idx += 1
    }
    if idx != -1 {
        s := fmt.Sprintf("cat static/temp/pipeline.%d.txt", idx) // 默认最近第idx次输出, 从0开始
        cmds = append(cmds, s)
    }
    for _, v := range arr{
        if v == "" {
            continue
        } else if v[: 1] == ":" {
            s := fmt.Sprintf("%s 2> %s", v[1: ], log_fpath)
            cmds = append(cmds, s)
        } else if v[: 1] == "#" {
            s := fmt.Sprintf("/bin/bash -x cmd.sh '%s' 2> %s", v[1: ], log_fpath)
            cmds = append(cmds, s)
        } else {
            s := fmt.Sprintf("python process_line.py '%s' 2> %s", v, log_fpath)
            cmds = append(cmds, s)
        }
    }
    cmd = strings.Join(cmds, "|") 
    return cmd
}

func EchoAjax(res http.ResponseWriter, req *http.Request) {
    str, _ := ioutil.ReadAll(req.Body)
    s := string(str)
    fmt.Printf("request. data:[%s]", s) 
    arr := strings.Split(s, "\n")
    fpath := arr[0]
    context := ""
    if len(arr) > 1{
        context = strings.Join(arr[1: ], "\n")
        fmt.Printf("get EchoAjax. context:[%s]", context) 
        Str_2_file(context, fpath, false)
    } else {
        context = File_2_str(fpath, -1)
        fmt.Printf("send EchoAjax. context:[%s]", context)
    }
    fmt.Fprintf(res, "%s", context)
}

func PathExists(path string) (bool, error) {
    _, err := os.Stat(path)
    if err == nil {
        return true, nil
    }
    if os.IsNotExist(err) {
        return false, nil
    }
    return false, err
}

func MkDir(dir string) bool {
    exist, err := PathExists(dir)
    if err != nil {
        return false
    }

    if exist {
        os.RemoveAll(dir)
    }
    err = os.Mkdir(dir, os.ModePerm)
    if err != nil {
        return false 
    } else {
        return true 
    }
}


func CmdAjax(res http.ResponseWriter, req *http.Request) {
    str, _ := ioutil.ReadAll(req.Body)
    s := string(str)
    arr := strings.Split(s, "\n")
    if len(arr) == 1 {
        arr = append(arr, "")
    }

    // 输入格式为: 命令 \n 数据文件路径 or 数据内容(tab或|分割)
    cmd := arr[0]
    ori_cmd := cmd
    context := strings.Join(arr[1:], "\n")
    sep := "\t"
    fpath := ""
    if strings.Index(context, "\t") != -1 {
        sep = "\t" 
    } else if strings.Index(context, "|") != -1 {
        sep = "|"
    } else if strings.Count(context, "\n") >= 1 {
        sep = "\t" 
    } else {
        fpath = context
    }

    cut := len(context)
    if cut > 100 {
        cut = 100
    }
    //fmt.Printf("process input.cmd:[%s], sep:[%s], fpath:[%s], context:[%s]\n", cmd, sep, fpath, context[: cut])
    fmt.Printf("process input.cmd:[%s], sep:[%s], fpath:[%s]\n", cmd, sep, fpath)

    rule, _ := regexp.Compile("[^a-zA-Z0-9_]")
    id := rule.ReplaceAllString(cmd, "_");
    if len(id) > 100{
        id = id[: 100] + "..."
    }
    id = fmt.Sprintf("%s_%s_%d", id, GetNowTime(), GetGID)

    input_fpath := fmt.Sprintf("static/temp/%s.input.txt", id)
    output_fpath := fmt.Sprintf("static/temp/%s.output.html", id)
    log_fpath := fmt.Sprintf("static/temp/%s.log.txt", id)
    if strings.Index(cmd, "chart_") == -1 && strings.Index(cmd, "html_") == -1{
        output_fpath = fmt.Sprintf("static/temp/%s.output.txt", id)
    }

    url := fmt.Sprintf("http://%s:%s", IP, HTTP_PORT) 
    input_url := fmt.Sprintf("%s/%s", url, input_fpath)
    output_url := fmt.Sprintf("%s/%s", url, output_fpath)
    log_url := fmt.Sprintf("%s/%s", url, log_fpath)

    if sep != "" {
        context = strings.Replace(context, sep, "\t", -1)
    }
   
    script := "" 
    cmd = parse_cmd(cmd, log_fpath)
    if strings.Index(cmd, "upload__") != -1 {
        reg := regexp.MustCompile(`(upload__[^ '\|]+)`)
        tag := reg.FindAllString(cmd, -1)[0][10:]
        files := str_2_json(context)
        if tag == "" {
            tag = "upload"
        }
        temp_dir := "static/temp/" + tag
        MkDir(temp_dir)
        fmt.Printf("begin to upload files:[%s]\n", tag)
        ret := "#!lcmd\n:cat " 
        for _, file := range files.([] interface {}) {
            fname := file.(map[string] interface {})["fname"].(string)
            //fcontext, _ := file.(map[string] interface {})["data"].([]byte)
            fcontext, _ := base64.StdEncoding.DecodeString(file.(map[string] interface {})["data"].(string))
            //fcontext, _ := file.(map[string] interface {})["data"].(string)
            //fcontext = u2s(fcontext)
            wpath := temp_dir + "/" + fname
            fmt.Printf("upload file. fname:%s, tag:%s\n", fname, tag)
            Byte_2_file(fcontext, wpath)
            ret +=  " " + wpath
        }
        fmt.Fprintf(res, "%s\n%s\n%s\n%s", "", "", "", ret)
        return
    } else if (len(fpath) >= 1 && fpath[: 1] ==  "/") || (len(fpath) >= 2 && fpath[: 2] == "./") { 
        script = fmt.Sprintf("cat %s | %s >  %s", fpath, cmd, output_fpath)
    } else if fpath != "" {
        script = fmt.Sprintf("wget '%s' -O %s &&  cat %s | %s >  %s", fpath, input_fpath, input_fpath, cmd, output_fpath) 
    } else if context == "" { 
        script = fmt.Sprintf("%s >  %s", cmd, output_fpath)
    } else {
        fmt.Printf("write input file. path:[%s], context:[%s]\n", input_fpath, context)
        Str_2_file(context, input_fpath, false)
        script = fmt.Sprintf("cat %s | %s > %s", input_fpath, cmd, output_fpath) 
    }
    fmt.Printf("run script. cmd:[%s]\n", script)
    r, _ := exec_shell(script)
    fmt.Printf("run script. log:[%s]\n", r)
    ret := File_2_str(output_fpath, 4000)
    fmt.Fprintf(res, "%s\t%s\t%s\t%s\t%s\n%s", ori_cmd, strings.Replace(script, "\t", " ", -1), input_url, output_url, log_url, ret)
    //fmt.Printf("response output.cmd:[%s], sep:[%s], fpath:[%s], context:[%s], res:[%s], output_fpath:[%s]\n", cmd, sep, fpath, context, ret, output_url)
    fmt.Printf("response output.cmd:[%s], sep:[%s], fpath:[%s], res:[%s], output_fpath:[%s]\n", cmd, sep, fpath, ret, output_url)
    is_check := true 
    for _, ch := range(ori_cmd) {
        if ch != '|' {
            is_check = false 
        }
    }
    fmt.Printf("ori_cmd:[%s], is_check:[%s]", ori_cmd, is_check)
    if ! is_check { 
        for i := 5; i >= 1; i-- { 
            dst := fmt.Sprintf("static/temp/pipeline.%d.txt", i)
            src := fmt.Sprintf("static/temp/pipeline.%d.txt", i - 1)
            Str_2_file(File_2_str(src, -1), dst, false)
        } 
        Str_2_file(ret, "static/temp/pipeline.0.txt", false) 
        history := fmt.Sprintf("%s\t%s\t%s\n", ori_cmd, input_fpath, arr[1])
        Str_2_file(history, "static/temp/cmd.history.txt", true) 
    }
}
