// note: 微服务，接收请求的命令和参数，返回结果 
// author: lichuan89@126.com
// date:   2019/01/01

package main

import (
    "bytes"
    "errors"
    "fmt"
    "html/template"
    "os/exec"
    "io/ioutil"
    "net/http"
    "runtime"
    "strings"
)

const (
    IP string = "localhost"
    HTTP_PORT  string = "8000"
)

func init() {
    runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
    fs := http.FileServer(http.Dir("static/"))
    http.Handle("/static/", http.StripPrefix("/static/", fs))

    http.HandleFunc("/", HomePage)
    http.HandleFunc("/ajax", OnAjax)

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

func HomePage(res http.ResponseWriter, req *http.Request) {
    t, err := template.ParseFiles("static/html/cmd.html")
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


func OnAjax(res http.ResponseWriter, req *http.Request) {
    str, _ := ioutil.ReadAll(req.Body)
    s := string(str)
    arr := strings.Split(s, "\x01")
    cmd := arr[0]
    args := ""
    if len(arr) >= 2{ 
        args = arr[1] 
    }
    fmt.Fprintf(res, "## 收到指令:\n[%s][%s]\n\n", cmd, args)
    ret, _ := exec_shell("/bin/bash cmd.sh '" + cmd + "' '" + args + "'")
    fmt.Fprintf(res, "## 返回结果:\n[%s]\n\n", ret)
}
