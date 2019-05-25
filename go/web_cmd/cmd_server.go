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
    "io"
    "io/ioutil"
    "net/http"
    "runtime"
    "strings"
    "bufio"
    "time"
    "os"
    "log"
    "strconv"
)

const (
    IP string = "localhost"
    HTTP_PORT  string = "8000"
)

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

func Str_2_file(content string, fileName string) {
    /*  
    os.O_WRONLY | os.O_CREATE | O_EXCL  【如果已经存在，则失败】
    os.O_WRONLY | os.O_CREATE   【如果已经存在，会覆盖写，不会清空原来的文件，而是从头直接覆盖写】
    os.O_WRONLY | os.O_CREATE | os.O_APPEND  【如果已经存在，则在尾部添加写】
    */
    //创建文件,给与创、写权限，421:读写操
    filePointer, err := os.OpenFile(fileName, os.O_CREATE|os.O_WRONLY, 0751)
    if err != nil {
        log.Println("文件创建失败！错误：", err)
        return
    }   
    //创建成功挂起关闭文件流,在函数结束前执行
    defer filePointer.Close()
    //NewWriter创建一个以目标文件具有默认大小缓冲、写入w的*Writer。
    writer := bufio.NewWriter(filePointer)
    //写入器将内容写入缓冲。返回写入的字节数。
    size, err := writer.Write([]byte(content + "\n"))
    //Flush方法将缓冲中的数据写入下层的io.Writer接口。缺少，数据将保留在缓冲区，并未写入io.Writer接口
    writer.Flush()
    if err == nil {
        log.Println("文件创建并写入成功！字节数：", size)
    } else {
        log.Println("文件创建并写入失败！错误：", err)
    }   
}

func File_2_str(fileName string) string {
    //创建文件,给与创、写权限，421:读写操
    filePointer, err := os.OpenFile(fileName, os.O_RDONLY, 0751)
    if err != nil {
        log.Println("文件创建失败！错误：", err)
        return ""
    }
    //创建成功挂起关闭文件流,在函数结束前执行
    defer filePointer.Close()
 
    reader := bufio.NewReader(filePointer)
    lines := []string {}
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
    }
    return strings.Join(lines, "")
}

func init() {
    runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
    fs := http.FileServer(http.Dir("static/"))
    http.Handle("/static/", http.StripPrefix("/static/", fs))

    http.HandleFunc("/markdown", MarkDown)
    http.HandleFunc("/cmd", Cmd)
    http.HandleFunc("/cmd_ajax", CmdAjax)
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

func Cmd(res http.ResponseWriter, req *http.Request) {
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

func CmdAjax(res http.ResponseWriter, req *http.Request) {
    str, _ := ioutil.ReadAll(req.Body)
    s := string(str)
    arr := strings.Split(s, "\n")

    // 输入格式为: 命令 \n 分隔符(空表示tab) \n 数据文件路径 \n 数据内容
    cmd := arr[0]
    sep := arr[1]
    fpath := arr[2]
    context := strings.Join(arr[3:], "\n")
    fmt.Printf("process input.cmd:[%s], sep:[%s], fpath:[%s], context:[%s]\n", cmd, sep, fpath, context)

    id := fmt.Sprintf("%s_%s_%d", strings.Replace(cmd, "|", "__", -1), GetNowTime(), GetGID)
    input_fpath := fmt.Sprintf("static/temp/%s.input.txt", id)
    output_fpath := fmt.Sprintf("static/temp/%s.output.html", id)
    input_url := fmt.Sprintf("/%s", input_fpath)
    output_url := fmt.Sprintf("/%s", output_fpath)

    if sep != "" {
        context = strings.Replace(context, sep, "\t", -1)
    }
   
    script := "" 
    if fpath != "" {
        script = fmt.Sprintf("wget '%s' -O %s &&  cat %s | /bin/bash -x cmd.sh '%s' '%s'", fpath, input_fpath, input_fpath, cmd, output_fpath) 
    } else {
        fmt.Printf("write input file. path:[%s], context:[%s]\n", input_fpath, context)
        Str_2_file(context, input_fpath)
        script = fmt.Sprintf("cat %s | /bin/bash -x cmd.sh '%s' '%s'", input_fpath, cmd, output_fpath) 
    }
    fmt.Printf("run script. cmd:[%s]\n", script)
    r, _ := exec_shell(script)
    fmt.Printf("run script. log:[%s]\n", r)
    ret := File_2_str(output_fpath)
    fmt.Fprintf(res, "%s\n%s\n%s", input_url, output_url, ret)
    fmt.Printf("response output.cmd:[%s], sep:[%s], fpath:[%s], context:[%s], res:[%s], output_fpath:[%s]\n", cmd, sep, fpath, context, ret, output_url)
}
