// note: 建设从标准流读数据,往标准流写结果的通用工具 
// author: lilll@126.com
// date:   2019/01/01

package main 

import (
    "time"
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

type IObj interface{}
type IChannel chan interface{}
type IProcessLine func(line IObj, params ...IObj) IObj

func stdin_2_queues(in IChannel, done chan int, batch_num int) {
	f := bufio.NewReader(os.Stdin)
    i := 0
    arr := []string{}
	for {
		line, err := f.ReadString('\n')
		if err == io.EOF {
			break
		}
		line = strings.Replace(line, "\n", "", -1)
        arr = append(arr, line)
        if len(arr) >= batch_num{
            in <- arr
            arr = []string{} 
        }
        i += 1
	}
    if len(arr) > 0 {
        in <- arr 
    }
    done <- i 
}


func queues_2_worker(in IChannel, out IChannel, process_line IProcessLine, params ...IObj){
    for {
        lines := (<-in).([]string)
        outputs := []IObj {}
        for _, line := range(lines){
            data := process_line(line, params...)
            outputs = append(outputs, data)
        }
        out <- outputs
    }
}


func queue_2_stdout(out IChannel, done chan int) {
    i := 0
    finish_n := -1
    for {
        select {
            case msg := <-done:
                finish_n = msg
            case <-time.After(time.Second * 3):
                if finish_n == i{
                    break
                } else {
                    continue
                } 
            case msg := <- out:
                lines := msg.([]IObj)
                for _, line := range(lines){
                    fmt.Println(line.(string))
                    i += 1
                }
        }
        if finish_n == i{
            break
        }
    }
}



func Work(process_line IProcessLine, worker_num int, queue_len int, batch_num int, params ...IObj) {
    in := make(IChannel, queue_len)
    out := make(IChannel, queue_len)
    done := make(chan int)
    // 将输入流一行一行插入通道
    go stdin_2_queues(in, done, batch_num)
    // 从in通道取数据,处理完后放入out通道, 输入和输出的行数必须一致
    for i := 0; i<= worker_num; i++ {
        go queues_2_worker(in, out, process_line, params)
    }
    // 将处理结果输出到stdout
    queue_2_stdout(out, done)
}


// 测试: echo -e "Tom\tmale\t18\nLili\tfemale\t22" | go run exec.go
func test(){
    // 定义一行数据的处理函数, 将改行字符串按\t分割并调整字段顺序输出
    extract := func(line IObj, params ...IObj) IObj{
        data := line.(string)
        idxs := params[0].([]IObj)[0].([]int)
        arr := strings.Split(data, "\t")

        vec := []string{}
        for _, i := range idxs{
            vec = append(vec, arr[i])
        }
        return strings.Join(vec, "\t") 
    }
    // 定义字段调整后的顺序
    params := []int{2, 0}
    // 5个线程、50个元素的队列、每个元素打包100行数据
    Work(extract, 5, 50, 100, params) 
}
