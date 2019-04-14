// note: 建设处理文本的工具包 
// author: lilll@126.com
// date:   2019/01/01

package main 

import (
	"os"
	"strings"
    "strconv"
)


func cat(line IObj, params ...IObj) IObj{
    return line
}

func extract(line IObj, params ...IObj) IObj{
    data := line.(string)
    idxs := params[0].([]IObj)[0].([]string)
    arr := strings.Split(data, "\t")

    vec := []string{}
    for _, i := range idxs{
        ii, _ := strconv.Atoi(i)
        vec = append(vec, arr[ii])
    }
    return strings.Join(vec, "\t") 
}

func Run(){
    opt := os.Args[1]
    params := os.Args[2: ]

    opts := map[string] IProcessLine{}
    opts["cat"] = cat
    opts["extract"] = extract

    Work(opts[opt], 5, 10, params)
}

func main(){
    Run()
}
