// note: 建设处理文本的工具包 
// author: lilll@126.com
// date:   2019/01/01

package main 

import (
	"fmt"
    "os"
	"strings"
    "strconv"
    "crypto/sha256"
)


func cat(line IObj, params ...IObj) IObj{
    return line
}

func cut(line IObj, params ...IObj) IObj{
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

func sign(line IObj, params ...IObj) IObj{
    data := line.(string)
    idxs := params[0].([]IObj)[0].([]string)
    arr := strings.Split(data, "\t")

    idx, _ := strconv.Atoi(idxs[0])
    sign := ""
    if idx == -1 {
        sign = fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
    } else {
        sign = fmt.Sprintf("%x", sha256.Sum256([]byte(arr[idx])))
    }
    arr = append(arr, sign) 

    return strings.Join(arr, "\t") 
}

func Run(){
    opt := os.Args[1]
    params := os.Args[2: ]

    opts := map[string] IProcessLine{}
    opts["cat"] = cat
    opts["cut"] = cut
    opts["sign"] = sign 

    Work(opts[opt], 5, 10, params)
}

func main(){
    Run()
}
