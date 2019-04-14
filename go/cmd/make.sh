#!/bin/bash
# 编译
go build -o lcmd lcmd.go  lexec.go
# 测试
echo -e "Tom\tmale\t18\nLili\tfemale\t22" | ./lcmd  cat 
