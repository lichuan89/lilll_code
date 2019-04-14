#!/bin/bash
# 编译
go build -o lcmd lcmd.go  lexec.go
# 测试
echo -e "Tom\tmale\t18\nLili\tfemale\t22" > tmp.input
cat tmp.input | ./lcmd cat 
cat tmp.input | ./lcmd cut 2 0
cat tmp.input | ./lcmd sign -1 

