#!/bin/bash
cp ../../python/common/*py ./
mkdir -p static/temp/
go run cmd_server.go 
