#!/bin/bash

# 将当前目录添加到PYTHONPATH中
# source run.sh
export ZH_NLP_DEMO_PATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(dirname $(pwd))
