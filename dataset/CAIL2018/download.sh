#!/bin/bash


# 进入到对应目录
cd $ZH_NLP_DEMO_PATH/dataset/CAIL2018/

# 下载数据
wget https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip

# 解压数据
unzip CAIL2018_ALL_DATA.zip

cd final_all_data
wget https://raw.githubusercontent.com/thunlp/CAIL2018/master/baseline/law.txt
wget https://raw.githubusercontent.com/thunlp/CAIL2018/master/baseline/accu.txt
