# 文本分类任务

## 数据集

1. weibo2018
2. CAIL2018

## 支持的模型架构

 - [x] LSTM
 - [x] BiLSTM
 - [x] BiLSTM + Attention
 - [x] CNN
 - [x] CNN+LSTM
 - [x] Bert
 - [ ] ERNIE 需要用到PaddlePaddle 且官方给的示例代码太啰嗦了，有时间再实现吧
 - [x] Bert + LSTM


## 模型架构

### LSTM
LSTM模型

使用
```
python keras/classification/lstm.py --do_train
```

### Bi-LSTM
双向LSTM模型

使用
```
python keras/classification/bi_lstm.py --do_train
```
