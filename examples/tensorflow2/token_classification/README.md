# 命名实体识别


CRF 的实现参考了项目 https://github.com/ngoquanghuy99/POS-Tagging-BiLSTM-CRF


其他可供参考的项目列表：
 - https://geeksrepos.com/topics/bilstm-crf
 - https://gitplanet.com/project/bi-lstm-crf-ner-tf20


## LSTM + CRF

代码见 `lstm_crf.py` 实现了基础的模型训练、测试、推理的能力。

运行示例

```bash
python lstm_crf.py \
    --dataset msra_ner
    --max_seq_length 128
    --batch_size 128
    --learning_rate 0.001
    --num_train_epochs 5
    --seed 42
    --do_train
    --do_test
    --do_predict
```


## Bert + LSTM + CRF

代码见 `bert_lstm_crf.py` 实现了基础的模型训练、测试、推理的能力。目前 Bert 不支持 finetune

运行示例

```bash
python bert_lstm_crf.py \
    --dataset msra_ner
    --max_seq_length 128
    --batch_size 128
    --learning_rate 0.001
    --num_train_epochs 2
    --seed 42
    --do_train
    --do_test
    --do_predict
```