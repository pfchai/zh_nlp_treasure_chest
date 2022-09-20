# -*- coding: utf-8 -*-

import torch
import numpy as np
from paddlenlp.datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
[train_examples, dev_examples, test_examples] = load_dataset('chnsenticorp', splits=('train', 'dev', 'test'))

print('训练集样本数量： ', len(train_examples))
print('验证集样本数量： ', len(dev_examples))
print('测试集样本数量： ', len(test_examples))
print('训练集样本示例：')
print(train_examples[0])


class MyDataset(Dataset):
    def __init__(self, examples, tokenizer):
        texts = [example['text'] for example in examples]
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=200)
        self.labels = [example['label'] for example in examples]

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


bert_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_name)

train_dataset = MyDataset(train_examples, tokenizer)
dev_dataset = MyDataset(dev_examples, tokenizer)
test_dataset = MyDataset(test_examples, tokenizer)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=2)
model.to(device)

# 优化方法
optim = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 训练函数
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新
        optim.step()
        scheduler.step()

        iter_num += 1
        if(iter_num % 100 == 0):
            print('epoch: %d, iter_num: %d, loss: %.4f, %.2f%%' % (epoch,
                                                                   iter_num, loss.item(), iter_num/total_iter*100))

    print('Epoch: %d, Average training loss: %.4f' % (epoch, total_train_loss/len(train_loader)))


def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in dev_loader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(dev_loader)
    print('Accuracy: %.4f' % (avg_val_accuracy))
    print('Average testing loss: %.4f' % (total_eval_loss / len(dev_loader)))
    print('-------------------------------')


for epoch in range(3):
    print('------------Epoch: %d ----------------' % epoch)
    train()
    validation()