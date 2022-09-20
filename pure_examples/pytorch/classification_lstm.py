# -*- coding: utf-8 -*-

import random

import numpy as np
import torch 
import torch.nn as nn
from torchtext.legacy.data import Field, Dataset, Example, BucketIterator, Iterator
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from paddlenlp.datasets import load_dataset


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
emb_dim = 128
hidden_size = 128
num_layers = 2
num_classes = 2

[train_examples, dev_examples, test_examples] = load_dataset('chnsenticorp', splits=('train', 'dev', 'test'))

print('训练集样本数量： ', len(train_examples))
print('验证集样本数量： ', len(dev_examples))
print('测试集样本数量： ', len(test_examples))
print('训练集样本示例：')
print(train_examples[0])


tokenize = lambda x: list(x)
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = Field(sequential=False, use_vocab=False)


class MyDataset(Dataset):
    def __init__(self, examples, text_field, label_field, test=False):
        fields = [("id", None), ("text", text_field), ("label", label_field), ("length", None)]
        
        if test:
            new_examples = [Example.fromlist([None, example['text'], None, len(example['text'])], fields) for example in examples]
        else:
            new_examples = [Example.fromlist([None, example['text'], example['label'], len(example['text'])], fields) for example in examples]
        super().__init__(new_examples, fields)



train = MyDataset(train_examples, text_field=TEXT, label_field=LABEL, test=False)
valid = MyDataset(dev_examples, text_field=TEXT, label_field=LABEL, test=False)
test = MyDataset(test_examples, text_field=TEXT, label_field=None, test=True)

# 构建词表
TEXT.build_vocab(train)
len_vocab = len(TEXT.vocab)

train_iter, val_iter = BucketIterator.splits(
        (train, valid),
        batch_sizes=(batch_size, batch_size),
        device=str(device),
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False
)
test_iter = Iterator(test, batch_size=batch_size, device=str(device), sort=False, sort_within_batch=False, repeat=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(len_vocab, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, text):
        text_emb = self.embedding(text)
        output, _ = self.lstm(text_emb)
        text_fea = self.fc(output[0, :, :])
        text_fea = torch.squeeze(text_fea, 1)
        out = torch.softmax(text_fea, dim=1)
        return out


model = Model()
model.to(device)
criterition = torch.nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters())


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train():
    model.train()
    for bacth_idx, batch in enumerate(train_iter):
        data = batch.text.to(device)
        labels = batch.label.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterition(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (bacth_idx + 1) % 100 == 0:
            _, y_pred = torch.max(outputs, -1)
            acc = torch.mean((torch.tensor(y_pred == labels, dtype=torch.float)))
            print('epoch: %d \t batch_id : %d \t loss: %.4f \t train_acc: %.4f'
                %(epoch, bacth_idx+1, loss, acc))   


def validation():
    model.eval()
    total_eval_accuracy = 0
    for batch in val_iter:
        with torch.no_grad():
            # 正常传播
            data = batch.text.to(device)
            labels = batch.label.to(device)
            outputs = model(data)

        _, y_pred = torch.max(outputs, -1)
        acc = torch.mean((torch.tensor(y_pred == labels, dtype=torch.float)))
        total_eval_accuracy += acc

    avg_val_accuracy = total_eval_accuracy / len(val_iter)
    print('Accuracy: %.4f' % (avg_val_accuracy))
    print('-------------------------------')


for epoch in range(10):
    train()
    validation()