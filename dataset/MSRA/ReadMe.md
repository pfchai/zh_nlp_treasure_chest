# MSRA

微软亚洲研究院（Microsoft Research Asia）中文命名实体识别数据集


## 数据介绍

训练集数据: `msra_train_bio.txt`
测试集数据: `msra_test_bio.txt`

Tags: LOC(地名), ORG(机构名), PER(人名)
Tag Strategy：BIO
Split: '\t' (北\tB-LOC)

Data Size:

| 数据类型 | 句数 | 字符数  | LOC数 | ORG数 | PER数 |
| -------- | ---- | ------- | ----- | ----- | ----- |
| 训练集  | 45000 | 2171573 | 36860 | 20584 | 17615 |
| 测试集  | 3442  | 172601  | 2886  | 1331  | 1973  |


## 来源

https://github.com/OYE93/Chinese-NLP-Corpus

参考
[The third international Chinese language processing bakeoff: Word segmentation and named entity recognition](https://faculty.washington.edu/levow/papers/sighan06.pdf)
