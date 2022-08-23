# zh_nlp_demo
中文NLP的一些经典算法实现


## 说明

作为NLPer，平时工作或者学习中会遇到各种各样的模型，本项目主要是用来记录自己觉着比较有意思的一些模型。


## 使用

在开始运行代码之前，请先执行

```shell
source initial_environment.sh
```

### Python 环境

如果你对本项目感兴趣，或是想要玩一玩项目中的代码，注意阅读对应目录下的 `README.md` 文件。

关于Python版本，我自己是使用的 Python3.6

```shell
conda create -n zh_nlp_demo python=3.6
```


### 相关第三方库依赖

用到的第三方库见 `requirements.txt`

```shell
pip install --upgrade pip
pip install -r requirements.txt
```

## ToDo

发现自己思路走歪了，之前想搞个大而全的，如果只有我个人维护，一方面自己用过的模型不多，经验欠缺；另一方面工作量也非常巨大。因此，转换思路，聚焦经典算法，数据集也不追求多，每种类型数据集各一种即可。

~~部分代码用到了PaddlePaddle，安装请参考[这里](https://www.paddlepaddle.org.cn/install/quick)~~
