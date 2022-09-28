# zh_nlp_treasure_chest

中文NLP的一些经典算法实现，只专注于中文NLP领域的算法实现。


## 说明

作为NLPer，平时工作或者学习中会遇到各种各样的模型，本项目主要是用来记录自己觉着比较有意思的一些模型。

**代码结构随时可能会调整，不建议直接复制粘贴**，仅供参考各任务的实现方式。


## 使用

在开始运行代码之前，请先执行

```shell
source initial_environment.sh
```

### Python 环境

如果你对本项目感兴趣，或是想要玩一玩项目中的代码，注意阅读对应目录下的 `README.md` 文件。

关于Python版本，推荐使用 Python3.8 ，其他Python版本没有做过测试。

```shell
conda create -n zh_nlp python=3.8
conda activate zh_nlp
```


### 相关第三方库依赖

用到的第三方库见 `requirements.txt`


```shell
pip install --upgrade pip
pip install -r requirements.txt
```

## ToDo

部分代码用到了PaddlePaddle，安装请参考[这里](https://www.paddlepaddle.org.cn/install/quick)
