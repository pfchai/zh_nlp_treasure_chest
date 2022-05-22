# 数据集

代码中用到的数据集，各数据集详情请看对应目录下`README.md`文件说明

## Text Classification

### 中文微博情感分类语料库

 - dataset: `weibo2018`
 - save path: `dataset/weibo2018`
 - description: @dengxiuqi 自己爬取标注的语料
 - labels: `二分类` `情感分类`

### 2018中国‘法研杯’法律智能挑战赛

#### 2018中国‘法研杯’法律智能挑战赛-罪名预测-练习赛

 - dataset: `cail2018_accu_e`
 - save path: `dataset/CAIL2018`
 - description: 183条罪名
 - labels: `多分类` `多标签`

#### 2018中国‘法研杯’法律智能挑战赛-罪名预测-第一阶段正赛

 - dataset: `cail2018_accu_fs`
 - save path: `dataset/CAIL2018`
 - description: 183条罪名
 - labels: `多分类` `多标签`

#### 2018中国‘法研杯’法律智能挑战赛-法条推荐-练习赛

 - dataset: `cail2018_ra_e`
 - save path: `dataset/CAIL2018`
 - description: 202条法条
 - labels: `多分类` `多标签`

#### 2018中国‘法研杯’法律智能挑战赛-法条推荐-第一阶段正赛

 - dataset: `cail2018_ra_fs`
 - save path: `dataset/CAIL2018`
 - description: 202条法条
 - labels: `多分类` `多标签`

#### 2018中国‘法研杯’法律智能挑战赛-刑期预测-练习赛

 - dataset: `cail2018_toi_e`
 - save path: `dataset/CAIL2018`
 - description: 刑期长短包括0-25年、无期、死刑
 - labels: `多分类`

#### 2018中国‘法研杯’法律智能挑战赛-刑期预测-第一阶段正赛

 - dataset: `cail2018_toi_fs`
 - save path: `dataset/CAIL2018`
 - description: 刑期长短包括0-25年、无期、死刑
 - labels: `多分类`

### 细粒度用户评论情感分析

[Sentiment-Analysis](https://github.com/foamliu/Sentiment-Analysis)


## Named Entity Recognition

### 微软亚洲研究院

 - dataset: `msra`
 - save path: `dataset/MSRA`
 - description: 微软亚洲研究院提供的中文命名实体识别数据集
 - labels: `NER`

### 人民日报

 - dataset: `people_daily`
 - save path: `dataset/People's Daily`
 - description: 人民日报中文命名实体识别数据集
 - labels: `NER`

### 微博

 - dataset: `weibo_ner`
 - save path: `dataset/Weibo_NER`
 - description: 微博命名实体识别数据集
 - labels: `NER`

## 待添加
 - dataset: ``
 - save path: ``
 - description:
 - labels:


## ChineseNlpCorpus 仓库中的数据集

数据来源 [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)

数据获取：

``` bash
cd $ZH_NLP_DEMO_PATH
git clone https://github.com/SophonPlus/ChineseNlpCorpus.git
```

数据说明详见 ChineseNlpCorpus 项目的说明

### 分类

| dataset | save path | lables | description |
| ------- | --------- | ------ | ----------- |
| `ChnSentiCorp_htl_all` | `dataset/ChineseNlpCorpus/datasets/ChnSentiCorp_htl_all` | ``| 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论。情感/观点/评论 倾向性分析 |
| `waimai_10k` | `dataset/ChineseNlpCorpus/datasets/waimai_10k` | `` | 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条 |
| `online_shopping_10_cats` | `dataset/ChineseNlpCorpus/datasets/online_shopping_10_cats` | `` | 10 个类别，共 6 万多条评论数据，正、负向评论各约 3 万条，包括书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店|
| `weibo_senti_100k` | `dataset/ChineseNlpCorpus/datasets/weibo_senti_100k` | `` | 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条 |
| `simplifyweibo_4_moods` | `dataset/ChineseNlpCorpus/datasets/simplifyweibo_4_moods` | `` | 36 万多条，带情感标注 新浪微博，包含 4 种情感，其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条|
| `dmsc_v2` | `dataset/ChineseNlpCorpus/datasets/dmsc_v2` | `` | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 |
| `yf_dianping` | `dataset/ChineseNlpCorpus/datasets/yf_dianping` | `` | 24 万家餐馆，54 万用户，440 万条评论/评分数据 |
| `yf_amazon` | `dataset/ChineseNlpCorpus/datasets/yf_amazon` | `` | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 |


### 命名实体识别

| dataset | save path | lables | description |
| ------- | --------- | ------ | ----------- |
| `dh_msra` | `dataset/ChineseNlpCorpus/datasets/dh_msra` | `` | 5 万多条中文命名实体识别标注数据（包括地点、机构、人物） |


### 推荐系统

| dataset | save path | lables | description |
| ------- | --------- | ------ | ----------- |
| `ez_douban` | `dataset/ChineseNlpCorpus/datasets/ez_douban` | `` | 5 万多部电影（3 万多有电影名称，2 万多没有电影名称），2.8 万 用户，280 万条评分数据 |
| `dmsc_v2` | `dataset/ChineseNlpCorpus/datasets/dmsc_v2` | `` | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 |
| `yf_dianping` | `dataset/ChineseNlpCorpus/datasets/yf_dianping` | `` | 24 万家餐馆，54 万用户，440 万条评论/评分数据 |
| `yf_amazon` | `dataset/ChineseNlpCorpus/datasets/yf_amazon` | `` | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 |

### FAQ 问答系统

| dataset | save path | lables | description |
| ------- | --------- | ------ | ----------- |
| `baoxianzhidao` | `dataset/ChineseNlpCorpus/datasets/baoxianzhidao` | `` | 保险知道：8000 多条保险行业问答数据 |
| `anhuidianxinzhidao` | `dataset/ChineseNlpCorpus/datasets/anhuidianxinzhidao` | `` | 安徽电信知道：15.6 万条电信问答数据 |
| `financezhidao` | `dataset/ChineseNlpCorpus/datasets/financezhidao` | `` | 金融知道: 77万 条金融行业问答数据 |
| `lawzhidao` | `dataset/ChineseNlpCorpus/datasets/lawzhidao` | `` | 法律知道：3.6 万条法律问答数据 |
| `liantongzhidao` | `dataset/ChineseNlpCorpus/datasets/liantongzhidao` | `` | 联通知道：20.3 万条联通问答数据 |
| `nonghangzhidao` | `dataset/ChineseNlpCorpus/datasets/nonghangzhidao` | `` | 农行知道： 4 万条农业银行问答数据 |
| `touzizhidao` | `dataset/ChineseNlpCorpus/datasets/touzizhidao` | `` | 投资知道： 58.8 万条投资行业问答数据 |


## 其他数据集

 - https://github.com/OYE93/Chinese-NLP-Corpus

