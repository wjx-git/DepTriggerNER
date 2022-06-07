## Dependency Trigger NER

基于依存分析触发词的命名实体识别方法，适用于低资源场景，使用20%数据量可达到常规模型70%数据量效果。

依存分析触发词是指句子中与实体存在一跳或者两条依存句法分析关系的词。本文
使用Stanford CoreNLP工具自动标注触发词。

本文提出的方法与 ACL 2020 paper：TriggerNER: Learning with Entity Triggers as Explanations for Named Entity Recognition
具有相同的效果。

详情参见论文：[Low-Resource Named Entity Recognition Based on Multi-hop Dependency Trigger](https://arxiv.org/abs/2109.07118)

### requirement

1. torch==1.8.1
2. scikit-learn==0.24.2

### Trigger dataset

1. trigger_20.txt: 用20%原始数据标注的含有触发词的训练集；
2. trigger_prim_100.txt: 只包含一级触发词的数据集；
3. trigger_sec_100.txt: 只包含二级触发词的数据集；

### download
词向量下载：glove.6B.100d.txt， 并存放到gloveEN目录下。

### train

1. 运行supervised.py文件；代码基于 https://github.com/INK-USC/TriggerNER

