# 文件夹介绍
## 1、config:配置文件夹
### default.yaml：训练相关参数，包括学习率、训练集文件地址、训练轮数等
### custom.yaml:  验证相关参数，包括学习率、训练集文件地址、训练轮数等
## 2、data:数据集文件夹
### loaddataset.py:下载数据集
### preprocess.py:预处理
### voc.py:VOC2012数据集操作类
## 3、experiments:训练日志
## 4、log:tensorboard（可忽略）
## 5、model_Picture:Graphviz可视化网络结构图片
## 6、models:模型结构
### [adaptive_attention.py](models%2Fadaptive_attention.py)：自适应注意力融合模块
### [augmented_edge_gcn.py](models%2Faugmented_edge_gcn.py)：边缘信息增强模块
### [candidate_region.py](models%2Fcandidate_region.py)：特征提取模块
### [dpt.py](models%2Fdpt.py)：模型主体
## 7、script：训练、验证、评估
### [evaluate.py](scripts%2Fevaluate.py)：评估
### [inference.py](scripts%2Finference.py)：验证
### [train.py](scripts%2Ftrain.py)：训练
## 8、utils：工具文件夹
### [init.py](utils%2Finit.py):初始化模型
### [logger.py](utils%2Flogger.py)：下载模型（暂时没有用）
### [metrics.py](utils%2Fmetrics.py)：准确率计算函数、计算平均交并比
### [visualization.py](utils%2Fvisualization.py)：可视化函数
## 使用方法
### 直接运行[train.py](scripts%2Ftrain.py)
