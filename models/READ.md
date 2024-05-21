# 模型组件说明

本文件夹包含四个模型模块。以下是每个组件的详细说明：

## 1. Adaptive Attention Fusion Module

该模块用于自适应融合空间和通道注意力，以提高特征表示能力。

### 说明

- **输入**：特征图，形状为 `(batch_size, in_channels, height, width)`。
- **输出**：融合后的特征图，形状与输入相同。
- **功能**：通过融合空间和通道注意力，提高特征表示能力。

## 2. Augmented Edge Graph Convolution

该模块用于增强边缘特征的图卷积操作。

### 说明

- **输入**：特征图 `x` 和边缘索引 `edge_index`。
- **输出**：增强后的特征图。
- **功能**：通过图卷积操作，增强边缘特征。

## 3. Candidate Region Generator

该模块用于生成候选区域，通过聚类方法来划分特征图。

### 说明

- **输入**：特征图，形状为 `(batch_size, in_channels, height, width)`。
- **输出**：候选区域特征图，形状与输入相同。
- **功能**：通过 KMeans 聚类方法生成候选区域。

## 4. DPT (Dense Prediction Transformer)

该模块是一个集成了自适应注意力融合和增强边缘图卷积的密集预测模型。

### 说明

- **输入**：图像，形状为 `(batch_size, 3, height, width)`。
- **输出**：密集预测结果，形状为 `(batch_size, num_classes, height, width)`。
- **功能**：通过编码器提取特征，应用自适应注意力融合和增强边缘图卷积，并通过解码器生成最终的密集预测结果。