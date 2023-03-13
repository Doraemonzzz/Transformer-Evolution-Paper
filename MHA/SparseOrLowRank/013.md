# Separable Self-attention for Mobile Vision Transformers

论文地址：

- [https://arxiv.org/abs/2206.02680](https://arxiv.org/abs/2206.02680)



## 整体思路以及计算方式

提出了一种Attention的计算方式，主要思想是将$$Q$$压缩为一个向量，这里带来的问题是，$$K$$交互的token变成了一个，所以应该会带来一些性能损失，计算公式如下：

- 输入：$$X\in \mathbb R^{n\times d}$$
- $$c_s = \mathrm{Softmax}(X W_1) \in \mathbb R^{n\times 1}$$
- $$X_k = XW_k \in \mathbb R^{n\times d}, X_v=XW_v \in \mathbb R^{n\times d}$$
- $$c_v= c_s^{\top} X_k \in \mathbb R^{1\times d}$$
- $$O_1=c_v \odot \mathrm{ReLU}(X_v)\in \mathbb R^{n\times d}$$
- $$O_2=O_1 W_o \in \mathbb R^{n\times d}$$



## 时间复杂度

$$O(nd^2)$$。



## 训练以及loss

不变。



## 代码

- [https://github.com/apple/ml-cvnets](https://github.com/apple/ml-cvnets)



## 实验以及适用场景

只适用于Encoder（$$c_s$$的计算），性能尚可，作者这里考虑的主要是效率，从效率角度来说却是不错。



## 简评

思路总体来说是很简单的，可以考虑适配到NLP任务中。