# Fast Transformers with Clustered Attention

论文地址：

- [https://arxiv.org/abs/2007.04825](https://arxiv.org/abs/2007.04825)



## 整体思路以及计算方式

对$$Q$$进行聚类，从而降低时间复杂度。

输入：

- $$Q\in \mathbb R^{n\times d}, K\in \mathbb R^{n\times d}, V\in \mathbb R^{n\times d}$$
- 聚类矩阵：$$S\in \{0,1\}^{n\times c}$$
- $$q_{j}^{c}=\frac{\sum_{i=1}^{N} s_{i j} q_{i}}{\sum_{i=1}^{N} s_{i j}}$$
- $$A^{c}=\operatorname{softmax}\left({Q^{c} K^{\top}}\right)\in \mathbb R^{c\times n}$$
- $$\bar O=A^{c} V\in \mathbb R^{c\times d}$$
- $$o_{i}=\sum_{j=1}^{c} s_{i j} \bar o_{j}$$

聚类方式见论文。



## 时间复杂度

$$O(ncd)$$。



## 训练以及loss

不变。



## 代码

- [https://github.com/idiap/fast-transformers](https://github.com/idiap/fast-transformers)



## 实验以及适用场景

作者跑了Encoder实验，Decoder部分需要适配。



## 细节

暂无。



## 简评

一个很简洁的思路，不过高效实现需要花一定的功夫，主要是聚类方式部分。