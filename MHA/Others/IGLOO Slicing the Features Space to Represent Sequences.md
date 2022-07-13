# IGLOO: Slicing the Features Space to Represent Sequences

论文地址：

- [https://arxiv.org/abs/1807.03402](https://arxiv.org/abs/1807.03402)



## 整体思路以及计算方式

引入了一个全新的计算Attention的方式，主要分为两个部分IGLOO-base和IGLOO-seq，原论文写的非常不清楚，所以这里按照自己的理解进行梳理。

IGLOO-base（记为$$f$$）：

- 输入：$$X \in \mathbb R^{n\times d}$$
- $$X_1 = \mathrm{Conv1d}(X)\in \mathbb R^{n\times d_1}$$
- 降采样：$$X_2= \mathrm{DownSample}(X_1)\in \mathbb R^{m\times d_1}$$
- 重复降采样$$l$$次得到：$$X_3 =\mathrm{Concat}([X_2]_1,\ldots, [X_2]_l)\in \mathbb R^{l\times m \times d_1}$$
- $$O_1=\mathrm{Sum}(X_3, d=1,2)\in \mathbb R^{l}$$
- 重复$$k$$次可得$$O\in \mathbb R^{k\times l}$$

IGLOO-seq：

- 输入：$$X\in \mathbb R^{n\times d}, Y \in \mathbb R^{n\times d}$$
- $$T_1=\mathrm{reshape}(f(Q))\in \mathbb R^{n\times 1 \times d_1}$$
  - $$k=n,l=d_1$$
- $$P=\mathrm{Softmax}(T_1)\in \mathbb R^{n\times 1 \times  d_1}$$
- $$T_2=Y W_1\in \mathbb R^{n\times d}$$
- $$T_3=\mathrm{repeat}(T_2)\in \mathbb R^{n\times d_1 \times d}$$
- 可学习矩阵：$$B\in \mathbb R^{n\times 1 \times d}$$
- $$T_4 = T_3\odot B \in \mathbb R^{n\times d_1 \times d}$$
- $$O_1=PT_4 \in \mathbb R^{n\times 1\times d}$$
- $$O_2=\mathrm{reshape}(O_1)\in \mathbb R^{n\times d}$$



## 时间复杂度

有点复杂，但关于$$n$$应该是线性复杂度。



## 训练以及loss

不变。



## 代码

- [https://github.com/redna11/lra-igloo](https://github.com/redna11/lra-igloo)



## 实验以及适用场景

作者做了一些实验，但是参数量无法对齐，所以有效性不太好说。



## 细节

暂无。



## 简评

论文写的非常不清楚，实验也不严格，是否有效需要验证。