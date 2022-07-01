# XCiT: Cross-Covariance Image Transformers

论文地址：

- [https://arxiv.org/abs/2106.09681](https://arxiv.org/abs/2106.09681)



## 整体思路以及计算方式

输入：

- $$X\in \mathbb R^{n\times d_1}$$
- $$Q,K,V=XW_Q, XW_K, XW_V\in \mathbb R^{n\times d_2}$$
- $$Q= \mathrm{Norm}(Q), K=\mathrm{Norm}(K)$$
- $$O=V\mathrm{Softmax}(K^{\top}  Q) W_o\in \mathbb R^{n\times  d_1}$$（分组计算）



## 时间复杂度

假设有$$h$$个分组，那么时间复杂度为$$O(n(d/h)^2\times h)=O(nd^2/h)$$。



## 训练以及loss

不变。



## 代码

- [https://github.com/facebookresearch/xcit](https://github.com/facebookresearch/xcit)



## 实验以及适用场景

适用于Encoder，作者进行了视觉任务，效果都不错。



## 细节

作者在Attention和FFN之间增加了一个模块，带来了不少提升，但是不加这个模块性能一般；另一方面，计算内积的同时增加了分组操作，这部分需要看源码。



## 简评

这篇思路过于简单，不知道该模块单独使用是否起作用。

