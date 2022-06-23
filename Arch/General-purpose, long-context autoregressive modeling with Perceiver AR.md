# General-purpose, long-context autoregressive modeling with Perceiver AR

论文地址：

- https://arxiv.org/abs/2202.07765



## 整体思路以及计算方式

基本上同Perceiver，将模型拓展为可以处理单向数据，唯一的区别是将输入拆分为：

- $X=[X_1, X_2]$
  - $X \in \mathbb R^{n\times d}$
  - $X_1\in \mathbb R^{(n-m)\times d}$
  - $X_2\in \mathbb R^{m\times d}$
- $Y_1= \mathrm{MHA}(X_2, X_1)\in \mathbb R^{m\times d}$（with mask）
- $O=\mathrm{MHA}(Y_1, Y_1)\in \mathbb R^{m\times d}$（with mask）

其余部分同Perceiver。



## 代码

- [https://github.com/lucidrains/perceiver-ar-pytorch](https://github.com/lucidrains/perceiver-ar-pytorch)



## 简评

是否可以将该方法推广为一种预训练方式？