# Luna: Linear Unified Nested Attention

论文地址：

- [https://arxiv.org/abs/2106.01540](https://arxiv.org/abs/2106.01540)



## 整体思路以及计算方式

思路非常简单，利用MHA降维得到中间状态，然后再利用一个MHA计算最终结果，整体思路如下：

双向版本：

- 外部输入$$P\in \mathbb R^{l\times d}$$，输入$$X\in \mathbb R^{n\times d}$$
- $$Y_P= \mathrm{MHA}(P, X)\in \mathbb R^{l\times d}$$
- $$Y_X=\mathrm{MHA}(X,Y_P)\in \mathbb R^{n\times d}$$

单向版本：

- 定义
  $$
  \begin{aligned}
  X&\in \mathbb R^{n\times d_1}\\
  Y&\in \mathbb R^{n\times d_1} \\
  Z&\in \mathbb R^{n\times d_2}\\
  F &\triangleq f(X, Y, Z) \in \mathbb R^{n\times d_2}\\
  f_{t}&=\frac{1}{t} x_{t} \sum_{j=1}^{\top} y_{j}^{\top} z_{j}\in \mathbb R^{d_2}
  
  \end{aligned}
  $$

- $$A_{pack}=w(P X^{\top} )\in \mathbb R^{l\times n}$$

  - $$w$$可选1 + elu / softplus（不能按行使用Softmax，因为会有信息泄露）

- $$A_{uppack}=w(f(X, X,A_{pack}^{\top}))\in \mathbb R^{n\times l}$$

  - $$w$$可选Softmax（按行归一化）

- 输出$$Y=f(A_{uppack},A_{pack}^{\top}, X)\in \mathbb R^{n\times d}$$



## 时间复杂度

单向双向的时间复杂度都为$$O(nd^2)$$，但是单向版本本质上是RNN，速度会比较慢。



## 训练以及loss

不变。



## 代码

- [https://github.com/XuezheMax/fairseq-apollo](https://github.com/XuezheMax/fairseq-apollo)



## 实验以及适用场景

适用于所有场景，效果总的来说不错。



## 细节

暂无。



## 简评

这篇论文思路还是挺不错的，利用Attention来降维的思路也见到过很多次，然后单向版本的算法可以再仔细思考下。