# Pay Attention to MLPs

论文地址：

- [https://arxiv.org/abs/2105.08050](https://arxiv.org/abs/2105.08050)



## 整体思路以及计算方式

该论文主要讨论Attention的必要性，提出了gMLP block，整体思路如下：

- 输入$X \in \mathbb R^{n\times d_1}$
- $X_1= \mathrm{Norm}(X)\in \mathbb R^{n\times d_1}$
- $X_2=f(X_1 W_1) \in \mathbb R^{n\times d_2}$
- $X_3= \mathrm{SGU}(X_2)W_2 \in \mathbb R^{n\times d_1}$

SGU有如下几个版本：

- $\mathrm{SGU}(Z)=f(Z)$
- $\mathrm{SGU}(Z)=Z+f(Z)$
- $\mathrm{SGU}(Z)=Z\odot f(Z)$
- $\mathrm{SGU}(Z)=Z_1\odot f(Z_2 ),Z=Z_{1} \| Z_{2}$

其中$f$表示：

- $f(Z)=\mathrm{Norm}(Z) W$

最后一个版本效果最好。



## 时间复杂度

$O(nd_1d_2)$，关于序列长度是线性的。



## 训练以及loss

不变。



## 代码

- [https://github.com/jaketae/g-mlp](https://github.com/jaketae/g-mlp)
- [https://github.com/lucidrains/g-mlp-pytorch](https://github.com/lucidrains/g-mlp-pytorch)



## 实验以及适用场景

进行了Encoder任务，在视觉任务上表现不错，在MLM上性能一般，不过在$\mathrm {SGU}$中添加Attention可以大幅提升性能（aMLP版本）。



## 细节

$\mathrm {SGU}$模块本质上是一个$\mathrm{GLU}$，而aMLP对应的$\mathrm {SGU}$则和FLASH模型非常相似（可以发现本文的一作也是FLASH的作者之一）。



## 简评

一个非常不错的思路，可以看到Attention在许多任务中不是必须的；不过该模块缺乏Token之间的交互，所以应该替换FFN更合适，可以进行这方面的实验。