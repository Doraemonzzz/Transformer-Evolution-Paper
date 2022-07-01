# Synthesizer: Rethinking Self-Attention in Transformer Models

论文地址：

- [https://arxiv.org/abs/2005.00743](https://arxiv.org/abs/2005.00743)



## 整体思路以及计算方式

本文的观点很有意思：Attention中的Attention Matrix可以不通过Query和Key计算，通过其他方式得到$$n\times n$$矩阵也能得到合理的结果，模型一共有5种，分别如下：

Dense:

- 输入：$$X\in \mathbb R^{n\times d}$$
- $$S_1= f(XW_1) \in \mathbb R^{n\times d_1}$$
- $$S_2= S_1 W_2 \in \mathbb R^{n\times n}$$
- $$O=\mathrm{Softmax}(S_2)(XW_3)\in \mathbb R^{n\times d}$$

Random：

- 输入：$$X\in \mathbb R^{n\times d}$$
- 初始化：$$S_1 \in \mathbb R^{n\times n}$$
- $$O=\mathrm{Softmax}(S_1 )(XW_1)\in \mathbb R^{n\times d}$$

Factorized Dense（降低空间复杂度）：

- 输入：$$X\in \mathbb R^{n\times d}$$
- 初始化：$$W_1\in \mathbb R^{d\times k_1}, W_2\in \mathbb R^{d\times k_2}$$，$$k_1 \times k_2 = n$$
- $$S_1= (XW_1) \odot (XW_2) \in \mathbb R^{n\times n}$$（复制到相同维度，然后点乘）
- $$O=\mathrm{Softmax}(S_1)(XW_3)\in \mathbb R^{n\times d}$$

Factorized Random（降低空间复杂度）：

- 输入：$$X\in \mathbb R^{n\times d}$$
- 初始化：$$S_1\in \mathbb R^{n\times k}, S_2\in \mathbb R^{n\times k}$$
- $$O=\mathrm{Softmax}(S_1 S_2^{\top})(XW_1)\in \mathbb R^{n\times d}$$

Mixture：

- 输入：$$X\in \mathbb R^{n\times d}$$
- 通过不同的方式计算Score Matrix $$S$$：
  - $$S=\sum\alpha_i S_i, \sum \alpha_i =1$$
- $$O=\mathrm{Softmax}(S)(XW)\in \mathbb R^{n\times d}$$



## 时间复杂度

本质上还需要计算Attention Matrix，所以依然为$$O(n^2d)$$



## 训练以及loss

不变。



## 代码

- [https://github.com/leaderj1001/Synthesizer-Rethinking-Self-Attention-Transformer-Models](https://github.com/leaderj1001/Synthesizer-Rethinking-Self-Attention-Transformer-Models)



## 实验以及适用场景

适用于所有场景，性能最好的是Mixture情形，这里使用了$$QK^{\top}$$，所以即使提升性能，其实意义并不大。



## 细节

模型参数和序列长度有关，如果所以要处理很长的序列，模型应该会非常大。



## 简评

提供了一个不同的视角，但是从实验上来说，不利用$$\mathbf Q \mathbf K^{\top}$$的情形性能其实比较差。

