# Compressive Transformers for Long-Range Sequence Modelling

论文地址：

- https://arxiv.org/abs/1911.05507



## 整体思路以及计算方式

计算方式：

- 构造记忆$m\in \mathbb R^{n_m\times d}$和压缩记忆$cm\in \mathbb R^{n_{cm}\times d}$；
- 对于输入$x\in \mathbb R^{n\times d}$，将记忆和压缩记忆拼接为整体记忆$\mathrm{mem}=\mathrm{concat}(cm, m,x) \in \mathbb R^{(n_m+ n_{cm}+n)\times d}$，得到输出$\mathrm{MHA}(x, \mathrm{mem}, \mathrm{mem}) \in \mathbb R^{n\times d}$。

记忆的更新方式为：

- 记忆：
  - 拼接，选择最近的$n_m$个记忆：$m =\mathrm{concat}(m, x)[-n_m:] \in \mathbb R^{n_m \times d}$
- 压缩记忆：
  - 对$m[:n]$的序列维度降维$c$倍得到$cm_{tmp}\in \mathbb R^{\left\lfloor\frac{n}{c}\right\rfloor \times d}$
  - 拼接，选择最近的$n_{cm}$个记忆：$cm=\mathrm{concat}(cm, cm_{tmp})[-n_{cm}:]\in \mathbb R^{n_{cm}\times d}$



## 时间复杂度

依然是标准Attention的计算方式，所以时间复杂度为$O(n_s(n_m+ n_{cm} +n) d)$。



## 训练以及loss

训练方式一致，loss部分增加了如下部分：
$$
\left\| \mathrm{MHA}(x, \text{old_mem}, \text{old_mem}) - \mathrm{MHA}(x, \text{mem}, \text{mem}) \right\|_2
$$
其中$\text{old_mem/mem}$表示$m,c_m$更新前/后拼接得到的整体记忆，应该是确保训练稳定。



## 代码

https://github.com/lucidrains/compressive-transformer-pytorch



## 实验以及适用场景

单向双向模型均适用；论文里只测试了lm（单向模型），效果有所提升。



## 细节

记忆和压缩记忆都不在计算图内，即不使用梯度方式更新。



## 简评

优点：

- 适用于单向和双向模型；
- 引入了记忆机制，提升了性能；

不足：

- 引入的记忆机制增增加了不少显存，时间复杂度也增加了；
- 压缩记忆的动机不够清晰；

总结：

- 是一种时间和空间换性能的方法，不会进行复现；