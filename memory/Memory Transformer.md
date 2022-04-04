# Memory Transformer

论文地址：

- [https://arxiv.org/abs/2006.11527](https://arxiv.org/abs/2006.11527)



## 整体思路以及计算方式

思路比较简洁，对输入部分增加$m$个mem token，记为$x^{mem}\in \mathbb R^{m\times d}$，原始输入记为$x^{seq}\in \mathbb R^{n\times d}$，合并后的输入记为$x^{mem+seq}=[x^{mem}; s^{seq}]\in \mathbb R^{(n+m)\times d}$。

论文一共介绍了三个模型，分别为：

- Mem Transformer：
  $$
  x^{mem+seq}= \mathrm{MHA}(x^{mem+seq},x^{mem+seq},x^{mem+seq})
  $$

- MemCtrl Transformer：
  $$
  \begin{aligned}
  x^{mem}&= \mathrm{MHA}(x^{mem},x^{mem+seq},x^{mem+seq}) \\
  x^{seq}&= \mathrm{MHA}(x^{mem},x^{mem+seq},x^{mem+seq}) 
  \end{aligned}
  $$

- MemBottleNeck Transformer:
  $$
  \begin{aligned}
  x^{mem}&= \mathrm{MHA}(x^{mem},x^{mem+seq},x^{mem+seq}) \\
  x^{seq}&= \mathrm{MHA}(x^{mem},x^{mem},x^{mem}) 
  \end{aligned}
  $$



## 时间复杂度

依然是标准Attention的计算方式，所以时间复杂度为$O((n+m)^2 d)$。



## 训练以及loss

不变。



## 代码

[https://github.com/lucidrains/memory-transformer-xl](https://github.com/lucidrains/memory-transformer-xl)



## 实验以及适用场景

Encoder和Decoder均适用；实验比较全，Encoder, Decoder以及Encoder-Decoder结构均测试过，总体效果积极。



## 细节

论文中$m$取的比较小，所以增加的时间并不多。



## 简评

优点：

- 非常简洁清晰的方法；
- 适用于单向和双向模型；

总结

- 值得复现；