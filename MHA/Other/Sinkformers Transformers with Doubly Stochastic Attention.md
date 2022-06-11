# Sinkformers: Transformers with Doubly Stochastic Attention

论文地址：

- [https://arxiv.org/abs/2110.11773](https://arxiv.org/abs/2110.11773)



## 整体思路以及计算方式

论文观察了一个有趣的性质，Attention Matrix是按行归一化（$\sum_{j}p_{ij}=1$），但是训练后的模型大部分情形也能满足列归一化：$\sum_{i}p_{ij}=1$，于是作者提出了如下改进方式：

- 定义$\mathbf K^0=\exp(\mathbf C)\in \mathbb R^{n\times c}$

- 定义：

  -  行归一化：$\left(N_{R}(\mathrm K)\right)_{ij}=\frac{\mathrm K_{ij}}{\sum_{l=1}^{n} \mathrm K_{il} }$
  -  列归一化：$\left(N_{C}(\mathrm K)\right)_{ij}=\frac{\mathrm K_{ij}}{\sum_{l=1}^{n} \mathrm K_{lj} }$
  -  行归一化：$\left(N_{R}(\mathrm K)\right)_{ij}=K$

- 执行如下操作：
  
  $$
  \mathbf K^{l+1}= \begin{cases}N_{R}\left(\mathbf K^{l}\right) & \text { if } l \text { is even } \\ N_{C}\left(\mathbf K^{l}\right) & \text { if } l \text { is odd }\end{cases}
  $$

- 极限为：
  
  $$
  \mathbf K^{\infty}:=\operatorname{Sinkhorn}(\mathbf C)
$$
  
  并且满足：
  
  $$
  \begin{aligned}
  \mathbf K^{\infty} \mathbf 1_n &= \mathbf 1_n \\
  {\mathbf K^{\infty}} ^\top  \mathbf 1_n& = \mathbf 1_n 
  \end{aligned}
  $$

后续作者补充了一些理论证明，这里从略。



## 时间复杂度

增加了时间复杂度，但总体还是$O(n^2d)$。



## 训练以及loss

不变。



## 代码

- [https://github.com/michaelsdr/sinkformers](https://github.com/michaelsdr/sinkformers)



## 实验以及适用场景

提升的性能有限，主要是一个观察和理论证明。



## 细节

暂无。



## 简评

一个有意思的结论（按行归一化），感觉原因是许多场景下query和key是相同的向量，所以算出来的Attention Matrix是接近对称的，按行归一化会导致按列也接近归一化。
