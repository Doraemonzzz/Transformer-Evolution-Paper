# GMAT: Global Memory Augmentation for Transformers

论文地址：

- [https://arxiv.org/abs/2006.03274](https://arxiv.org/abs/2006.03274)



## 整体思路以及计算方式

整体思路是attention模块增加memory模块，为了验证memory模块有效性，作者使用chunk的方式计算attention（因为无法捕捉全局信息）。

整体计算方式如下：

- 输入：$X\in \mathbb R^{n\times d}$

- memory：$X_M\in \mathbb R^{m\times d}$

- 记：
  $$
  Y=\left[\begin{array}{l}
  X \\
  X_{M}
  \end{array}\right] \in \mathbb R^{(n+m)\times d}
  $$

- for $i=1,\ldots, n_1$:

  - $X_M=\mathrm{MHA}(X_M, Z)\in \mathbb R^{m\times d}$

- for $i=1,\ldots,n_2$:

  - $X_M=\mathrm{MHA}(X_M, X_M)\in \mathbb R^{m\times d}$

- for $i=1,\ldots,n_3$:

  - $X=\mathrm{MHA}(X, Z)\in \mathbb R^{n\times d}$

每个阶段的作用分别为：

- 第一阶段：压缩信息至memory；
- 第二阶段：编码memory信息；
- 第三阶段：解压缩信息；



## 时间复杂度

$O(m(m+n )d)$，其中$m$是memory长度。



## 训练以及loss

不变。



## 代码

- [https://github.com/ag1988/gmat](https://github.com/ag1988/gmat)



## 实验以及适用场景

适用于所有场景，可以带来一定提升。



## 细节

暂无。



## 简评

主要是验证这样训练设置下，memory的有效性，不过改方法应该会增加不少时间复杂度，所以是否值得有待商榷。