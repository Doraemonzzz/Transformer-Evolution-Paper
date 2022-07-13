# Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kern

论文地址：

- [https://arxiv.org/abs/1908.11775](https://arxiv.org/abs/1908.11775)



## 整体思路以及计算方式

整体思路是从Kernel的角度理解Attention，然后调整内积的计算方式：

- 输入$$X\in \mathbb R^{n\times d}$$，位置矩阵$$P\in \mathbb R^{n\times d}$$
- $$Q=XW_Q\in \mathbb R^{n\times d}, K = XW_K\in \mathbb R^{n\times d}$$
- $$Q_P=PW_{1}\in \mathbb R^{n\times d}, K_P = P W_2\in \mathbb R^{n\times d}$$
- 内积计算$$QK^\top + Q_P Q_K^{\top} \in \mathbb R^{n\times n}$$
- 剩余部分相同

作者还测试了一些Kernel的性能：

- $$f(x, y)= x^{\top} y$$（非常差）
- $$f(x, y)= (x^{\top} y)^2$$（很差）
- $$f(x, y)= \exp (x^{\top} y)$$（默认设置，效果不错）
- $$f(x, y)= \exp (-\|x -y \|^2)$$（效果不错，和前者接近）



## 时间复杂度

不变。



## 训练以及loss

不变。



## 代码





## 实验以及适用场景

适用于所有场景，该论文是分析性的文章，性能不是卖点。



## 细节

作者给出了permutation equivariant的严格定义，还讨论了Value中是否应该包含位置信息，结论是不一定需要。



## 简评

提供了一个新的角度，之前也看过这篇论文，但没细看，这次重读得到了一些新的信息。