# EL-Attention: Memory Efficient Lossless Attention for Generation

论文地址：

- [https://arxiv.org/abs/2105.04779](https://arxiv.org/abs/2105.04779)



## 整体思路以及计算方式

思路非常简单，降低infercence时间复杂度：

- 输入：$X\in \mathbb R^{n\times d}$
- $W_1= W_QW_K^T\in \mathbb R^{d\times d}, W_2= W_v W_o \in \mathbb R^{d\times d}$
- $S_1 = XW_1 X^T \in \mathbb R^{n\times n}(=QK^T)$
- $O_1=\mathrm{Softmax(S_1)}XW_2\in \mathbb R^{n\times d}$



## 时间复杂度

不考虑。



## 训练以及loss

不考虑。



## 代码

- [https://github.com/microsoft/fastseq/blob/main/examples/EL-attention/README.md](https://github.com/microsoft/fastseq/blob/main/examples/EL-attention/README.md)



## 实验以及适用场景

主要是用于inference，可以提升不少速度。



## 细节

暂无。



## 简评

非常好的思路，感觉可以尝试在training上。

