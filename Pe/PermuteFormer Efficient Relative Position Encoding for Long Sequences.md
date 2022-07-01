# PermuteFormer: Efficient Relative Position Encoding for Long Sequences

论文地址：

- [https://arxiv.org/abs/2109.02377](https://arxiv.org/abs/2109.02377)



## 整体思路以及计算方式

整体思路是介绍一种适用于Linear Attention的相对位置编码方式。在Vanilla Attention中，因为会计算$$\mathrm S=\mathrm Q \mathrm K^{\top}$$，所以可以使用各种各样的相对位置编码。但是在Linear Attention中，因为不计算$$\mathbf S$$，所以可以使用的相对位置编码较少，本文就是解决这点，最后提供的方案为：
$$
\operatorname{sim} \left(\mathbf{q}_{i}, \mathbf{k}_{j}\right)=\left(r^{i} \mathbf{P}_{\pi}^{i} \mathbf {\phi}\left(\mathbf{q}_{i}\right)\right)^{\top}\left(r^{-j} \mathbf{P}_{\pi}^{j} \mathbf {\phi}\left(\mathbf{k}_{j}\right)\right)
$$
这里$$\mathbf P_{\pi}$$为置换矩阵，$$0<r<1$$，$$r$$的本意是想提供远程衰减性，但是个人认为这里的实现不太合理，因为$$r^{i-j}\neq r^{j-i}$$，不过论文似乎效果还不错。



## 时间复杂度

不考虑。



## 训练以及loss

不变。



## 代码

- [https://github.com/cpcp1998/PermuteFormer](https://github.com/cpcp1998/PermuteFormer)



## 实验以及适用场景

总体来说，都提升了Performer的效果。



## 细节

暂无。



## 简评

思路很巧妙，性能也不错，值得复现的工作。