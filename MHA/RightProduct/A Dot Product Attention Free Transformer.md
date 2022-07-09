## A Dot Product Attention Free Transformer

论文地址：

- [https://openreview.net/forum?id=JVR4JswsEM](https://openreview.net/forum?id=JVR4JswsEM)



## 整体思路以及计算方式

利用点乘的方式计算Attention：

- 输入：$$\mathbf Q, \mathbf K, \mathbf V\in \mathbb R^{n\times d}, \mathbf p_1,\mathbf p_2\in \mathbb R^{n\times d_1}$$

- 计算$$\mathbf W= \mathbf p_1\mathbf p_2 ^\top \in \mathbb R^{n\times n}$$

- 输出：
  $$
  \mathbf o_{i}=\sigma_{q}\left(\mathbf q_{i}\right) \odot 
  \frac
  {\sum_{j=1}^{n} \exp \left(\mathbf k_{j}+w_{i,j}\right) \odot \mathbf v_{j}}
  {\sum_{j=1}^{n} \exp \left(\mathbf k_{j}+w_{i,j}\right)}
  $$



## 时间复杂度

$$O(d^2n + n^2d_1)$$



## 训练以及loss

不变。



## 代码

暂无，但是论文里有伪代码。



## 实验以及适用场景

Encoder和Decoder情形都进行了实验，总体来说效果还不错。



## 细节

因为没有计算Attention matrix，所以token之间的交互是通过$$\mathbf W$$矩阵。



## 简评

挺好的一个思路，可以考虑复现。