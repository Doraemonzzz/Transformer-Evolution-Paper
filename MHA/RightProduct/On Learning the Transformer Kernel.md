# On Learning the Transformer Kernel

论文地址：

- [https://arxiv.org/abs/2110.08323](https://arxiv.org/abs/2110.08323)
- [https://openreview.net/forum?id=C7ViqmpuBl](https://openreview.net/forum?id=C7ViqmpuBl)



## 整体思路以及计算方式

Kernel法，使用了不同的kernel进行测试：
$$
\mathbf O=\phi(\mathbf Q)[\phi(\mathbf K^\top) \mathbf V]
$$


## 时间复杂度

$$O(nd^2)$$。



## 训练以及loss

不变。



## 代码

- [https://openreview.net/attachment?id=C7ViqmpuBl&name=supplementary_material](https://openreview.net/attachment?id=C7ViqmpuBl&name=supplementary_material)



## 实验以及适用场景

论文测试了encoder场景，性能可以相当。



## 细节

暂无。



## 简评

创新点不太多，但是可以学习下代码。