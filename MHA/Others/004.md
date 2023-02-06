# Ripple Attention for Visual Perception with Sub-quadratic Complexity

论文地址：

- [https://arxiv.org/abs/2110.02453](https://arxiv.org/abs/2110.02453)



## 整体思路以及计算方式

本文首先利用了Linear Attention，然后对Vit中的Attention提出局部性假设：每个$$q$$交互的$$k$$限制在某个范围内，利用动态规划算法计算该范围内的结果，然后计算加权和，整体计算式如下：
$$
O_{ij}=\frac{\phi\left(\mathbf{q}_{i j}\right)^{\top} \sum_{r=0}^{R}\alpha_{r}(i, j) \sum_{(m, n) \in \mathcal{N}_{r}(i, j)} \phi\left(\mathbf{k}_{m n}\right) \mathbf{v}_{m n}^{\top}}{\phi\left(\mathbf{q}_{i j}\right)^{\top} \sum_{r=0}^{R}\alpha_{r}(i, j) \sum_{\left(m^{\prime}, n^{\prime}\right) \in \mathcal{N}_{r}(i, j)} \phi\left(\mathbf{k}_{m^{\prime} n^{\prime}}\right)}
$$
这里的下标$$ij$$表示第$$i$$行，第$$j$$个patch，$$\mathcal{N}_{r}(i, j)$$表示：
$$
\mathcal{N}_{r}(i, j)=\{(m,n) | |m-i|+|n-j| \le r\}
$$
动态规划算法见论文。



## 时间复杂度

利用动态规划算法，时间复杂度可达$$O(nR)$$。



## 训练以及loss

不变。



## 代码

暂无。



## 实验以及适用场景

该Attention基于VIT设计，所以实验也是CV相关，总体效果还可以。



## 细节

反向传播也使用了DP。



## 简评

总体来说是个挺巧妙的算法，而且也可以向nlp任务扩展。