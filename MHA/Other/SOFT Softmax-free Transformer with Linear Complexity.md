# SOFT: Softmax-free Transformer with Linear Complexity

论文地址：

- [https://arxiv.org/abs/2110.11945](https://arxiv.org/abs/2110.11945)

参考资料：

- [https://zhuanlan.zhihu.com/p/427028271](https://zhuanlan.zhihu.com/p/427028271)



## 整体思路以及计算方式

首先加Attention Matrix的计算方式改写为（忽略常数）：
$$
S_{ij}= \exp \left(-\|q_i-k_j \|^2\right)
$$
记为：
$$
S=\exp (Q \ominus K)
$$
由于时间复杂度没有降低，其实并无意义，后续的操作是降低时间复杂度。

作者首先假设$Q=K$，那么$S$变成对称矩阵，将其表示为：
$$
S=\left[\begin{array}{cc}
A & B \\
B^{\top} & C
\end{array}\right] \in \mathbb{R}^{n \times n}
$$
根据对称性，可以利用Nystrom分解进行计算：
$$
\hat{S}=\left[\begin{array}{c}
A \\
B^{\top}
\end{array}\right] A^{\dagger}\left[\begin{array}{ll}
A & B
\end{array}\right]=P^{\top} A^{\dagger} P,P=\left[\begin{array}{ll}
A & B
\end{array}\right]\\
A\in \mathbb R^{m\times m}, B\in \mathbb R^{m\times (n-m)}, C\in \mathbb R^{(n-m)\times (n-m)}, m \ll n
$$
其中$A^{\dagger}$是$A$的Moore-Penrose逆矩阵。

后续的做法是，通过采样的方式得到$\tilde Q,\tilde K$（降维），然后计算
$$
A=\exp (\tilde{Q} \ominus \tilde{K}), \quad P=\exp (\tilde{Q} \ominus K)
$$
最后可得：
$$
\hat{S}=\exp (Q \ominus \tilde{K})(\exp (\tilde{Q} \ominus \tilde{K}))^{\dagger} \exp (\tilde{Q} \ominus K)
$$
伪逆可以利用现成的方法计算。



## 时间复杂度

$O(mnd)$，如果$m$较小，可视为线性。



## 训练以及loss

不变。



## 代码

- [https://github.com/fudan-zvg/SOFT](https://github.com/fudan-zvg/SOFT)



## 实验以及适用场景

主要实验是CV相关，感觉该方法也可以使用到NLP中。



## 细节

暂无。



## 简评

该论文提供了一个视角，$Q$是否可以和$K$相同，在self attention中，似乎对性能不会有损失，这也是后续可以研究的点。