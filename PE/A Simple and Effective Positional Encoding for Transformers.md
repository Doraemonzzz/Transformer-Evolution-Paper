# A Simple and Effective Positional Encoding for Transformers

论文地址：

- [https://arxiv.org/abs/2104.08698](https://arxiv.org/abs/2104.08698)



## 整体思路以及计算方式

作者提出加性位置编码会增加矩阵的秩，具体来说，定义
$$
\begin{aligned}
    \mathrm{A}_{a}&=(\mathrm{X}+\mathrm{P}) \mathrm{W}_{Q} \mathrm{W}_{K}^{\top}(\mathrm{X}+\mathrm{P})^{\top}\\
    \mathrm{A}_{r}&=\mathrm{X} \mathrm{W}_{Q} \mathrm{W}_{K}^{\top} \mathrm{X}^{\top}+\hat{\mathrm{P}} \hat{\mathrm{P}}^{\top}
    \end{aligned}
$$
那么
$$
\inf (\mathrm{rank}(A_r)) > \mathrm{rank}(A_a)
$$
作者默认秩越大，性能越好，于是定义了两种位置编码方式：
$$
\begin{aligned}
\mathrm{A}_{i, j}^{\mathrm{ABS}} &=\left(\mathrm{X}_{i:} \mathrm{W}_{Q}\right)\left(\mathrm{X}_{j:} \mathrm{W}_{K}\right)^{\top} / \sqrt{d} +\left(\mathrm{P}_{Q} \mathrm{P}_{K}^{\top}\right)_{i, j}+E_{S}(S(i), S(j)) \\
\mathrm{A}_{i, j}^{\mathrm{REL}} &=\left(\mathrm{X}_{i:} \mathrm{W}_{Q}\right)\left(\mathrm{X}_{j:} \mathrm{W}_{K}\right)^{\top} / \sqrt{d} +\mathrm{R}_{i-j}+E_{S}(S(i), S(j)) \\
E_{S}(S(i), S(j))&=\mathrm{S}_{\hat{i}, \hat{j}}
\end{aligned}
$$
其中$$S$$为映射函数，$$\mathrm S$$为可学习的矩阵。



## 时间复杂度

理论复杂度不变，实际增加的计算开销微乎其微。



## 训练以及loss

不变。



## 代码

暂无，但是很好实现。



## 实验以及适用场景

适用于所有场景，可以带来一定提升，但是并不明显。



## 细节

暂无。



## 简评

个人感觉，本文的先验假设：过Softmax之前的矩阵秩越大，模型性能就越好这点站不住脚，因为即使是秩很小的矩阵，过完Softmax之后一般秩也会增加。