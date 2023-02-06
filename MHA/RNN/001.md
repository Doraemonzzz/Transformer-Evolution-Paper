# Linear Transformers Are Secretly Fast Weight Programmers

论文地址：

- [https://arxiv.org/abs/2102.11174](https://arxiv.org/abs/2102.11174)



## 整体思路以及计算方式

指出了Linear Attention有容量问题，据此对Linear Attention进行了修改。

更新规则：
$$
\begin{aligned}
\mathbf{k}_{i}, \mathbf{v}_{i}, \mathbf{q}_{i} &=\mathbf{W}_{k} \mathbf{x}_{i}, \mathbf{W}_{v} \mathbf{x}_{i}, \mathbf{W}_{q} \mathbf{x}_{i} \\
\overline{\mathbf{v}}_{i} &=\mathbf{W}^{(i-1)} \phi\left(\mathbf{k}_{i}\right) \\
\beta_{i} &=\sigma\left(\mathbf{W}_{\beta} \mathbf{x}_{i}\right) \\
\mathbf{v}'_{i} &=\beta_{i} \mathbf{v}_{i}+\left(1-\beta_{i}\right) \overline{\mathbf{v}}_{i}\\
\mathbf W_i&=\mathbf {W}_{i-1}+\beta^{(i)}\left(\mathbf {v}_{i}- \overline{\mathbf {v}}_{i}\right) \otimes \phi\left(\mathbf {k}_{i}\right)
\end{aligned}
$$

激活函数：
$$
\phi: \mathbb R^{d} \to \mathbb R^{2d\times \nu}
$$
其中：
$$
\phi_{i \nu}(\mathbf {k})=r\left(\left[\begin{array}{c}
\mathbf {k} \\
-\mathbf {k}
\end{array}\right]\right)_{i} r\left(\left[\begin{array}{c}
\mathbf {k} \\
-\mathbf {k}
\end{array}\right]\right)_{(i+\nu) \bmod 2d} \\
i=1,\ldots, 2d
$$
备注：这里省略了分母部分。



## 时间复杂度

$O(nd^2)$，由于使用了循环，计算比较慢。



## 训练以及loss

不变。



## 代码

- [https://github.com/ischlag/fast-weight-transformers](https://github.com/ischlag/fast-weight-transformers)
- [https://github.com/IDSIA/lmtool-fwp](https://github.com/IDSIA/lmtool-fwp)



## 实验以及适用场景

该方法是普适的。



## 细节

暂无。



## 简评

该工作属于LSTM之父的组，所以整个思路还是会向LSTM靠齐，由于无法并行，所以感觉方法一般，但是部分代码可以学习一下。