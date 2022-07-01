# Translational Equivariance in Kernelizable Attention

论文地址：

- https://arxiv.org/abs/2102.07680



## 整体思路以及计算方式

论文讨论的是在线性Attention中添加相对位置编码信息，整体思路如下。

首先回顾Attention计算方式，其中$$e_i$$是word embedding，$$u_i$$是position embedding：
$$
\begin{aligned}
\tilde{\mathbf{A}}_{i, j} &=\left\langle\left(e_{i}+u_{i}\right) \mathbf {W}_{q},\left(e_{j}+u_{j}\right) \mathbf {W}_{k}\right\rangle=\left(e_{i}+u_{i}\right) \mathbf {W}_{q} \mathbf {W}_{k}^{\top}\left(e_{j}+u_{j}\right)^{\top} \\
&=e_{i} \mathbf {W}_{q} \mathbf {W}_{k}^{\top} e_{j}^{\top}+e_{i} \mathbf {W}_{q} \mathbf {W}_{k}^{\top} u_{j}^{\top}+u_{i} \mathbf {W}_{q} \mathbf {W}_{k}^{\top} e_{j}^{\top}+u_{i} \mathbf {W}_{q} \mathbf {W}_{k}^{\top} u_{j}^{\top}
\end{aligned}
$$
论文的思路是对该计算方式进行重构，并且仍然能保持线性Attention的性质。

方案1：
$$
\begin{aligned}
\tilde{\mathbf{A}}_{i, j} 
&=e_{i} \mathbf {W}_{q} \mathbf {W}_{k}^{\top} e_{j}^{\top}

+u_{i} \mathbf {W}_{q} \mathbf {W}_{k}^{\top} u_{j}^{\top}
\end{aligned}
$$
其中：
$$
\begin{aligned}
\mathbf {W}_{q}^{*}
&=\operatorname{blockdiag}\left(\left[\begin{array}{cc}
\alpha_{1} & \beta_{1} \\
-\beta_{1} & \alpha_{1}
\end{array}\right], \ldots,\left[\begin{array}{cc}
\alpha_{m} & \beta_{m} \\
-\beta_{m} & \alpha_{m}
\end{array}\right]\right)\\
\mathbf {W}_{k}^{*}&=\mathbb{I}_{2 m}


\end{aligned}
$$
该方案的特点是，如果位置编码的形式为：
$$
u_x =\phi(x)=\left[\sin \left(\omega_{1} x\right), \cos \left(\omega_{1} x\right), \ldots, \sin \left(\omega_{m} x\right), \cos \left(\omega_{m} x\right)\right]
$$
那么满足如下性质：
$$
\left\langle u_{i-u}  \mathbf {W}_{q},u_{j-u} \mathbf {W}_{k}\right\rangle
=\left\langle u_{i}  \mathbf {W}_{q},u_{j} \mathbf {W}_{k}\right\rangle
$$
方案2：
$$
\tilde{\mathbf{A}}_{i j}=e_{i} \mathbf{W}_{q}\left(e_{j} \mathbf{W}_{k}+\mathbf{a}_{i j}\right)^{\top}=e_{i} \mathbf{W}_{q} \mathbf{W}_{k}^{\top} e_{j}^{\top}+e_{i} \mathbf{W}_{q} \mathbf{a}_{i j}^{\top}
$$
其中
$$
\begin{aligned}
\mathbf{a}_{i j} &=\mathbf{w}_{\operatorname{clip}(j-i, k)} \\
\operatorname{clip}(x, k) &=\max (-k, \min (k, x))
\end{aligned}
$$
clip函数的含义是将输入截断至$$[-k, k]$$之间。

如果使用普通的实现方式，那么时间复杂度为$$O(L^2 d)$$，但是注意到：
$$
\sum_{j=1}^{L}\left\langle\mathbf{q}_{i}^{\prime}, \mathbf{a}_{i j}^{\prime}\right\rangle \mathbf{v}_{j}=\left\langle\mathbf{q}_{i}^{\prime}, \mathbf{w}_{k}^{\prime}\right\rangle \sum_{j=1}^{L} \mathbf{v}_{j}+\sum_{m}\left\langle\mathbf{q}_{i}^{\prime}, \mathbf{a}_{i m}^{\prime} \right\rangle 
$$
可以将时间复杂度降低为$$O(Ld^2)$$，其中$$d$$为embedding维度。



## 时间复杂度

和线性Attention一致，依然为$$O(Ld^2)$$



## 训练以及loss

不变。



## 代码

- [https://github.com/ExpectationMax/Translational-Equivariant-Performers](https://github.com/ExpectationMax/Translational-Equivariant-Performers)



## 实验以及适用场景

适用于所有场景，实验只测试了图像任务，但是效果一般。



## 细节

暂无。



## 简评

总体感觉新意一般，效果也一般。