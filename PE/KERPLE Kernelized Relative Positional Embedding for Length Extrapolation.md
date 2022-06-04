# KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation

论文地址：

- [https://arxiv.org/abs/2205.09921](https://arxiv.org/abs/2205.09921)



## 整体思路以及计算方式

本文利用PD kernel来构造相对位置编码，得到了非常好的外推效果（训练长度为512，inference长度为1024），定义这里不再复述，理一下论文思路：

- 相对位置编码形式：$k(m,n)=f(m-n)$；
- CPD kernel可以描述高维空间中的距离，这一点和相对位置编码很像，但是由于无法表述内积，所以和Attention无法兼容；
- CPD kernel通过平移可以转换为PD Kernel，即对于CPD kernel $\tilde k$，存在$c$，使得$c+\tilde k$为PD kernel，尽管$c$无法直接给出，但是由于Softmax的平移不变性，可以在计算的时候再使用；
- 常见的CPD kernel：
  - $\tilde{k}\left(x, x^{\prime}\right)=-a\left\|x-x^{\prime}\right\|^{p} \text { with } 0<p \leq 2 \text { and } a>0$
  - $\tilde{k}\left(x, x^{\prime}\right)=-b \cdot \log \left(1+a\left\|x-x^{\prime}\right\|^{p}\right) \text { with } 0<p \leq 2 \text { and } a, b>0$
- 实际计算公式：
  - $s_{m,n}=q_m^T k_n + \tilde k(m, n)$



## 时间复杂度

不变。



## 训练以及loss

不变。



## 代码

暂无，但是实现起来很简单。



## 实验以及适用场景

适用于所有场景，论文测了LM，结果是外推性非常好。



## 细节

暂无。



## 简评

非常好的想法，将理论和实际结合，这里给出两个小问题：

- 该方法只和Softmax兼容，实际上是否需要Softmax本身就是个问题；
- 为什么外推性比较好，没有给出理论或者直觉解释；

