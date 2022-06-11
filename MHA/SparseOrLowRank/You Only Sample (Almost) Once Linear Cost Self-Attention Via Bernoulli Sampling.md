# You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling

论文地址：

- [https://arxiv.org/abs/2111.09714](https://arxiv.org/abs/2111.09714)



## 整体思路以及计算方式

利用LSH的方式减少Attention的计算量，核心公式如下：
$$
\sum_{j=1}^{n} \mathcal{B}(Q, K)_{i, j} V_{j}
$$
其中：
$$
\mathcal{B}(Q, K)_{i, j}=\mathbb{1}_{f\left(Q_{i}\right)=f\left(K_{j}\right)} \\
\mathbb{E}\left[\mathcal{B}(Q, K)_{i, j}\right]=\left(1-\frac{\arccos \left(Q_{i} K_{j}^{T}\right)}{\pi}\right)^{\tau}
$$


## 时间复杂度

$n m \tau \log _{2}(d)+n m d$，其中$m$是采样数量。



## 训练以及loss

不变。



## 代码

- [https://github.com/mlpen/YOSO](https://github.com/mlpen/YOSO)



## 实验以及适用场景

论文测试了Encoder的效果，从选择的实验结果来看，效果还不错，但是缺少和同类方法的对比。



## 细节

实现起来应该挺复杂的，不过作者提供了代码，可以研究下。



## 简评

LSH方法个人一直认为是思路简单，实现复杂的典型例子，个人不太喜欢这类方法，不过这篇写的还算容易懂，对LSH有兴趣的可以从这篇入手。

