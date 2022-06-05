# Transformer with a Mixture of Gaussian Keys

论文地址：

- [https://arxiv.org/abs/2110.08678](https://arxiv.org/abs/2110.08678)



## 整体思路以及计算方式

作者对Attention中Softmax部分利用GMM替换，最后达到了相当的效果，这里回顾下技术细节。

全篇文章的出发点是如下假设：
$$
\mathbb{P}\left({q}_{i} | {t}_{j}=1\right)=\mathcal{N}\left({q}_{i} |{k}_{j}, \sigma_{j}^{2} {I}\right)
$$
其中等号左边的概率表示$q_i$和$k_j$有交互的概率（相当于Softmax中$q_i$和$k_j$对应的权重）。

通过该假设，以及$q_i, k_j$模长相等的假设，可以得到Softmax函数。

随后作者对上式进行推广，利用GMM可以拟合任意分布，作者假设：
$$
\mathbb{P}\left({q}_{i} | {t}_{j}=1\right)=\sum_r \pi_{jr}\mathcal{N}\left({q}_{i} |{k}_{jr}, \sigma_{jr}^{2} {I}\right)
$$
所以：
$$
\mathbb{P}\left({t}_{j}=1 | {q}_{i}\right)=\frac{\sum_{r} \pi_{j r} \exp \left(-\left\|{q}_{i}-{k}_{j r}\right\|^{2} / 2 \sigma_{j r}^{2}\right)}{\sum_{j^{\prime}} \sum_{r} \pi_{j^{\prime} r} \exp \left(-\left\|{q}_{i}-{k}_{j^{\prime} r}\right\|^{2} / 2 \sigma_{j^{\prime} r}^{2}\right)}
$$
最后的输出为：
$$
{h}_{i}=\sum_{j}\left(\frac{\sum_{r} \pi_{j r} \exp \left(-\left\|{q}_{i}-{k}_{j r}\right\|^{2} / 2 \sigma_{j r}^{2}\right)}{\sum_{j^{\prime}} \sum_{r} \pi_{j^{\prime} r} \exp \left(-\left\|{q}_{i}-{k}_{j^{\prime} r}\right\|^{2} / 2 \sigma_{j^{\prime} r}^{2}\right)}\right) {v}_{j}
$$
Linear版本：

上述方法可以推广到Linear Attention，唯一的区别就是增加了权重$\pi_{jr}$：
$$
{h}_{i}=\frac{\sum_{j} \sum_{r} \pi_{j r} \phi\left({q}_{i}\right)^{\top} \phi\left({k}_{j r}\right) {v}_{j}}{\sum_{j} \sum_{r} \pi_{j r} \phi\left({q}_{i}\right)^{\top} \phi\left({k}_{j r}\right)}=\frac{\phi\left({q}_{i}\right)^{\top} \sum_{j} \sum_{r} \pi_{j r} \phi\left({k}_{j r}\right) {v}_{j}^{\top}}{\phi\left({q}_{i}\right)^{\top} \sum_{j} \sum_{r} \pi_{j r} \phi\left({k}_{j r}\right)}
$$
学习策略：

具体的细节可以参考论文，主要是利用了EM算法。



## 时间复杂度

Vanilla版本为$O(N^2d)$，Linear版本为$O(Nd^2)$。



## 训练以及loss

不变。



## 代码

- [https://github.com/minhtannguyen/transformer-mgk](https://github.com/minhtannguyen/transformer-mgk)



## 实验以及适用场景

适用于所有场景，从结果来看提升并不明显。



## 细节

见代码。



## 简评

一个很好的思路，但是缺点也比较明显，性能基本没有提升，而且感觉学习的效率会降低。