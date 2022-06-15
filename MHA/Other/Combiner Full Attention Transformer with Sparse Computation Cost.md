# Combiner Full Attention Transformer with Sparse Computation Cost

论文地址：

- [https://arxiv.org/abs/2107.05768](https://arxiv.org/abs/2107.05768)



## 整体思路以及计算方式

整体思路：

- vanilla Attention中，每个token和全部token交互；
- combiner将全部token分解为几组（其中有一个组只有一个token，其余为多个），每个token只和组中某个元素交互，从而减少计算量，是一种Sparse的方法；

思路其实不难，但实现比较复杂，具体如下。

该论文首先将Attention的计算理解为条件期望：
$$
A\left(x_{i}\right)=\mathbb{E}_{p(j \mid i)}\left[v_{j}\right], \quad p(j | i)=\frac{1}{Z\left(x_{i}\right)} \exp \left(\frac{q_{i}}{\sqrt{d}} k_{j}^{\top}\right) \tag 1
$$
然后将条件概率利用全概率公式进行分解：
$$
p(j | i)=\sum_{r=0}^{n_{i}} p\left(j, \Omega_{i}^{r} | i\right)=\sum_{r=0}^{n_{i}} p\left(j | \Omega_{i}^{r}, i\right) p\left(\Omega_{i}^{r} | i\right)=p\left(j | \Omega_{i}^{r_{j}}, i\right) p\left(\Omega_{i}^{r_{j}} | i\right) \tag 2
$$
其中$\Omega_i$表示$i$可取的全部集合全体，$\Omega_{i}^r$表示集合分解：
$$
\cup_{r=0}^{n_{i}} \Omega_{i}^{r}=\Omega_{i}, \Omega_{i}^{r} \cap \Omega_{i}^{s}=\varnothing, \forall r \neq s
$$
因为这里$i, j$都属于：
$$
[L]=\{k| 1\le k \le L, k\in \mathbb Z\}
$$
所以根据上述分解，有且仅有一个$r_j$，使得：
$$
p\left(j | \Omega_{i}^{r_{j}}, i\right) \neq 0
$$
将公式(2)带入(1)可得：
$$
\begin{aligned}
    A\left(x_{i}\right) 
    &=\mathbb{E}_{p(j | i)}\left[v_{j}\right]\\
    &=\sum_{r=0}^{n_{i}} \sum_{j \in \Omega_{i}^{r}} p\left(j, \Omega_{i}^{r} | i\right) v_{j} \\
    &=  \sum_{j \in \Omega_{i}^{r}} p\left(j, \Omega_{i}^{0} | i\right) v_{j}
    +\sum_{r=1}^{n_{i}} \sum_{j \in \Omega_{i}^{r}} p\left(j, \Omega_{i}^{r} | i\right) v_{j}\\
    &=\underbrace{\sum_{j \in \Omega_{i}^{0}} \tilde{p}(j | i) v_{j}}_{\text {direct expectation }}+\sum_{r=1}^{n_{i}} p\left(\Omega_{i}^{r} | i\right) \underbrace{\left(\sum_{j \in \Omega_{i}^{r}} p\left(j | \Omega_{i}^{r}\right) v_{j}\right)}_{\text {local expectation }}\\
 &= \sum_{j \in \Omega_{i}} \underbrace{\left[\mathbb{I}\left(j \in \Omega_{i}^{0}\right) \tilde{p}(j | i)+\sum_{r=1}^{n_{i}} \mathbb{I}\left(j \in \Omega_{i}^{r}\right) p\left(j | \Omega_{i}^{r}\right) p\left(\Omega_{i}^{r} | i\right)\right]}_{\text {the new effective conditional probability } q(j | i)} v_{j} \\   
 \end{aligned}
$$
中括号内有三项：

- $\tilde{p}(j | i) \propto \exp \left(\frac{q_{i}}{\sqrt{d}} k_{j}^{\top}\right)$
- $p\left(\Omega_{i}^{r} | i\right) \propto \exp \left(\frac{q_{i}}{\sqrt{d}} k_{\Omega_{i}^{r}}^{\top}\right)$
- $p\left(j | \Omega_{i}^{r}\right) \propto \exp \left(\frac{q_{\Omega_{i}^{r}}}{\sqrt{d}} k_{j}^{\top}\right)$

划分集合的方式见论文。



## 时间复杂度

$O(n\sqrt n)$或$O(n\log n)$。



## 训练以及loss

不变。



## 代码

- [https://github.com/google-research/google-research/tree/master/combiner](https://github.com/google-research/google-research/tree/master/combiner)



## 实验以及适用场景

总体来说效果还行，打败对手方法，但是无法完全超越Transformer。



## 细节

暂无。



## 简评

这篇论文提供的信息和其他Sparse Transformer类似，即Attention中只有部分计算是必要的，不过方法实现起来有点复杂。