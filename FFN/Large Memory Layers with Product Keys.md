# Large Memory Layers with Product Keys

论文地址：

- [https://arxiv.org/abs/1907.05242](https://arxiv.org/abs/1907.05242)

参考资料：

- [https://zhuanlan.zhihu.com/p/76501184](https://zhuanlan.zhihu.com/p/76501184)
- [https://jishuin.proginn.com/p/763bfbd60db9](https://jishuin.proginn.com/p/763bfbd60db9)
- https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb
- https://www.pragmatic.ml/large-memory-layers-with-product-keys/



## 整体思路以及计算方式

对Transformer中$\mathrm{FFN}$的改进，称为$\mathrm{PKM}$（Product Key Memory），注意这里也有$q,k,v$，思路是值得借鉴的。

首先给出符号：

- $\mathcal T_m$表示$\mathrm{Top}-m$
- $k\in \mathcal K, k\in \mathbb R^{1\times d}, |\mathcal K|= n$
- $q(x),v_i \in \mathbb R^{1\times d}$

核心为如下计算问题：
$$
\begin{aligned}
\mathcal{I} &=\mathcal{T}_{m}\left(q(x)^{T} k_{i}\right) \\
w &=\operatorname{Softmax}\left(\left(q(x)^{T} k_{i}\right)_{i \in \mathcal{I}}\right) \\
m(x) &=\sum_{i \in \mathcal{I}} w_{i} v_{i}
\end{aligned}
$$
注意该模块依然为$\mathbb R^d\to \mathbb R^d$的映射，所以可以类比$\mathrm {FFN}$。

分析：

- 第一步需要计算$\mathcal{T}_{m}(qk^T)\in \mathbb R^{1\times m},k\in \mathcal K$，
  - 由于需要求出全部$n$项，每一项的计算复杂度为$O(d)$，所以总计算复杂度为$O(nd)$
  - $\mathcal T_m$操作的时间复杂度为$O(m\log n)$
- 第二步的时间复杂度为$O(m)$
- 第三步的时间复杂度为$O(md)$

由于第一步是主要开销，为了提速，论文里做了如下假设：

- $k\in \mathcal K=\{(c, c')| c\in \mathcal C, c'\in \mathcal C\}$

  - 这里$(c,c')$表示向量拼接，$c,c'\in \mathbb R^{1\times d/2}$
  - $|c|=|c'|= \sqrt{n}$

注意到：
$$
\begin{aligned}
qk^T 
&= q[:d/2] k[:d/2]^T + q[d/2:] k[d/2:]^T \\
&\triangleq q^{(1)} (k^{(1)})^T + q^{(2)} (k^{(2)})^T


\end{aligned}
$$
结合假设：
$$
q^{(1)} (k^{(1)}_i)^T, q^{(2)} (k^{(2)}_j)^T,k^{(1)}_i\in c, k^{(2)}_j\in c'
$$
所以：

- 只要求出$2\sqrt n$项即可，每一项的计算复杂度为$O(d/2)$，所以总计算复杂度为$O(\sqrt nd)$

接着要从$2\sqrt n$项中恢复$qk_T$，作者使用的方式为：
$$
qk^T=\{q^{(1)} (k^{(1)}_i)^T, q^{(2)} (k^{(2)}_j)^T|k^{(1)}_i\in c, k^{(2)}_j\in c' \}
$$
这里一共有$\sqrt n\times  \sqrt n =n$个元素，从这$n$个元素中进行$\mathcal{T}_{m}$运行即可，因此总时间复杂度为
$$
O(\sqrt nd +m\log n + md +d ) = O((\sqrt n+m)d)
$$
备注，这里假设$\log n < d$。



## 时间复杂度

假设$x\in \mathbb R^{L\times d}$，所以总时间复杂度为：
$$
O(L (\sqrt n+m)d)
$$
注意到$\mathrm{FFN}$的时间复杂度为：
$$
O(4Ld^2)
$$
所以一般来说前者比$\mathrm{FFN}$快。



## 训练以及loss

保持一致。



## 代码

- [https://github.com/lucidrains/product-key-memory](https://github.com/lucidrains/product-key-memory)



## 实验以及适用场景

因为是替换FFN，所以适用于所有场景，但是这样做的动机还不明确；从实验效果来说非常不错。



## 细节

实现细节需要看查看官方代码。



## 简评

总结：

- 思路挺特别的，而且效果出人意料的好；
- 值得复现；