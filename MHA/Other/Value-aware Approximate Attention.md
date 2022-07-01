# Value-aware Approximate Attention

论文地址：

- [https://arxiv.org/abs/2103.09857](https://arxiv.org/abs/2103.09857)



## 整体思路以及计算方式

之前优化Attention的方式都是近似$$\mathrm{sim}(q, k)$$，在这篇工作中，作者指出，应该考虑整体，即近似：
$$
\frac{\sum_{i=1}^{n} \kappa\left(q, k_{i}\right) v_{i}}{\sum_{i=1}^{n}  \kappa\left(q, k_{i}\right)}
$$
定义$$o$$为vanilla attention的输出：
$$
o=\frac{\sum_{i=1}^{n} \kappa\left(q, k_{i}\right) v_{i}}{\sum_{i=1}^{n}  \kappa\left(q, k_{i}\right)}
$$
作者将考虑$$v$$的近似方式称为optimal-v-aware-r（OVA）。

对于OVA，作者考虑如下集合：
$$
C_{r}=\left\{\tilde o = \sum_{i=1}^{n} \beta_{i} v_{i}: \forall i \beta_{i} \geq 0, \sum_{i} \beta_{i}=1,\left|\left\{\beta_{i}: \beta_{i}>0\right\}\right| \leq r\right \}
$$
作者定义OVA-r​为：
$$
\operatorname{argmin}_{\tilde{o} \in C_{r}}\|o-\tilde{o}\|^{2}
$$
对于$$r \ge d+1$$情形，根据[Carathéodory定理](https://en.wikipedia.org/wiki/Carath%C3%A9odory's_theorem_(convex_hull))，必然存在$$\tilde o$$，使得：
$$
\tilde o = o
$$
对于$$r=1$$，那么：
$$
\tilde o = o_k,  k=\arg\min_{i} \|o-v_i \|^{2}
$$
另一方面，定义：
$$
\operatorname{att}_{\kappa, S}=\frac{\sum_{i \in S} \kappa\left(q, k_{i}\right) v_{i}}{\sum_{i \in S} \kappa\left(q, k_{i}\right)} \\
S \subseteq\{1, \ldots, n\},|S| \ll n
$$
对于这种方法，作者称其为optimal-v-oblivious-r（OVO）。

定义OVO-r为：
$$
S =\{a_1,a_2,\ldots, a_r| \kappa\left(q, k_{a_1}\right) \ge \kappa\left(q, k_{a_2}\right)
\ge \ldots \ge \kappa\left(q, k_{a_n}\right)\}
$$
作者的结论是对于相同的$$r$$，OVA-r比OVO-r的效果好。



## 时间复杂度

不考虑。



## 训练以及loss

不变。



## 代码

暂无。



## 实验以及适用场景

作者主要比较了OVA-r和OVO-r的效果，OVA-r效果更好。



## 细节

暂无。



## 简评

思路挺特别的，从整体角度考虑近似；另一方面，根据Carathéodory定理，假设存在最优的线性组合：
$$
o=\frac{\sum_{i=1}^{n} \kappa\left(q, k_{i}\right) v_{i}}{\sum_{i=1}^{n}  \kappa\left(q, k_{i}\right)}
$$
那么必然存在：
$$
\tilde o\in C_{d+1}=\left\{\tilde o = \sum_{i=1}^{n} \beta_{i} v_{i}: \forall i, \beta_{i} \geq 0, \sum_{i} \beta_{i}=1,\left|\left\{\beta_{i}: \beta_{i}>0\right\}\right| \leq d+1\right \}
$$
使得：
$$
\tilde o = o
$$
从这个角度来说，只需要稀疏性注意力，就能捕捉到关键信息。

