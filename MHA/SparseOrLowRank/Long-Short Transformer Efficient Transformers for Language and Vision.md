# Long-Short Transformer: Efficient Transformers for Language and Vision

论文地址：

- [https://arxiv.org/abs/2107.02192](https://arxiv.org/abs/2107.02192)

参考资料：

- [https://blog.csdn.net/qq_43542339/article/details/118771339](https://blog.csdn.net/qq_43542339/article/details/118771339)



## 整体思路以及计算方式

利用Local Attention计算局部Attention(short-term)，利用降维的方式计算全局Attention(long-term)，最终达到降低时间复杂度的效果。

short-term计算方式：

![](../.Photo/Sparse_And_LowRank/2.jpg)

假设绿色部分长宽分别为$$l_1, l_2$$，那么总时间复杂度为$$O(l_1l_2 d \times n/l_1)=O(nl_2 d)$$

long-term计算方式：

- 给定$$Q\in \mathbb R^{n\times d}, K,V\in \mathbb R^{m\times d}$$
- $$W^{p} \in \mathbb R^{d\times r},P=\mathrm{Softmax}( K W^p)\in \mathbb R^{m\times r}$$
- $$\bar K = P^{\top}  K \in \mathbb R^{r\times d}, \bar V = P^{\top}  V \in \mathbb R^{r\times d}$$
- $$O=\mathrm{Softmax}(Q \bar K^{\top} ) \bar V \in \mathbb R^{n\times d}$$

总的时间复杂度为$$O((n+m)dr)$$。

融合：

- 记short-term对应的$$K, V$$分别为$$K_1, V_1\in \mathbb R^{w\times d}$$
- 记long-term对应的$$K, V$$分别为$$K_2, V_2\in \mathbb R^{r\times d}$$
- $$O=\mathrm{Softmax}(Q [\mathrm{LN}_1(K_1): \mathrm{LN}_2(K_2)]^{\top} )  [\mathrm{LN}_1(V_1): \mathrm{LN}_2(V_2)] \in \mathbb R^{n\times d}$$

总时间复杂度为$$O(n(r+w)d)$$。



## 时间复杂度

总时间复杂度为$$O(n(r+w)d)$$。



## 训练以及loss

不变。



## 代码

- [https://github.com/NVIDIA/transformer-ls](https://github.com/NVIDIA/transformer-ls)



## 实验以及适用场景

测试了lra, lm以及imagenet，效果都很好。



## 细节

单向模型中还有一些实现细节。



## 简评

效果挺好的，但是整体方法感觉不算优雅，一些实现细节可以参考。