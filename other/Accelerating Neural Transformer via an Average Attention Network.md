# Accelerating Neural Transformer via an Average Attention Network

论文地址：

- [https://arxiv.org/abs/1805.00631](https://arxiv.org/abs/1805.00631)

参考资料：

- [https://blog.csdn.net/wwx123521/article/details/83238989](https://blog.csdn.net/wwx123521/article/details/83238989)
- [http://www.xuwei.io/2019/08/09/accelerating-neural-transformer-via-an-average-attention-network/](http://www.xuwei.io/2019/08/09/accelerating-neural-transformer-via-an-average-attention-network/)
- [https://zhuanlan.zhihu.com/p/77434191](https://zhuanlan.zhihu.com/p/77434191)



## 整体思路以及计算方式

替换Deocder中self Attention为AAN，计算方式如下：

- $y_j\in \mathbb R^{1\times d_1}$
- ${g}_{j}=\operatorname{FFN}\left(\frac{1}{j} \sum_{k=1}^{j} {y}_{k}\right)\in \mathbb R^{1\times d_2}$
- ${i}_{j}, {f}_{j}=\sigma\left(W\left[{y}_{j} ; {g}_{j}\right]\right)\in \mathbb R$
- $\tilde{{h}}_{j}={i}_{j} \odot {y}_{j}+{f}_{j} \odot {g}_{j} \in \mathbb R^{1\times d_1}$
- ${h}_{j}=\operatorname{LayerNorm}\left({y}_{j}+\tilde{{h}}_{j}\right)\in \mathbb R^{1\times d_1}$



## 时间复杂度

循环实现的时间复杂度为$O(nd_1 d_2)$，并行实现的时间复杂度为$O(n^2d_1 + nd_1 d_2)$。



## 训练以及loss

没有变化。



## 代码

- [https://github.com/bzhangGo/transformer-aan/blob/master/code/thumt/models/transformer.py](https://github.com/bzhangGo/transformer-aan/blob/master/code/thumt/models/transformer.py)



## 实验以及适用场景

适用于Causal Attention，可以替换LM中的Attention；论文测试了NMT实验，取得了相当的效果，但是没有速度提升。



## 细节

暂无。



## 简评

- 本质上和Attention类似，只不过假定等权重，但是不能提速；
- 可以在lm上测试；