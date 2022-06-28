# **Do Transformers Need Deep Long**-Range Memory?

论文地址：

- [https://arxiv.org/abs/2007.03356](https://arxiv.org/abs/2007.03356)



## 整体思路以及计算方式

之前Transformer中引入Memory的方法在每层中的Memory长度都一样，本文的方式是将Memory分为Long和Short，减少了一定的内存和计算开销，同时保证效果不会差太多。

备注，引入Memory的方式：

- Memory为向量$M\in \mathbb R^{m\times d}$；
- 输入为$X\in \mathbb R^{n\times d}$；
- 拼接为$Z=[X; M] \in \mathbb R^{(n+m)\times d}$
- 返回$\mathrm{MHA}(X,Z,Z)$



## 时间复杂度

时间复杂度为$O(n (n+m)d)$。



## 训练以及loss

不变。



## 代码

暂无，但实现起来也很简单。



## 实验以及适用场景

适用于Encoder，Decoder；论文里测试了lm，short/long memory结合的效果和全是long memory的效果相当。



## 细节

short/long memory结合使用的方式有3种：

- 前几层使用short memory
- 最后几层使用short memory
- short/long memory交替使用



## 简评

总体感觉一般，因为还是拿参数量换性能的方式。