# Flowformer: Linearizing Transformers with Conservation Flows

论文地址：

- [https://arxiv.org/abs/2202.06258](https://arxiv.org/abs/2202.06258)

参考资料：

- [https://zhuanlan.zhihu.com/p/530502907](https://zhuanlan.zhihu.com/p/530502907)



## 整体思路以及计算方式

利用网络流的思路计算Attention。

输入：

- $$Q\in \mathbb R^{n\times d}, K\in \mathbb R^{m\times d}, V\in \mathbb R^{m\times d}$$
- $$Q=\phi(Q)\in \mathbb R^{n\times d}$$
- $$K=\phi(K)\in \mathbb R^{m\times d}$$
- Calculate incoming and outgoing flow
  - $$Q_{sum}=\mathrm{Sum}(Q,d=0) \in \mathbb R^{d}$$
  - $$K_{sum}=\mathrm{Sum}(K,d=0) \in \mathbb R^{d}$$
  - $$d_Q = 1/(Q K_{sum}^{\top})\in \mathbb R^{n}$$
  - $$d_K = 1/(K Q_{sum}^{\top})\in \mathbb R^{m}$$
- conservation refine for source and sink
  - $$t_Q= \mathrm{Sum}(K\odot d_K, d=0)\in \mathbb R^{d}$$
  - $$t_K= \mathrm{Sum}(Q\odot d_Q, d=0)\in \mathbb R^{d}$$
  - $$sink= Q \odot  t_Q \in \mathbb R^{n\times d}$$
  - $$source = K \odot t_K \in \mathbb R^{m\times d}$$
- Competition & Allocation
  - $$\alpha = \mathrm{Sigmoid}(sink) \in \mathbb R^{n\times d}$$
  - $$\beta= \mathrm{Softmax}(source) \in \mathbb R^{m\times d}$$
- dot product
  - $$Q_1 = Q\odot \alpha \in \mathbb R^{n\times d}$$
  - $$K_1 = Q\odot \beta \in \mathbb R^{m\times d}$$
  - $$O=\alpha \odot (Q_1(K_1^{\top}  V)) \in \mathbb R^{n\times d}$$



## 时间复杂度

理论上依然是$$O((n+m)d^2)$$，但是实际上应该不会太快。



## 训练以及loss

不变。



## 代码

- [https://github.com/thuml/Flowformer](https://github.com/thuml/Flowformer)



## 实验以及适用场景

测试了各种常见，总体来说性能都有提升。



## 细节

暂无。



## 简评

从理论和实验来说都还不错，是一篇不错的工作，但是计算的方式有点生硬，感觉并没有抓住问题的核心。