# Object-Centric Learning with Slot Attention

论文地址：

- [https://arxiv.org/pdf/2006.15055.pdf](https://arxiv.org/pdf/2006.15055.pdf)

参考资料：

- [https://zhuanlan.zhihu.com/p/344979830](https://zhuanlan.zhihu.com/p/344979830)



## 整体思路以及计算方式

对任务背景没有特别的了解，感觉是一种抽特征的方式，直接讨论计算方式，忽略Normlize相关部分：

- $$x\in \mathbb R^{N\times d_1}$$
- $$\text { slots } \sim \mathcal{N}(\mu, \operatorname{diag}(\sigma)) \in \mathbb{R}^{K \times d_2}$$
- for $$t=0,\ldots ,T-1$$:
  - $$\mathrm{slots}_{\mathrm{prev}}=\mathrm{slots}\in \mathbb R^{K\times d_2}$$
  - $$q= \mathrm{slots} W_q \in \mathbb R^{K\times d},k=x W_k\in \mathbb R^{N\times d}, v=xW_v \in \mathbb R^{N\times d}$$
  - $$\mathrm{attn}=\mathrm{Softmax}(qk^{\top} , \mathrm{dim}=0)\in \mathbb R^{K\times N}$$
  - $$\mathrm{updates=attn}.v\in \mathbb R^{K\times d}$$
  - $$\mathrm{slots= GRU(slots_{prev}, updates)} \in \mathbb R^{K\times d_2}$$



## 时间复杂度

$$\mathrm{MHA}$$的时间复杂度为$$O(KNd)$$，总时间复杂度为$$O(TKNd)$$。



## 训练以及loss

没有变化。



## 代码

- [https://github.com/lucidrains/slot-attention](https://github.com/lucidrains/slot-attention)
- [https://github.com/google-research/google-research/tree/master/slot_attention](https://github.com/google-research/google-research/tree/master/slot_attention)



## 实验以及适用场景

作者进行的实验比较简单，这里不进行讨论。



## 细节

略过。



## 简评

个人理解是一种抽特征的方式，不知道能否适用于NLP任务。

