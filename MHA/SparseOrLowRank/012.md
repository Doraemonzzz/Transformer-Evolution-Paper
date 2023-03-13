# Memory-efficient Transformers via Top-k Attention

论文地址：

- [https://arxiv.org/abs/2106.06899](https://arxiv.org/abs/2106.06899)



## 整体思路以及计算方式

利用两点来减少计算：

- 利用top-k取每行attention score的前$$k$$个最大值；
- 通过chunk的方式减少内存的锋值；



## 时间复杂度

时间复杂度为$$O(nkd)$$，内存峰值为$$O(knd)$$。



## 训练以及loss

不变。



## 代码

- [https://github.com/ag1988/top_k_attention](https://github.com/ag1988/top_k_attention)



## 实验以及适用场景

适用于所有场景，效果尚可。



## 细节

暂无。



## 简评

算是一个工程优化，不过简单优雅。