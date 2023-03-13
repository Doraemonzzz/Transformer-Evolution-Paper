# ChunkFormer: Learning Long Time Series with Multi-stage Chunked Transformer

论文地址：

- [https://arxiv.org/abs/2112.15087](https://arxiv.org/abs/2112.15087)



## 整体思路以及计算方式

思路非常简单，对序列进行分组(chunk)，每$$k$$个序列进行计算Attention，是一种Local Attention，如下图所示：

![](../.Photo/Sparse_And_LowRank/4.jpg)



## 时间复杂度

假设每组有$$k$$个token，那么总时间复杂度为$$O(k^2 n/k d)=O(nkd)$$。



## 训练以及loss

不变。



## 代码

暂无，但是实现起来不难。



## 实验以及适用场景

适用于所有场景，论文中应该测试了Encoder，效果尚可。



## 细节

不同层之间的chunk size可以不同。



## 简评

非常简洁的思路，不过如果作用在更难的任务，可能不太好用，因为没有全局信息交互。

