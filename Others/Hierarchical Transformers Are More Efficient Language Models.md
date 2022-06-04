# Hierarchical Transformers Are More Efficient Language Models

论文地址：

- [https://arxiv.org/abs/2110.13711](https://arxiv.org/abs/2110.13711)



## 整体思路以及计算方式

对seqlen维度利用全连接进行downsample和upsample，从而降低时间复杂度。



## 时间复杂度

假设downsample的比例为$k$，那么时间复杂度为$O\left(\frac{n^2 d}{k^2}\right)$。



## 训练以及loss

不变。



## 代码

- [https://github.com/lucidrains/hourglass-transformer-pytorch](https://github.com/lucidrains/hourglass-transformer-pytorch)



## 实验以及适用场景

作者测试了LM，以及一些图像分类分类任务，均取得不错的效果。



## 细节

如何在downsample和upsample的同时保证信息不会泄露(LM情形)值得仔细看源码研究。



## 简评

整体思路是很简单的，类似Linformer，不过可以处理LM任务，值得复现。