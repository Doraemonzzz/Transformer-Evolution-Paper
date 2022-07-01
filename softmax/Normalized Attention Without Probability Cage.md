# Normalized Attention Without Probability Cage

论文地址：

- https://arxiv.org/abs/2005.09561



## 整体思路以及计算方式

本文对$$\mathrm{MHA}$$中$$\mathrm{SoftMax}$$归一化方式提出了疑问，通过理论，实验的方式证明了其他归一化方式也能达到相当的效果。原文测试了很多种方法，这里给出效果最好的一种$$\mathrm{Normalized Attention Pooling (NAP)}$$的计算方式：

- 给定$$q, k, v\in \mathbb R^{n\times d}$$

- 计算相似度$$s_{tr}= q_t^{\top} k_r \in \mathbb R$$

- 定义归一化函数：
  $$
  \begin{aligned}
  \operatorname{normalize}({x})_{j}&=g \cdot \frac{x_{j}-\mu_{{x}}}{\sigma_{{x}}}+b \\
  \mu_{{x}}&=\frac{1}{N} \sum_{j} x^{j}\\
  \sigma_{{x}}&=\frac{1}{N} \sum_{j}\left(x^{j}-\mu_{{x}}\right)^{2}
  \end{aligned}
  $$

- 按行归一化相似度：$$\mathrm{normalize}([(s_{t1},\ldots,s_{tn})])$$

- 剩余部分同$$\mathrm{MHA}$$计算



## 时间复杂度

只是换了归一化的方式，所以时间复杂度为$$O(n^2 d)$$。



## 训练以及loss

没有变化。



## 代码

- [https://github.com/lucidrains/all-normalization-transformer](https://github.com/lucidrains/all-normalization-transformer)
- [https://github.com/OliverRichter/normalized-attention](https://github.com/OliverRichter/normalized-attention)



## 实验以及适用场景

适用于Encoder, Decoder；原论文只做了验证实验，没有跑性能实验。



## 细节

暂无。



## 简评

总结：

- 讨论了$$\mathrm{SoftMax}$$是否必要的问题，从验证实验上来说不是必须的；
- 比较简单，值得复现；