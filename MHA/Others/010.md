# ETSformer: Exponential Smoothing Transformers for Time-series Forecasting

论文地址：

- [https://arxiv.org/abs/2202.01381](https://arxiv.org/abs/2202.01381)



## 整体思路以及计算方式

针对时间序列问题的特点，提出了Exponential Smoothing Attention和Frequency Attention，这里主要讨论Exponential Smoothing Attention。

- 输入：$$V\in \mathbb R^{n\times d}$$

- 输出：$${A}_{\mathrm{ES}} \cdot\left[\begin{array}{c}
  \mathbf {v}_{0}^{\top} \\ {V}
  \end{array}\right]\in \mathbb R^{n\times d}$$，其中
  $$
  {A}_{\mathrm{ES}}=\left[\begin{array}{cccccc}
  (1-\alpha)^{1} & \alpha & 0 & 0 & \ldots & 0 \\
  (1-\alpha)^{2} & \alpha(1-\alpha) & \alpha & 0 & \ldots & 0 \\
  (1-\alpha)^{3} & \alpha(1-\alpha)^{2} & \alpha(1-\alpha) & \alpha & \ldots & 0 \\
  \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
  (1-\alpha)^{n} & \alpha(1-\alpha)^{n-1} & \ldots & \alpha(1-\alpha)^{j} & \ldots & \alpha
  \end{array}\right]
  $$

利用论文提出的算法，可以再$$O(nd\log n)$$中完成。



## 时间复杂度

$$O(nd\log n)$$。



## 训练以及loss

不变。



## 代码

- [https://github.com/lucidrains/ETSformer-pytorch](https://github.com/lucidrains/ETSformer-pytorch)



## 实验以及适用场景

本文主要讨论的是时间序列问题，实验部分忽略。



## 细节

暂无。



## 简评

本质上还是一种相对位置编码的思路，可以在$$O(nd\log n)$$时间内完成，可以进行尝试。