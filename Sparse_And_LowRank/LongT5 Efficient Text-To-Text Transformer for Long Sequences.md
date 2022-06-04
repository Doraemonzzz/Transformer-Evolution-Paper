# LongT5: Efficient Text-To-Text Transformer for Long Sequences

论文地址：

- [https://arxiv.org/abs/2112.07916](https://arxiv.org/abs/2112.07916)



## 整体思路以及计算方式

利用Local Attention + Global Attention减少运算量：

![](../.Photo/Sparse_And_LowRank/3.jpg)

Local Attention：

每个token只和附近$2r+1$个token交互。

Global Attention：

将$n$个token按每组$l$个划分，组内token取均值，这样一共得到$n/l$个token，每个token和这$n/l$个token交互。



## 时间复杂度

时间复杂度为$O(n(2r+1)d + n^2 /l d)$。



## 训练以及loss

不变。



## 代码

- [https://github.com/google-research/longt5](https://github.com/google-research/longt5)



## 实验以及适用场景

论文测试了encoder-decoder结构。



## 细节

因为global token是通过求均值得到的，所以单向模型时实现起来较为复杂。



## 简评

很简单并且优雅的方式，可以考虑复现。