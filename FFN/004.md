# Simple Recurrence Improves Masked Language Models

论文地址：

- [https://arxiv.org/abs/2205.11588](https://arxiv.org/abs/2205.11588)



## 整体思路以及计算方式

将Transformer中的FFN模块换成RNN，最终带来了提升，计算公式如下：

- 输入$X\in \mathbb R^{n\times d}$
- 隐藏状态$X_1= XW_1\in \mathbb R^{n\times d_1},X_2= XW_2\in \mathbb R^{n\times d_1}$
- 计算$C\in \mathbb R^{n\times d_1}$
  - $c[0]=0$
  - $c[i]=\mathrm{Swish}\left(c[i-1]- x_1[i]\right)+x_1[i]$
- ${H}=\left(\left({C}+{b}_{c}\right) \odot \sigma\left({X}_{2}+{b}_{\sigma}\right)\right){W}_{3}+{b}_{3} \in \mathbb R^{n\times d}$

改进：

由于循环太慢，另一种计算方式是对$k$个位置同时计算，$k=1$退化到前一种情形：

- $c[0:k]=0$
- $c[ik:(i+1)k]=\mathrm{Swish}\left(c[(i-1)k:ik]- x_1[ik:(i+1)k]\right)+x_1[ik:(i+1)k]$



## 时间复杂度

总时间为$O(n dd_1)$，但是由于是RNN，实际上会慢很多，作者给出的数字是$k=1$时耗时为140%，$k=1$时耗时为120%。



## 训练以及loss

不变。



## 代码

暂无。



## 实验以及适用场景

适用于所有场景，作者测试了BERT(Encoder)和GLUE任务，带来了一定的提升，注意这里是时间换性能，所以是否值得需要视场景而定；Decoder的结果作者没有测试，后续可以尝试一下。



## 细节

暂无。



## 简评

思路很简单的一篇论文，但是可以带来如下思考：

- Transformer中FFN的作用到底是啥，之前一直理解为特征融合模块，但是利用RNN这样的序列融合模块也能达到同样作用；
- 既然FFN和RNN起的作用相当，而RNN可以用Attention模块代替，那是否可以将FFN换成Attention？