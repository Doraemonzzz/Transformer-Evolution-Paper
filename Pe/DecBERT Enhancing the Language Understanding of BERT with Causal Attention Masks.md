# DecBERT: Enhancing the Language Understanding of BERT with Causal Attention Masks

论文地址：

- [https://arxiv.org/abs/2204.08688](https://arxiv.org/abs/2204.08688)



## 整体思路以及计算方式

在MLM预训练任务中，token的顺序是通过position embedding提供的；而在Causal LM中，token的顺序则可以通过计算方式（Causal Mask）捕捉到，事实上，在Causal LM中，是否添加position embedding对结果影响并不大（见论文）；基于这点，作者提出了能否在MLM任务中使用Causal Mask，最终得到了相当的提升。

计算方式：

- 在Bert前两层中加入上/下三角mask；



## 时间复杂度

不变。



## 训练以及loss

不变。



## 代码

暂无，但是实现起来很简单。



## 实验以及适用场景

适用于Encoder中，作者做了不少实验，这里总结如下：

- Causal Lm中是否使用位置编码，结果是区别非常小；
- BERT(MLM)中是否使用位置编码，结果是区别非常大；
- DecBERT(作者的方法)中是否使用位置编码，结果是有一定的区别，不使用的效果低于BERT，使用后效果好于BERT；



## 细节

暂无。



## 简评

挺有意思的结果，可以考虑复现。