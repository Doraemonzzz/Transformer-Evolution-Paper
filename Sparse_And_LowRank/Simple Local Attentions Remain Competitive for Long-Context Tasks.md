# Simple Local Attentions Remain Competitive for Long-Context Tasks

论文地址：

- [https://arxiv.org/abs/2112.07210](https://arxiv.org/abs/2112.07210)



## 整体思路以及计算方式

论文没有提新方法，比较了sparse attention和local attention的效果，这里的local attention分为两种：

![](../.Photo/Sparse_And_LowRank/1.jpg)

最后结论如下：

- LRA benchmark太简单，结果基本一致；
- 在pretrain + finetune设置下，简单的local attention比其他花里胡哨的方法都要好；
  - Blockwise效果比Local更好，并且不重合的效果最好；



## 时间复杂度

不考虑。



## 训练以及loss

不考虑。



## 代码

无。



## 实验以及适用场景

只比较了encoder(roberta)场景，在decoder(lm)和encoder-decoder(nmt)上没有进行测试。



## 细节

无。



## 简评

结论挺反直觉的，只能用大道至简来形容。