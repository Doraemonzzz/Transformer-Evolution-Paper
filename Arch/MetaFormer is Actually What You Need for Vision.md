# MetaFormer is Actually What You Need for Vision

论文地址：

- [https://arxiv.org/abs/2111.11418](https://arxiv.org/abs/2111.11418)



## 整体思路以及计算方式

这篇文章的方法看似很简单，带背后带来的信息其实非常多：

- 只要一个模型有TokenMixer和FeatureMixer两部分，就能带来不错的效果；
- TokenMixer部分作者选择的是pooling；



## 时间复杂度

因为Tokenmixer使用pooling操作，所以总时间复杂度应该为$O(nd^2)$。



## 训练以及loss

不变。



## 代码

- [https://github.com/sail-sg/poolformer](https://github.com/sail-sg/poolformer)
- [https://github.com/lucidrains/metaformer-gpt](https://github.com/lucidrains/metaformer-gpt)



## 实验以及适用场景

目前的由于使用了pooling，所以只适用于Encoder，但是将其修改，应该可以适配到Deocder中。



## 细节

暂无。



## 简评

大道至简，这篇文章指出来一个本质问题，从这点来说比其提供的方法更重要。