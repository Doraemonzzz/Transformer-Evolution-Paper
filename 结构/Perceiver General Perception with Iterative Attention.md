# Perceiver General Perception with Iterative Attention

论文地址：

- https://arxiv.org/abs/2103.03206

参考资料：

- https://zhuanlan.zhihu.com/p/360773327
- https://blog.csdn.net/weixin_39707121/article/details/117258115



## 整体思路以及计算方式

整体思路是利用CrossAttention来降维。

具体计算方式如下：

- 给定输入$x\in \mathbb R^{n\times d}, y\in \mathbb R^{m\times d}$
  - $x$对应Latent array，$y$对应Byte array，这里假设$n\ll m$；
  - 一个例子是$x$为图像的patch表示，$y$为像素级表示；
- $x= \mathrm{MHA}_1(x, y, y)\in \mathbb R^{n\times d}$
- $x= \mathrm{MHA}_2(x,x,x)\in \mathbb R^{n\times d}$

备注：这里省略了FFN以及NORM操作。



## 时间复杂度

$\mathrm{MHA}_1$的时间复杂度为$O(mnd)$，$\mathrm{MHA}_2$的时间复杂度为$O(n^2d)$，总时间复杂度为$O(mnd+n^2d)$，论文里假设$n\ll m$，所以总复杂度为$O(mnd)$。



## 训练以及loss

不变。



## 代码

[https://github.com/lucidrains/perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch)



## 实验以及适用场景

感觉还是主要适用于Encoder场景，像LM，NMT这样的任务似乎没法直接应用；论文做了除NLP以外的实验，效果还行。



## 细节

暂无，需要复现的时候体会。



## 简评

优点：

- 把CrossAttention理解为降维是一个很好的点；

总结：

- 值得复现，可以尝试应用于Roberta模型中；
- LM, NMT场景是否能使用需要思考；