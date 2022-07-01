# ELECTRA Pre-training Text Encoders as Discriminators Rather Than Generators

论文地址：

- [https://arxiv.org/abs/2003.10555](https://arxiv.org/abs/2003.10555)

参考资料：

- https://zhuanlan.zhihu.com/p/89763176
- https://www.zhihu.com/question/354070608



## 整体思路以及计算方式

ELECTRA引入了一种新的预训练方式。

步骤如下：

- 首先由Generator和Discriminator，Generator通常为一个MLM，Discriminator为ELECTRA
- Generator：给定一段文本$$x_0\in \mathbb R^{n\times d}$$，首先mask掉一部分内容得到$$x_1\in \mathbb R^{n\times d}$$，喂给MLM得到输出$$x_2\in \mathbb R^{n\times m}$$，$$m$$为词表大小，根据$$x_2$$对被mask掉的部分进行采样，然后和未被mask掉的部分拼接得到$$x_3\in \mathbb R^{n\times d}$$，然后输出给Discriminator；
- Discriminator：给定$$x_3\in \mathbb R^{n\times d}$$，得到输出$$x_4\in \mathbb R^{n\times 2}$$，然后判断被mask掉的部分是否和原来相同；



## 时间复杂度

因为是预一种训练方式，所以不考虑这点。



## 训练以及loss

本身就是一种预训练方式，主要修改了loss：
$$
\sum_{\mathbf {x} \in \mathcal{X}} \mathcal{L}_{\mathrm{MLM}}\left(\mathbf {x}, \theta_{G}\right)+\lambda \mathcal{L}_{\text {Disc }}\left(\mathbf {x}, \theta_{D}\right)
$$


## 代码

- [https://github.com/lucidrains/electra-pytorch](https://github.com/lucidrains/electra-pytorch)
- [https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py](https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py)



## 实验以及适用场景

因为是预一种训练方式，所以不考虑这点。



## 细节

和GAN很像，但还是有所不同：

- loss不一样，这一点比较直观；
- Discriminator的梯度无法传给Generator；



## 简评

文章比较早，但是由于入行晚，最近才注意到，感觉后续可以复现一下。