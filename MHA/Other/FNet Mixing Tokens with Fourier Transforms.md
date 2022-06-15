# FNet: Mixing Tokens with Fourier Transforms

论文地址：

- [https://arxiv.org/abs/2105.03824](https://arxiv.org/abs/2105.03824)



## 整体思路以及计算方式

思路非常简单，利用FFT进行token mix和feature mix，整体计算公式如下：

- 输入：$X\in \mathbb R^{n\times d}$
- $O=\mathrm{FFT}(\mathrm {FFT}(X, d=-1),d=-2).real\in \mathbb R^{n\times d}$



## 时间复杂度

$O(n^2 d + n d\log d + dn\log n )$。



## 训练以及loss

不变。



## 代码

- [https://github.com/erksch/fnet-pytorch](https://github.com/erksch/fnet-pytorch)
- [https://github.com/rishikksh20/FNet-pytorch](https://github.com/rishikksh20/FNet-pytorch)



## 实验以及适用场景

实验比较有意思，列出几个比较有意思的实验：

- BERT：83.3
- FNet（在seq和feature两个维度进行FFT）：76.7
- Linear（在feature和seq两个维度过全连接，可学）：77.0
- Random（在feature和seq两个维度过全连接，不可学）：56.6
- FFN-only（只有FFN）：49.3

实验结果说明，只有进行seq和feature两个维度的mix，就能得到相当合理的结果。

备注：该方法只能在Encoder中使用。



## 细节

暂无。



## 简评

一个有意思的思路，不知道能否应用在单向模型中。