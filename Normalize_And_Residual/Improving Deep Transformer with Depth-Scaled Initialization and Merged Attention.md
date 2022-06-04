# Improving Deep Transformer with Depth-Scaled Initialization and Merged Attention

论文地址：

- [https://arxiv.org/abs/1908.11365](https://arxiv.org/abs/1908.11365)



## 整体思路以及计算方式

论文工作量两点：

1. 解释了深层Transformer为什么难以训练，针对这点，提出了一种新的初始化方式；
2. 结合作者之前的AAN，提出加速解码的方案；

第一点：

作者的观点是，深层Transformer难以训练是因为反传的原因，而主要的方面是因为残差和Layer Norm。

回顾Transformer的计算公式：
$$
\begin{aligned}
\mathrm{r}&=\operatorname{RC}(\mathrm{z}, f(\mathrm{z})) \\
\mathrm{o}&=\operatorname{LN}(\mathrm{r})
\end{aligned}
$$
其中$f$表示$\mathrm{MHA}$或$\mathrm{FFN}$，$\mathrm{RC}$表示残差连接，$\mathrm{LN}$表示layer norm。

计算反传可得：
$$
\begin{aligned}
&\delta_{r}=\frac{\partial \mathrm{o}}{\partial \mathrm{r}} \delta_{o}=\operatorname{diag}\left(\frac{\mathrm{g}}{\sigma_{r}}\right)\left(\mathrm{I}-\frac{1-\overline{\mathrm{r}} \overline{\mathrm{r}}^{T}}{d}\right) \delta_{o} \\
&\delta_{z}=\frac{\partial \mathrm{r}}{\partial \mathrm{z}} \delta_{r}=\left(1+\frac{\partial f}{\partial \mathrm{z}}\right) \delta_{r},
\end{aligned}
$$
其中$\overline {\mathrm r}$表示归一化的输入，所以$\overline{\mathrm{r}} \overline{\mathrm{r}}^{T}$表示方差，注意到$\delta_r$和方差正相关。

接着作者考虑了如下量：
$$
\beta=\beta_{\mathrm{LN}} \cdot \beta_{\mathrm{RC}}=\frac{\left\|\delta_{z}\right\|_{2}}{\left\|\delta_{r}\right\|_{2}} \cdot \frac{\left\|\delta_{r}\right\|_{2}}{\left\|\delta_{o}\right\|_{2}}
$$

从直觉上看，我们希望$\beta\approx 1$，因为太大会造成梯度消失，太小会造成梯度爆炸。那么$\beta,\beta_{\mathrm{LN}} , \beta_{\mathrm{RC}}$的值大概是多少？作者经过分析，给出如下结论：

- $\mathrm{FFN}$的$\beta<1$，$\mathrm{MHA}$的$\beta>1$，包含Self和Cross Attention；
- 无论是$\mathrm{FFN}$还是$\mathrm{MHA}$，都有$\beta_{\mathrm{LN}}<1$ , $\beta_{\mathrm{RC}}>1$；
  - 在Transformer中，$\beta_{\mathrm{LN}}\in [0.82, 0.86], \beta_{\mathrm{RC}}\in [1.10,1.22]$

由于$\beta_{\mathrm{RC}}$和方差正相关，所以减少方差可以使得$\beta_{\mathrm{RC}}$更接近1，作者给出的方案是修改初始化方案：
$$
\mathrm{W} \in \mathbb{R}^{d_{i} \times d_{o}} \sim \mathcal{U}\left(-\gamma \frac{\alpha}{\sqrt{l}}, \gamma \frac{\alpha}{\sqrt{l}}\right)
$$
其中$l$表示层数。

第二点：

把NMT中Decoder的Self Attention换成AAN，从而提高解码效率：
$$
\begin{aligned}
&\operatorname{MATT}\left(\mathrm{S}^{l-1}\right)=\operatorname{SAAN}\left(\mathrm{S}^{l-1}\right)+\operatorname{ATT}\left(\mathrm{S}^{l-1}, \mathrm{H}^{L}\right) \\
&\overline{\mathrm{S}}^{l}=\mathrm{L N}\left(\operatorname{RC}\left(\mathrm{S}^{l-1}, \operatorname{MATT}\left(\mathrm{S}^{l-1}\right)\right)\right)
\end{aligned}
$$



## 训练以及loss

没有变化。



## 代码

- [https://github.com/bzhangGo/zero/tree/master/docs/depth_scale_init_and_merged_attention](https://github.com/bzhangGo/zero/tree/master/docs/depth_scale_init_and_merged_attention)



## 实验以及适用场景

初始化部分适用于所有场景，解码部分适用于Decodeer；实验测试了NMT，带来了一定的效果。



## 细节

暂无。



## 简评

第一点提供了一种分析的思路，值得复现。