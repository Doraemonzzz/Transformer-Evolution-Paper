# Rethinking Positional Encoding in Language Pre-training

论文地址：

- [https://arxiv.org/abs/2006.15595](https://arxiv.org/abs/2006.15595)



## 整体思路以及计算方式

给出引入相对位置编码的一种方案，主要是将位置向量和词向量分开计算。

改进1，修改相似度计算函数：
$$
\begin{aligned}
\alpha_{i j}&=\frac{1}{\sqrt{2 d}}\left(x_{i}^{l} W^{Q, l}\right)\left(x_{j}^{l} W^{K, l}\right)^{\top}+\frac{1}{\sqrt{2 d}}\left(p_{i} U^{Q}\right)\left(p_{j} U^{K}\right)^{\top}\\
\alpha_{i j}&=\frac{1}{\sqrt{2 d}}\left(x_{i}^{l} W^{Q, l}\right)\left(x_{j}^{l} W^{K, l}\right)^{\top}+\frac{1}{\sqrt{2 d}}\left(p_{i} U^{Q}\right)\left(p_{j} U^{K}\right)^{\top}+b_{j-i}

\end{aligned}
$$
改进2，CLS特殊处理：
$$
\operatorname{reset}_{\theta}(v, i, j)= \begin{cases}v_{i j} & i \neq 1, j \neq 1,(\text { not related to }[\mathrm{CLS}]) \\ \theta_{1} & i=1,(\text { from }[\mathrm{CLS}] \text { to others }) \\ \theta_{2} & i \neq 1, j=1,(\text { from others to }[\mathrm{CLS}])\end{cases}
$$
整体计算公式为：
$$
\begin{aligned}
&\alpha_{i j}^{\mathrm{TUPE}-\mathrm{A}}=\quad \frac{1}{\sqrt{2 d}}\left(x_{i}^{l} W^{Q, l}\right)\left(x_{j}^{l} W^{K, l}\right)^{\top}+\operatorname{reset}_{\theta}\left(\frac{1}{\sqrt{2 d}}\left(p_{i} U^{Q}\right)\left(p_{j} U^{K}\right)^{\top}, i, j\right)\\
&\alpha_{i j}^{\mathrm{TUPE}-\mathrm{R}}=\frac{1}{\sqrt{2 d}}\left(x_{i}^{l} W^{Q, l}\right)\left(x_{j}^{l} W^{K, l}\right)^{\top}+\operatorname{reset}_{\theta}\left(\frac{1}{\sqrt{2 d}}\left(p_{i} U^{Q}\right)\left(p_{j} U^{K}\right)^{\top}+b_{j-i}, i, j\right)
\end{aligned}
$$



## 时间复杂度

不变，关于序列长度还是二次。



## 训练以及loss

不变。



## 代码

- [https://github.com/guolinke/TUPE](https://github.com/guolinke/TUPE)



## 实验以及适用场景

适用于Encoder，Decoder；论文测试了部分GLUE任务，提升比较明显。



## 细节

如果不单独考虑CLS，那么其实并没有提升。



## 简评

CLS单独考虑提供了一个新思路。