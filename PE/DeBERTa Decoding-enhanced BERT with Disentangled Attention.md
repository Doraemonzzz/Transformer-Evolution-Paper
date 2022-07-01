# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

论文地址：

- [https://arxiv.org/abs/2006.03654](https://arxiv.org/abs/2006.03654)



## 整体思路以及计算方式

传统的Attention计算，$$Q,K$$可以拆成context和pos部分：
$$
\begin{aligned}
Q_{c}&=H W_{q, c}\\
K_{c}&=H W_{k, c}\\
V_{c}&=H W_{v, c}\\
Q_{r}&=P W_{q, r}\\
K_{r}&=P W_{k, r}

\end{aligned}
$$
所以Attention Score的计算可以拆成4项：
$$
\begin{aligned}
\tilde{A}_{i, j}=Q_{i}^{c} K_{j}^{c \top}+Q_{i}^{c} K_{j}^{r\top}+K_{j}^{c} Q_{j}^{r \top}
+K_{i}^{r} Q_{i}^{r \top}
\end{aligned}
$$
DeBERTa的计算方式是将上式修改为：
$$
\tilde{A}_{i, j}=\underbrace{Q_{i}^{c} K_{j}^{c \top}}_{\text {(a) content-to-content }}+\underbrace{Q_{i}^{c} K_{\delta(i, j)}^{r{\top}}}_{\text {(b) content-to-position }}+\underbrace{K_{j}^{c} Q_{\delta(j, i)}^{r{\top}}}_{\text {(c) position-to-content }}
$$
其中：
$$
\delta(i, j)=\left\{\begin{array}{rcl}
0 & \text { for } & i-j \le-k \\
2 k-1 & \text { for } & i-j \ge k \\
i-j+k & \text { others. } &
\end{array}\right.
$$
即在一定范围内由相对位置确定，该范围外为固定值。



## 时间复杂度

Attention Matrix的时间复杂度由$$n^2d$$增加为$$3n^2d$$，其余部分不变。



## 训练以及loss

不变。



## 代码

- [https://github.com/microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)



## 实验以及适用场景

适用于所有场景，论文主要测试了在BERT中的效果。



## 细节

暂无。



## 简评

性能很好，但是无法适用于Linear Attention，所以暂时不考虑复现。