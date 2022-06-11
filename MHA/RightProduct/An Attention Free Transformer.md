# An Attention Free Transformer

论文地址：

- [https://arxiv.org/abs/2105.14103](https://arxiv.org/abs/2105.14103)



## 整体思路以及计算方式

提出了一种代替Attention的模块，最一般的计算形式为：
$$
\begin{aligned}
f(Q, K, V)_{td}&=\sigma_{q}\left(Q_{td}\right) \odot \frac{\sum_{s\le t} w_{s,t}\left(\sigma_{k}\left(K_{sd}\right) \odot V_{sd}\right)}
{\sum_{s\le t} w_{s,t} \sigma_{k}\left(K_{sd}\right)} \\
f(Q, K, V)_{t}&=\sigma_{q}\left(Q_t\right) \odot \frac{\sum_{s\le t} w_{s,t}\left(\sigma_{k}\left(K_{s}\right) \odot V_{s}\right)}
{\sum_{s\le t} w_{s,t} \sigma_{k}\left(K_{s}\right)} 

\end{aligned}
$$
该方法和Linear Attention非常接近，不过使用了点乘（求和符号是关于每个特征维度）



## 时间复杂度

时间复杂度为$O(nd)$，不过由于需要存储$w_{s,t}$，所以会增加$O(n^2)$空间复杂度。



## 训练以及loss

不变。



## 代码

暂无。



## 实验以及适用场景

实验跑了Encoder和Decoder，效果尚可。



## 细节

由于点乘的存在，所以不适用于NMT。



## 简评

非常好的一个思路，之前没有仔细关注，最主要的创新点是增加了$w_{s,t}$，可以增加模型表达能力。