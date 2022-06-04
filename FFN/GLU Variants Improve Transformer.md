# GLU Variants Improve Transformer

论文地址：

- [https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)



## 整体思路以及计算方式

FFN的替代品，首先是GLU模块：
$$
\mathrm{GLU}(x, W, V, b, c)=f(x W+b) \otimes(x V+c)
$$
其中$f$是任意激活函数。

定义新的FFN模块：
$$
\mathrm{FFN}_{\mathrm{GLU}}\left(x, W, V, W_{2}\right)=(f(x W) \otimes x V) W_{2}
$$


## 时间复杂度

假设：

- $x\in \mathbb R^{n\times d_1}$
- $W,V\in \mathbb R^{d_1\times d_2},W_2\in \mathbb R^{d_2\times d_1}$

所以时间复杂度为：
$$
O(nd_1d_2)
$$
这里引入的参数数量为$3d_1d_2$，传统FFN的参数数量为$8d_1^2$，要对标参数数量，取
$$
d_2 =\frac{8}{3}d_1 =\frac{2}{3} \times 4d_1
$$


## 训练以及loss

不变。



## 代码

无，很简单，直接实现即可。



## 实验以及适用场景

由于是FFN的替代，所以适用于所有场景；作者测试了GLUE任务，效果相当不错。



## 细节

暂无。



## 简评

总结：

- FFN的简单改进，效果不错，速度如何需要进行测试；
- 值得复现；