# Kronecker Attention Networks

论文地址：

- https://arxiv.org/pdf/2007.08442.pdf



## 整体思路以及计算方式

利用Kronecker内积的方式计算Attention，但实际上这里使用只是外积。

计算方式：

- 给定$q, k, v\in \mathbb R^{n\times d}$
- 计算$q_{mean}, k_{mean}\in \mathbb R^{n\times 1}$
- 计算$\mathrm{MHA}(q_{mean}, k_{mean},v) \in \mathbb R^{n\times d}$



## 时间复杂度

尽管使用了降维，但是计算复杂度仍然为$O(n^2d)$。



## 训练以及loss

不变。



## 代码

[https://github.com/lucidrains/kronecker-attention-pytorch](https://github.com/lucidrains/kronecker-attention-pytorch)



## 实验以及适用场景

原始方法只适用于Encoder，但是将mean修改为前$i$项的均值可以适用于Decoder；论文里测试了CV的结果，效果比较一般。



## 细节

暂无。



## 简评

优点：

- 实现比较简单，算一种降维方法；

总结：

- 虽然效果一般，但可以尝试复现；

