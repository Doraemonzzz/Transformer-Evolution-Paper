# SimpleTRON: Simple Transformer with O(N) Complexity

论文地址：

- [https://arxiv.org/abs/2111.15588](https://arxiv.org/abs/2111.15588)



## 整体思路以及计算方式

思路非常简单，不带激活函数的Kernel method：
$$
\mathbf O=\frac{1}{\sqrt{n}} \mathbf Q \left(\mathbf K^{\top} \mathbf V \right)
$$


## 时间复杂度

$O(nd^2)$。



## 简评

思路比较简单，实验也比较简单，效果应该不会好。