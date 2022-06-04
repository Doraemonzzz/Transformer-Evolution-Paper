# RealFormer: Transformer Likes Residual Attention

论文地址：

- [https://arxiv.org/abs/2012.11747](https://arxiv.org/abs/2012.11747)



## 整体思路以及计算方式

对残差部分进行了修改，将前一层的Attention Score传到下一层：

- $O=\mathrm{SoftMax}(QK^T + S) V$
- $S=QK^T + S$



## 时间复杂度

依然是$O(n^2d)$，但是系数上有差别，总体是增加了。



## 训练以及loss

不变。



## 代码

- [https://github.com/google-research/google-research/tree/master/realformer](https://github.com/google-research/google-research/tree/master/realformer)



## 实验以及适用场景

作者在Bert上测试了性能，比较了Post-LN, Pre-LN以及RealFormer(论文提出的方法)的性能，总体来说，RealFormer的性能更好。



## 细节

暂无。



## 简评

感觉速度上会慢一点，性能提升不算很明显。