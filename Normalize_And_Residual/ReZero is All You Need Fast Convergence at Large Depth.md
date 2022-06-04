# ReZero is All You Need: Fast Convergence at Large Depth

论文地址：

- [https://arxiv.org/abs/2003.04887](https://arxiv.org/abs/2003.04887)



## 整体思路以及计算方式

作者提出了一种使得深度网络更容易训练的方式，比较新颖的是，该方法没有使用normalize。

该方法非常简单， 作为对比，给出常见一些常见的normalize方式：

- Deep Network: $\boldsymbol{x}_{i+1}=F\left(\boldsymbol{x}_{i}\right)$
- Residual Network: $\boldsymbol{x}_{i+1}=\boldsymbol{x}_{i}+F\left(\boldsymbol{x}_{i}\right)$
- Deep Network + Norm: $\boldsymbol{x}_{i+1}=\operatorname{Norm}\left(F\left(\boldsymbol{x}_{i}\right)\right)$
- Residual Network + Pre-Norm: $\boldsymbol{x}_{i+1}=\boldsymbol{x}_{i}+F\left(\operatorname{Norm}\left(\boldsymbol{x}_{i}\right)\right)$
- Residual Network + Post-Norm: $\boldsymbol{x}_{i+1}=\operatorname{Norm}\left(\boldsymbol{x}_{i}+F\left(\boldsymbol{x}_{i}\right)\right)$
- ReZero: $\boldsymbol{x}_{i+1}=\boldsymbol{x}_{i}+\alpha_{i} F\left(\boldsymbol{x}_{i}\right)$

注意$\alpha_i$需要初始化为0。



## 时间复杂度

不考虑。



## 训练以及loss

不变。



## 代码

- [https://github.com/majumderb/rezero](https://github.com/majumderb/rezero)



## 实验以及适用场景

适用于所有场景；从实验中可以看出确实提升了网络的收敛速度。



## 细节

主要就是初始化为0。



## 简评

第一印象会感觉该方法不会work，但是结果非常反直觉，由于该方法应该会提升不少速度，所以非常值得复现。