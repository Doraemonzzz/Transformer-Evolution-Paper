# Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks

论文地址：

- [https://arxiv.org/abs/2002.10444](https://arxiv.org/abs/2002.10444)



## 整体思路以及计算方式

思路和ReZero类似，用加权残差取代Normalize，唯一的区别是这篇讨论的是Batch Normalize，而ReZero讨论的是Layer Normalize：
$$
 \boldsymbol{x}_{i+1}=\boldsymbol{x}_{i}+\alpha_{i} F\left(\boldsymbol{x}_{i}\right)
$$
后续部分从略。



## 时间复杂度





## 训练以及loss



## 代码



## 实验以及适用场景





## 细节



## 简评

