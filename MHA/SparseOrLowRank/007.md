# Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting

论文地址：

- [https://arxiv.org/abs/1907.00235](https://arxiv.org/abs/1907.00235)

参考资料：

- [https://zhuanlan.zhihu.com/p/391337035](https://zhuanlan.zhihu.com/p/391337035)



## 整体思路以及计算方式

思路很简单，分为两点：

- Token mix：每个token和周围几个token进行融合得到$$Q,K,V$$，这样做可以明显加速收敛；
- Local attention：每个token只和局部token做attention，每层关注的局部位置不同，最后达到全局attention的效果；



## 时间复杂度

$$O(n\log_2^2n)$$



## 训练以及loss

不变。



## 代码

- [https://github.com/mlpotter/Transformer_Time_Series/blob/master/Transformer_Decoder_nologsparse.ipynb](https://github.com/mlpotter/Transformer_Time_Series/blob/master/Transformer_Decoder_nologsparse.ipynb)
- [https://github.com/mlpotter/Transformer_Time_Series/blob/master/causal_convolution_layer.py](https://github.com/mlpotter/Transformer_Time_Series/blob/master/causal_convolution_layer.py)



## 实验以及适用场景

作者测试了时间序列任务，该方法可以推广到其他任务。



## 细节

暂无。



## 简评

Token mix的思路非常好，相当于强行让每个token关注周围的信息；Local attention实现较为复杂，只要理解思路即可；