# UFO-ViT: High Performance Linear Vision Transformer without Softmax

论文地址：

- [https://arxiv.org/abs/2109.14382](https://arxiv.org/abs/2109.14382)



## 整体思路以及计算方式

依然是利用了linear attention的方式，只不过这里$$Q,K$$没有过特征变换$$\phi$$，计算方式如下：

- 定义：$$\operatorname{XN}(x):=\frac{\gamma x}{\sqrt{\sum_{i=0}^{h}\|x\|^{2}}}$$
- $$Q,K, V = XW_Q, XW_K, XW_V \in \mathbb R^{n\times d}$$
- $$Y_1 = \mathrm{XN}_{\mathrm{axis=1}}(Q)$$
- $$Y_2=\mathrm{XN}_{\mathrm{axis=0}}(K^{\top}  V)$$
- $$O=Y_1 Y_2$$



## 时间复杂度

线性时间复杂度，依然是$$O(nd^2)$$。



## 训练以及loss

不变。



## 代码

非官方实现：

- [https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/UFOAttention.py](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/UFOAttention.py)



## 实验以及适用场景

适用于Encoder，Decoder，效果还行，不过感觉可能是由于在Attention和FFN之间加了卷积层的原因。



## 细节

暂无。



## 简评

个人感觉如果不加上卷积层，效果不会太好。