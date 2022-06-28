# LaMemo: Language Modeling with Look-Ahead Memory

论文地址：

- [https://arxiv.org/abs/2204.07341](https://arxiv.org/abs/2204.07341)



## 整体思路以及计算方式

之前Transformer中使用memory的方式都是当前token和memory中token交互，但是memory中token无法和当前token交互，本文就是对这点进行改进。

符号：

- 当前token：$\boldsymbol{X}_{\tau}=\left[\boldsymbol{x}_{\tau+1}, \cdots, \boldsymbol{x}_{\tau+N}\right] \in \mathbb{R}^{N \times d}$
- memory：$\boldsymbol{X}_{\tau-1}=\left[\boldsymbol{x}_{\tau-M+1}, \cdots, \boldsymbol{x}_{\tau}\right] \in \mathbb{R}^{M \times d}$
- $\tilde{\boldsymbol{X}}_{\tau-1}=\left[\boldsymbol{x}_{\tau-N+2}, \cdots, \boldsymbol{x}_{\tau+1}\right] \in \mathbb{R}^{N \times d}$

计算：

- $\boldsymbol{Q}_{\tau}=\boldsymbol{X}_{\tau} \boldsymbol{W}_{q}, \boldsymbol{K}_{\tau}=\boldsymbol{X}_{\tau} \boldsymbol{W}_{k}, \boldsymbol{V}_{\tau}=\boldsymbol{X}_{\tau} \boldsymbol{W}_{v}$
- $\tilde{\boldsymbol{K}}_{\tau-1}=\tilde{\boldsymbol{X}}_{\tau-1} \boldsymbol{W}_{k}, \tilde{\boldsymbol{V}}_{\tau-1}=\tilde{\boldsymbol{X}}_{\tau-1} \boldsymbol{W}_{v}$
- $\boldsymbol{C}_{\tau}^{\leftarrow}=\operatorname{softmax}_{\text{upper-triangle}} \left(\frac{\boldsymbol{Q}_{\tau} \tilde{\boldsymbol{K}}_{\tau}^{\top}}{\sqrt{d}}\right) \tilde{\boldsymbol{V}}_{\tau}$
- $\boldsymbol{C}_{\tau}^{\rightarrow}=\operatorname{softmax}_{\text{upper-triangle}} \left(\frac{\boldsymbol{Q}_{\tau} \tilde{\boldsymbol{K}}_{\tau}^{\top}}{\sqrt{d}}\right) \tilde{\boldsymbol{V}}_{\tau}$
- $\boldsymbol{C}_{\tau-1}^{\leftrightarrow}=\boldsymbol{\alpha}_{\tau} \operatorname{sg}\left(\boldsymbol{C}_{\tau}^{\rightarrow}\right)+\left(1-\boldsymbol{\alpha}_{\tau}\right) \boldsymbol{C}_{\tau}^{\leftarrow}$
  - $\mathrm{sg}$表示不计算梯度；
  - $\boldsymbol{\alpha}_{\tau}=\frac{\operatorname{sg}\left(\boldsymbol{s}_{\tau}^{\rightarrow}\right)}{\operatorname{sg}\left(\boldsymbol{s}_{\tau}^{\rightarrow}\right)+\boldsymbol{s}_{\tau}^{{\leftarrow}}+\varepsilon}$；
  - 其中$\boldsymbol{s}_{\tau}^{\rightarrow}$表示$\boldsymbol{C}_{\tau}^{\rightarrow}$中Softmax矩阵归一化之前的元素和；



## 时间复杂度

时间复杂度为$O(N(N+M)d)$。



## 训练以及loss

不变。



## 代码

- [https://github.com/thu-coai/LaMemo](https://github.com/thu-coai/LaMemo)



## 实验以及适用场景

适用于encoder和decoder；论文只测试了lm(decoder)场景，获得了一定的提升。



## 细节

暂无。



## 简评

提供了一种新的memory交互方式。