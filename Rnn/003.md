# Going Beyond Linear Transformers with Recurrent Fast Weight Programmers

论文地址：

- [https://arxiv.org/abs/2106.06295](https://arxiv.org/abs/2106.06295)



## 整体思路以及计算方式

首先回顾Linear Attention的计算方式：
$$
\begin{aligned}
\mathbf{k}_{t}, \mathbf{v}_{t}, \mathbf{q}_{t} &=\mathbf{W}_{k} \mathbf{x}_{t}, \mathbf{W}_{v} \mathbf{x}_{t}, \mathbf{W}_{q} \mathbf{x}_{t} \\
\mathbf{W}_{t} &=\mathbf{W}_{t-1}+\mathbf{v}_{t} \otimes \mathbf{k}_{t} \\
\mathbf{y}_{t} &=\mathbf{W}_{t} \mathbf{q}_{t}
\end{aligned}
$$

其中$$ \otimes $$表示向量外积。

作者将公式二改写为：
$$
\mathbf{W}_{t}=\mathbf{W}_{t-1}+\beta_{t}\left(\mathbf{v}_{t}-\overline{\mathbf{v}}_{t}\right) \otimes \mathbf{k}_{t}
$$
将公式一改写为：
$$
\begin{aligned}
\mathbf{k}_{t} &=\mathbf{W}_{k} \mathbf{x}_{t}+\mathbf{R}_{k} \tanh \left(\mathbf{y}_{t-1}\right) \\
\mathbf{v}_{t} &=\mathbf{W}_{v} \mathbf{x}_{t}+\mathbf{R}_{v} \tanh \left(\mathbf{y}_{t-1}\right) \\
\mathbf{q}_{t} &=\mathbf{W}_{q} \mathbf{x}_{t}+\mathbf{R}_{q} \tanh \left(\mathbf{y}_{t-1}\right) \\
\beta_{t} &=\sigma\left(\mathbf{W}_{\beta} \mathbf{x}_{t}+\mathbf{R}_{\beta} \tanh \left(\mathbf{y}_{t-1}\right)\right)
\end{aligned}
$$



## 时间复杂度

$$O(nd^2)$$，但是因为使用了循环，所以实际会慢很多。



## 训练以及loss

不变。



## 代码

- [https://github.com/IDSIA/recurrent-fwp](https://github.com/IDSIA/recurrent-fwp)



## 实验以及适用场景

测试了各种场景，总体性能不错。



## 细节

暂无。



## 简评

把Attention修改为RNN，个人感觉是一种退步，不看好这个工作。