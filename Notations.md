# 数学符号

这里统一规定笔记中的数学记号。



## 基本符号

1. 向量用小写mathbf字体表示：$\mathbf  x \in \mathbb R^d$(所有向量均为列向量，即$$\mathbf x \in \mathbb R^{d\times 1}$$)；

2. 矩阵用大写mathbf字体表示，$$\mathbf X\in \mathbb R^{n\times d}$$：
   $$
   \begin{aligned}
   \mathbf X&= \left[
    \begin{matrix}
   \mathbf x_1^T  \\
   \vdots \\
   \mathbf x_n^T
     \end{matrix}
     \right]\in \mathbb R^{n\times d};
   \end{aligned}
   $$

3. 标量用常规字体表示$$\alpha, \beta$$；

4. 默认用$$n$$表示序列长度，$$d$$表示嵌入维度，$$b$$表示batch size；

5. Score Matrix：$$\mathbf S=\mathbf Q \mathbf K^{\top}$$；

6. Attention Matrix：$$\mathbf A = f(\mathbf S)$$；

   - 一般场景下$$f=\mathrm{Softmax}$$，但是也可以有别的选择；

7. 一些常用算子符号：

   - $$\mathrm{Softmax}(\mathbf X,d=-1): \mathbb R^{n\times d}\to \mathbb R^{n\times d}$$：
     - $$d$$为归一化维度，不指定时为最后一维，这里表示映射时没有考虑$d$，做个不严格的简化定义；
   - $$\mathrm{Norm}(\mathbf X,d=-1): \mathbb R^{n\times d}\to \mathbb R^{n\times d}$$：
     - 各种归一化方式，具体类型使用文字说明，符号中不体现，$$d$$为归一化维度，不指定时为最后一维;
   - $$\mathrm{MHA}(\mathbf X, \mathbf Y):\mathbb R^{n\times d}\times \mathbb R^{m\times d}\to \mathbb R^{n\times d}$$：
     - 经过一段时间的思考，最终还是将多头注意力机制定义为上述形式，具体来说$\mathbf X$对应query，$\mathbf Y$对应key, value；
   - $\mathrm {FFN}(\mathbf{X}): \mathbb R^{n\times  d} \to \mathbb R^{n\times d}$：
     - Transformer中FFN层；
   
8. $\mathrm{Sum}(\mathbf X,d=0): \mathbb R^{n\times d} \to \mathbb R^{d}$

目前先定义这些，后续再进行补充。

   