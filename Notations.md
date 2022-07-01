# 数学符号

这里统一规定笔记中的数学记号。



## 基本符号

1. 向量用小写mathbf字体表示：$$\mathbf  x \in \mathbb R^d$$(所有向量均为列向量，即$$\mathbf x \in \mathbb R^{d\times 1}$$)；

2. 矩阵用大写mathbf字体表示，$$\mathbf X\in \mathbb R^{n\times d}$$：
   $$
   \begin{aligned}
   \mathbf X&= \left[
    \begin{matrix}
   \mathbf x_1^{\top}   \\
   \vdots \\
   \mathbf x_n^{\top} 
     \end{matrix}
     \right]\in \mathbb R^{n\times d};
   \end{aligned}
   $$

3. $$\mathbf x_i$$表示矩阵$$\mathbf X$$的第$$i$$行的转置；

4. 标量用常规字体表示$$\alpha, \beta$$；

5. 默认用$$n$$表示序列长度，$$d$$表示嵌入维度，$$b$$表示batch size；

6. Score Matrix：$$\mathbf S=\mathbf Q \mathbf K^{\top}$$；

7. Attention Matrix：$$\mathbf A = f(\mathbf S)$$；

   - 一般场景下$$f=\mathrm{Softmax}$$，但是也可以有别的选择；

8. 一些常用算子符号：

   - $$\mathrm{Softmax}(\mathbf X,d=-1): \mathbb R^{n\times d}\to \mathbb R^{n\times d}$$：
     - $$d$$为归一化维度，不指定时为最后一维，这里表示映射时没有考虑$$d$$，做个不严格的简化定义；
   - $$\mathrm{Norm}(\mathbf X,d=-1): \mathbb R^{n\times d}\to \mathbb R^{n\times d}$$：
     - 各种归一化方式，具体类型使用文字说明，符号中不体现，$$d$$为归一化维度，不指定时为最后一维;
   - $$\mathrm{MHA}(\mathbf X, \mathbf Y):\mathbb R^{n\times d}\times \mathbb R^{m\times d}\to \mathbb R^{n\times d}$$：
     - 一种$$\mathrm {MHA}$$的接口，最具体来说$$\mathbf X$$对应query，$$\mathbf Y$$对应key, value；
   - $$\mathrm{MHA}(\mathbf Q, \mathbf K,\mathbf V):\mathbb R^{n\times d}\times \mathbb R^{m\times d}\times \mathbb R^{m\times d}\to \mathbb R^{n\times d}$$：
     - 另一种$$\mathrm{MHA}$$的接口，不常使用；
   - $$\mathrm {FFN}(\mathbf{X}): \mathbb R^{n\times  d} \to \mathbb R^{n\times d}$$：
     - Transformer中FFN层；

9. $$\mathrm{Sum}(\mathbf X,d=0): \mathbb R^{n\times d} \to \mathbb R^{d}$$

目前先定义这些，后续再进行补充。

   