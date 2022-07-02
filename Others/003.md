## Is Attention Better Than Matrix Decomposition?

论文地址：

- https://arxiv.org/abs/2109.04553

参考资料：

- https://zhuanlan.zhihu.com/p/369769485
- https://zhuanlan.zhihu.com/p/369855045
- https://zhuanlan.zhihu.com/p/370410446



## 整体思路以及计算方式

思路是利用矩阵分解代替Attention。

假设$$X\in \mathbb R^{d\times n}$$可以分解为如下形式：
$$
{X}=\overline{{X}}+{E}={D} C+{E}
$$
这些变量满足如下条件（$$E$$表示噪声）：
$$
\begin{aligned}
\overline{{X}}& \in \mathbb{R}^{d \times n} \\
{E}& \in \mathbb{R}^{d \times n} \\
D&\in \mathbb R^{d\times r}\\
C&\in \mathbb R^{r\times n}\\

\end{aligned}
$$
使用流程为通过某种方式计算$$D,C$$，最后输出$$DC$$。

作者给出了两种方式计算$$D,C$$：

- Soft VQ
  - for $$k=1,\ldots, K$$:
    - $${C} \leftarrow \operatorname{Softmax}\left(\frac{1}{T} \operatorname{cosine}({D},{X})\right)$$
    - $${D} \leftarrow {X} \mathbf {C}^{\top} \operatorname{diag}\left({C} \mathbf{1}_{n}\right)^{-1}$$
  - return $$\bar X =DC$$
- NMF with MU
  - for $$k=1,\ldots, K$$:
    - $${C}_{i j} \leftarrow {C}_{i j} \frac{\left({D}^{\top} X\right)_{i j}}{\left({D}^{\top} {D C}\right)_{i j}}$$
    - $${D}_{i j} \leftarrow {D}_{i j} \frac{\left({X} C^{\top}\right)_{i j}}{\left({D C} {C}^{\top}\right)_{i j}}$$
  - return $$\bar X =DC$$



## 时间复杂度

Soft VQ时间复杂度：

- $${C} \leftarrow \operatorname{Softmax}\left(\frac{1}{T} \operatorname{cosine}({D},{X})\right)$$，所以时间复杂度为$$O(nrd)$$
  - $$\operatorname{cosine}({D},{X})$$需要计算$$D^{\top}  X$$，即$$r\times d,d\times n\to  r\times n$$，所以时间复杂度为$$O(nrd)$$
  - $$ \operatorname{Softmax}$$：$$r\times n \to r\times n$$，所以时间复杂度为所以时间复杂度为$$O(nr)$$
  - 总时间复杂度为$$O(nrd)$$
- $${D} \leftarrow {X} {C}^{\top} \operatorname{diag}\left({C} \mathbf{1}_{n}\right)^{-1}:$$
  - $$\operatorname{diag}\left({C} \mathbf{1}_{n}\right)^{-1}:r\times n,n\times 1 \to r\times 1 \to r\times r $$，时间复杂度为$$O(nr)$$
  - $${X} {C}^{\top}:d\times n, n\times r \to d\times r$$，时间复杂度为$$O(nrd) $$ 
  - $${C}^{\top} \operatorname{diag}\left({C} \mathbf{1}_{n}\right)^{-1}: d\times r, r\times r \to r\times r$$，时间复杂度为$$O(dr^2)$$
- 所以时间复杂度为$$O(nrd+ dr^2)$$
- 循环$$K$$次，时间复杂度为$$O(K(nrd+ dr^2))$$
- $$\bar X =DC: d\times r , r\times n \to d\times n$$，时间复杂度为$$O(nrd)$$
- 总时间复杂度为$$O((K+1)nrd + Kdr^2)$$

NMF with MU时间复杂度：

- $${C}_{i j} \leftarrow {C}_{i j} \frac{\left({D}^{\top} X\right)_{i j}}{\left({D}^{\top} {D C}\right)_{i j}}$$
  - $${D}^{\top} X: r\times d, d\times n\to r\times n$$，时间复杂度为$$O(nrd)$$
  - $${D}^{\top} {D C} $$
    - 先计算$$DC$$，再计算$${D}^{\top} {D C} $$
      - $$DC:d\times r, r\times n \to d\times n$$，时间复杂度为$$O(nrd)$$
      - $${D}^{\top} {D C}: r\times d, d\times n \to r\times n$$，时间复杂度为$$O(nrd)$$
    - 先计算$${D}^{\top} D$$，再计算$${D}^{\top} {D C} $$
      - $${D}^{\top} D: r\times d, d\times r \to r\times r $$，时间复杂度为$$O(dr^2)$$
      - 再计算$${D}^{\top} {D C}:r\times r, r\times n\to r\times n $$，时间复杂度为$$O(nrd)$$
    - 一般$$r<n$$，所以选择第二种算法，时间复杂度为$$O(nrd+dr^2)$$
  - 两次element wise乘法/除法：$$r\times n \to r\times n \to r\times n$$，时间复杂度为$$O(nr)$$
  - 循环$$K$$次，时间复杂度为$$O(K(nrd+dr^2))$$
  - $$\bar X =DC: d\times r , r\times n \to d\times n$$，时间复杂度为$$O(nrd)$$
  - 总时间复杂度为$$O((K+1)nrd + Kdr^2)$$

注意$$r, K$$一般不会很大（远小于$$n$$），所以该方法的时间复杂度关于序列长度大概能到线性。



## 训练以及loss

没有区别。



## 代码

- [https://github.com/lucidrains/hamburger-pytorch](https://github.com/lucidrains/hamburger-pytorch)
- [https://github.com/Gsunshine/Enjoy-Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger)



## 实验以及适用场景

目前只适用于Encoder结构，不适用于Decoder结构；实验主要是基于CV的，能达到和Attention相当的结果。



## 细节

在进行循环的时候不计算梯度，只有最后一次操作计算梯度。



## 简评

优点：

- 提供了一种新的理解Attention的视角；
- 方法实现比较简洁；

缺点：

- 没法直接应用到Decoder结构中，即无法训练lm；

总结

- 值得复现；