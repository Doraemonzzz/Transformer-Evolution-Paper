# Linear Complexity Randomized Self-attention Mechanism

论文地址：

- [https://arxiv.org/abs/2204.04667](https://arxiv.org/abs/2204.04667)



## 整体思路以及计算方式

之前像RFA和Performer(后续统称为RFA)都是$$\exp(q^{\top} v)$$的无偏估计，但并不是$$\exp(q^{\top} v)/(\sum_v \exp(q^{\top} v))$$的无偏估计，这偏论文的主要出发点就是解决这点，论文的整体思路如下：

- 介绍RFA和重要度抽样；
- 指出RFA不是无偏估计，通过重要度抽样引入RA(Randomized Attention);
- 指出RA的计算复杂度太高，作为一个折中方案，引入LARA(Linear Randomized Attention);



### RFA和重要度抽样

RFA：

如果有：
$$

\exp \left({x}^{\top} {y}\right)=\mathbb{E}_{\omega \sim \mathcal{N}(\omega ; 0, {I})}\left[\xi({x}, \omega)^{\top} \xi({y}, \omega)\right] \tag 1
$$
那么：
$$
\begin{aligned}
    &\frac{\sum_{m=1}^{M} \exp \left({q}_{n}^{\top} {k}_{m}\right) {v}_{m}^{\top}}{\sum_{m^{\prime}=1}^{M} \exp \left({q}_{n}^{\top} {k}_{m^{\prime}}\right)} \\
    &\approx \frac{\sum_{m=1}^{M} \sum_{s=1}^{S} \xi\left({q}_{n}, \omega_{s}\right)^{\top} \xi\left({k}_{m}, \omega_{s}\right) {v}_{m}^{\top}}{\sum_{m^{\prime}=1}^{M} \sum_{s=1}^{S} \xi\left({q}_{n}, \omega_{s}\right)^{\top} \xi\left({k}_{m^{\prime}}, \omega_{s}\right)} \\
    &=\frac{\sum_{s=1}^{S} \xi\left({q}_{n}, \omega_{s}\right)^{\top} \sum_{m=1}^{M} \xi\left({k}_{m}, \omega_{s}\right) {v}_{m}^{\top}}{\sum_{s=1}^{S} \xi\left({q}_{n}, \omega_{s}\right)^{\top} \sum_{m^{\prime}=1}^{M} \xi\left({k}_{m^{\prime}}, \omega_{s}\right)} \\
    &:=\operatorname{RFA}\left({q}_{n}, {K}, {V}\right)
    \end{aligned}
$$
从这里不难看出，尽管公式(1)是$$\mathrm {exp}$$的无偏估计，但是RFA并不是Attention的无偏估计，这里是利用了如下事实：
$$
\mathbb E [x_i] = x,\mathbb E [y_i] = y  \not \Rightarrow  \mathbb E\left[\frac{x_i}{y_i} \right] = \frac x y \tag 2
$$
这也是本文的主要出发点，注意到公式(2)涉及到分母，这一点是比较难处理的，因此，作者引入了重要度抽样的方法：
$$
\mathbb{E}_{p(\omega)}[f(\omega)]=\mathbb{E}_{g(\omega)}\left[\frac{p(\omega)}{g(\omega)} f(\omega)\right] \approx \frac{1}{S} \sum_{s=1}^{S} \frac{p\left(\omega_{s}\right)}{g\left(\omega_{s}\right)} f\left(\omega_{s}\right) \tag 3
$$
注意到概率分布$$p(\omega _s)$$一般可以写成：
$$
p(\omega)=\tilde{p}(\omega) / Z
$$
而$$Z$$作为分母通常很难计算，所以公式(3)通常无法直接使用，为了消去$$Z$$，在公式(3)中取$$f=1$$：
$$
1=\mathbb{E}_{p(\omega)}[1]=\mathbb{E}_{g(\omega)}\left[\frac{p(\omega)}{g(\omega)} \right] \approx \frac{1}{S} \sum_{s=1}^{S} \frac{p\left(\omega_{s}\right)}{g\left(\omega_{s}\right)}  \tag 3
$$
那么：
$$
\begin{aligned}
&\mathbb{E}_{p(\omega)}[f(\omega)]=\frac{\mathbb{E}_{g(\omega)}\left[\frac{p(\omega)}{g(\omega)} f(\omega)\right]}{\mathbb{E}_{g(\omega)}\left[\frac{p(\omega)}{g(\omega)}\right]} \\
&\approx \frac{\frac{1}{S} \sum_{s=1}^{S} \frac{1}{Z} \frac{\tilde{p}\left(\omega_{s}\right)}{g\left(\omega_{s}\right)} f\left(\omega_{s}\right)}{\frac{1}{S} \sum_{s=1}^{S} \frac{1}{Z} \frac{\tilde{p}\left(\omega_{s}\right)}{g\left(\omega_{s}\right)}}=\frac{\sum_{s=1}^{S} \frac{\tilde{p}\left(\omega_{s}\right)}{g\left(\omega_{s}\right)} f\left(\omega_{s}\right)}{\sum_{s=1}^{S} \frac{\tilde{p}\left(\omega_{s}\right)}{g\left(\omega_{s}\right)}}
\end{aligned}
$$
这样就可以消去分母$$Z$$，从而让重要度抽样的方法可计算。



### RA

将之前的内容结合，最终作者得到如下结论：
$$
\operatorname{Softmax} \operatorname{Attn}\left({q}_{n}, {K}, {V}\right)=\mathbb{E}_{p_{n}(\omega)}\left[f_{n}(\omega)\right] \tag 4
$$
其中：
$$
\begin{aligned}
p(\omega)&=\sum_{m=1}^{M} \pi_{m} \mathcal{N}\left(\omega ; {q}_{n}+{k}_{m}, {I}\right) \\
f(\omega)&=\frac{\xi\left({q}_{n}, \omega\right)^{\top} \sum_{m=1}^{M} \xi\left({k}_{m}, \omega\right) {v}_{m}^{\top}}{\xi\left({q}_{n}, \omega\right)^{\top} \sum_{m^{\prime}=1}^{M} \xi\left({k}_{m^{\prime}}, \omega\right)}

\end{aligned}
$$
注意到这里一共涉及$$MN$$个概率分布$$p(\omega)$$，所以时间复杂度依然为$$O(MNd)$$，并没有带来速度提升，所以后续需要解决这点。



### LARA

首先引出MIS(multiple importance sampling)：
$$
\mathbb{E}_{p_{n}(\omega)}\left[f_{n}(\omega)\right] \approx \sum_{c=1}^{C} \alpha_{n c}\left(\omega_{c}\right) \frac{p_{n}\left(\omega_{c}\right)}{g_{c}\left(\omega_{c}\right)} f_{n}\left(\omega_{c}\right) \\
\omega_{c} \sim g_{c}(\omega), c=1,\ldots, C\\
 \sum_{c=1}^{C} \alpha_{n c} = 1
$$
这样做的好处是可以将分布数量降低到$$C\ll MN$$，通过比较复杂的推导，最终作者给出：
$$
\begin{aligned}
\alpha_{n c}\left(\omega_{c}\right)&=\frac{q_{c}\left(\omega_{c}\right)}{\sum_{c^{\prime}=1}^{C} g_{c^{\prime}}\left(\omega_{c}\right)}+r_{n c}^{\prime}-\frac{1}{C} \sum_{c=1}^{C} r_{n c}^{\prime} \\
r_{n c}^{\prime}&=\frac{\exp \left({q}_{n}^{\top} \tilde{{q}}_{c}\right)}{\sum_{n=1}^{N} \exp \left({q}_{n}^{\top} \tilde{{q}}_{c^{\prime}}\right)} \\

\end{aligned}
$$
其中$$\tilde{{q}}_{c}$$是如何计算还没有完全理清，后续进行补充。

最终的计算式：
$$
\begin{aligned}
\mathbb{E}_{p_{n}(\omega)}\left[f_{n}(\omega)\right] & \approx \frac{\sum_{c=1}^{C} \alpha_{n c}\left(\omega_{c}\right) \frac{\tilde{p}_{n}\left(\omega_{c}\right)}{q_{c}\left(\omega_{c}\right)} f_{n}\left(\omega_{c}\right)}{\sum_{c=1}^{C} \alpha_{n c}\left(\omega_{c}\right) \frac{\tilde{p}_{n}\left(\omega_{c}\right)}{q_{c}\left(\omega_{c}\right)}} \\
&:=\operatorname{LARA}\left({q}_{n}, {K}, {V}\right)\\
\tilde p_n (\omega )&=\mathcal{N}(\omega ; 0, {I}) \xi\left({q}_{n}, \omega\right)^{\top} \sum_{m=1}^{M} \xi\left({k}_{m}, \omega\right)
\end{aligned}
$$


## 时间复杂度

不难看出为$$O(NC d^2)$$。



## 训练以及loss

不变。



## 代码

暂无，详细的伪代码可以参考原论文。



## 实验以及适用场景

论文主要测试了Encoder，效果还不错，Decoder还没进行测试。



## 细节

实现的时候应该有不少技巧，等后续复现的时候进行讨论。



## 简评

理论性很强的一篇文章，但是写的很容易懂，出发点也比较明确，个人感觉比Performer这篇更值得关注。