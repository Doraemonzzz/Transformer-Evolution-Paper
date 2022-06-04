## Multi-Head Attention: Collaborate Instead of Concatenate

论文地址：

- https://arxiv.org/abs/2006.16362



## 整体思路以及计算方式

对$Q,K$降维，然后通过对角阵增加模型表达力，最后达到相当的效果。

计算方式：

- 压缩率$p$，$h$为头数，输入$X\in \mathbb R^{n\times d}, Y\in \mathbb R^{m\times d}$，$m_i\in \mathbb R,i=1,\ldots, h$，记$d_1=\lfloor  pd\rfloor$
- for $i=1,\ldots, h$
  - 计算$Q= XW_Q^{(i)} \in \mathbb R^{n\times (d_1 /h)}, K= XW_K^{(i)} \in \mathbb R^{m\times (d_1 /h)}, V = XW_V^{(i)}\in \mathbb R^{m\times (d /h)}$
  - $H^{(i)}=\mathrm{MHA}(Q\mathrm{diag}(m_i), K, V)$
- $\mathrm{Concat}[H^{(i)}]$

说明：

- 尽管原文中不同头算$Q,K$的$W_Q,W_K$是共享的，但实际实现的时候并不是；



## 时间复杂度

对于每个头，时间复杂度为：
$$
O(nmd_1/h + mnd/h)=O(nmpd/h + mnd/h)
$$
所以$h$个头的时间复杂度为：
$$
O(mn(p+1)d )
$$


## 训练以及loss

略过。



## 代码

- [https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py#L514](https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py#L514)
- [https://github.com/epfml/collaborative-attention/blob/master/src/collaborative_attention/collaborative_attention.py](https://github.com/epfml/collaborative-attention/blob/master/src/collaborative_attention/collaborative_attention.py)
- [https://github.com/epfml/collaborative-attention](https://github.com/epfml/collaborative-attention)



## 实验以及适用场景

因为只改了Head部分，所以适用于所有场景；作者进行了大量实验，效果均不错。



## 细节

降维比例为30%的时候也能达到相当效果。



## 简评

总结：

- 很简洁的思路，通过降维减少参数量，然后再通过少量参数恢复性能；
- 非常简洁，值得复现；