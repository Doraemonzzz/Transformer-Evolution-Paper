# Transformer-Evolution

记录Transformer升级的论文笔记，主要是一些简要记录，能达到看懂并复现出代码的程度即可。



## 为什么建该仓库

Transformer模块可以用如下两个式子表示：


$$
\begin{aligned}
\mathbf X_1 &=\mathrm{Norm}(\mathbf X + \mathrm{MHA}(\mathbf X,\mathbf Y))\\
\mathbf O &= \mathrm{Norm}(\mathbf X_1 + \mathrm{FFN}(\mathbf X_1))
\end{aligned}
$$
通过上式，很容易将Transformer模块解耦，本仓库的目的就是记录对每个解耦后的模块改进的论文，最终给出一个更好的Transformer，即Transformer-Evolution。



## 目录说明

- Act(Activation function)：激活函数；
- Arch(Architecture)：改进Transformer整体结构；
- FFN：讨论Transformer中FFN的作用，或者其改进；
- Head：讨论Attention中多头的作用；
- Memory：在Transformer中增加memory模块；
- MHA_Kernel：利用Kernel method改进Attention模块（$QKV\to Q(KV)$）；
- MHA_Sparse_And_LowRank：利用稀疏或者低秩的假设降低Attention计算复杂度；
  - 这个描述不太准确，后续会修改；
- Normalize_And_Residual：讨论Transformer中各种Norm的Residual的区别；
  - 该仓库包括范围有点广，后续可能会精细分类；
- Pe(Positional Embedding)：讨论各种位置编码，主要以相对位置编码为主；
- Pretrain：一些NLP的预训练方式（非关注重点，主要是做个笔记）；
- Softmax：专门针对Softmax的讨论，可能是Softmax的作用，缺陷等等；
- Other：其他论文；



## TODO

相对位置编码综述：

- [https://arxiv.org/abs/2102.11090](https://arxiv.org/abs/2102.11090)