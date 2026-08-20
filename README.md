# minimind 模型架构

![MiniMind 模型架构](assets/LLM-structure.jpg)

## 1 逐层分析

> 假设输入 batch 为 1，token 数量为 26。

### 1.1 Tokenizer Encode

- **输入**：字符串。

- **计算**：将输入字符串转换成 token。
- **输出**：每个 token 的 token ID， `(1, 26)`。

### 1.2 Input Embedding

- **输入**：token ID，`(1, 26)`。
- **权重**：Embedding 权重，`(6400, 768)`。
- **计算**：根据 token ID 查权重表。
- **输出**：隐藏状态，`(1, 26, 768)`。

### 1.3 RMSNorm

- **输入**：`(1, 26, 768)`。

- **权重**：缩放参数 $\gamma_i$，`(768)`。

- **计算**：对每一行 768 维向量计算均方根并缩放。

    > $\epsilon$ 用于防止除以零。

    $$
    \text{RMS}(x)=\sqrt{\frac{1}{768}\sum_{i=1}^{768}x_i^2+\epsilon}\\
    y_i=\frac{x_i}{\text{RMS}(x)}\cdot\gamma_i
    $$

- **输出**：`(1, 26, 768)`。

### 1.4 生成 Q、K、V

- **输入**：`(1, 26, 768)`。

- **权重**：$W_Q$ `(768, 768)`，$W_K$ `(768, 384)`，$W_V$ `(768, 384)`。

    > pytorch 中是转置的，如 `k_proj.weight: (384, 768)`。

- **计算**：

    - $Q=XW_Q$ `(1, 26, 768)`；$K=XW_K$ `(1, 26, 384)`；$V=XW_V$ `(1, 26, 384)`。
    - Q 拆成 8 个头，`(1, 26, 8, 96)`；K 和 V 拆成 4 个头，`(1, 26, 4, 96)`。
    - 交换 token 和 head 维，变成 `(1, 8, 26, 96)` 和 `(1, 4, 26, 96)`。

- **输出**：Q `(1, 8, 26, 96)`，K `(1, 4, 26, 96)`，V `(1, 4, 26, 96)`。

### 1.5 对 Q、K 做 RMSNorm

形状不变。

### 1.6 RoPE 位置编码

- **输入**：Q `(1, 8, 26, 96)`，K `(1, 4, 26, 96)`，V `(1, 4, 26, 96)`。
- **计算**：只对 Q 和 K 进行旋转，形状不改变。
- **输出**：Q `(1, 8, 26, 96)`，K `(1, 4, 26, 96)`，V `(1, 4, 26, 96)`。

### 1.7 扩展 K、V 头

- **输入**：Q `(1, 8, 26, 96)`，K `(1, 4, 26, 96)`，V `(1, 4, 26, 96)`。
- **计算**：当前 Q、K、V 头数不一样，让每一组 K、V 被两个 Q 头共享，即将 K、V 的 4 个头每个重复 2 次。
- **输出**：Q `(1, 8, 26, 96)`，K `(1, 8, 26, 96)`，V `(1, 8, 26, 96)`。

### 1.8 计算 Attention 权重

- **输入**：Q `(1, 8, 26, 96)`，K `(1, 8, 26, 96)`。

- **计算**：

    - 每个头取后 2 维做 $S=\frac{QK^T}{\sqrt{96}}$，然后合到一起得到 `(1, 8, 26, 26)`。结果的后 2 维表示第 i 个 token 与第 j 个 token 的匹配程度。

    - 加入掩码，一个 token 只能读取自己和前面的 token，被禁用的位置会加上负无穷。

        ```
        i\j  0    1    2    3
        0    ✓    ×    ×    ×
        1    ✓    ✓    ×    ×
        2    ✓    ✓    ✓    ×
        3    ✓    ✓    ✓    ✓
        ```

    - 对每一行进行 Softmax，$k$ 表示循环一行的所有 $j$。
        $$
        A_{i,j}=\frac{e^{S_{i,j}}}{\sum_k e^{S_{i,k}}}
        $$

- **输出**：注意力权重 A `(1, 8, 26, 26)`。

### 1.9 用注意力权重读取 V

> 这一步 Attention 的结果不再只是当前 token 自己的信息，而是融合了上下文。

- **输入**：注意力权重 A `(1, 8, 26, 26)`，V `(1, 8, 26, 96)`。
- **计算**：$O=AV$。对于每一个 token，设其注意力权重为 $[a_0,\cdots,a_{25}]$（A 的最后一维的每个数），而每一个 token 都有一个 $V_i$（V 的最后一维长度为 96 的向量），实际上就是一个内积 $O_i=\sum a_iV_i$。
- **输出**：O `(1, 8, 26, 96)`。

### 1.10 拼接多头并进行输出投影

- **输入**：O `(1, 8, 26, 96)`。
- **权重**：$W_O$ `(768, 768)`。
- **计算**：
    - 先交换维度得到 `(1, 26, 8, 96)`。
    - 然后合并最后两维得到 `(1, 26, 768)`。
    - 混合各个头：$Y=OW_O$。
- **输出**：`(1, 26, 768)`。

### 1.11 残差连接

- **输入**：原始输入 x `(1, 26, 768)`，Attention 输出 `(1, 26, 768)`。
- **计算**：逐元素相加，$h=x+\text{Attention}(x)$。
- **输出**：h `(1, 26, 768)`。

### 1.12 RMSNorm

同理，形状不变。

### 1.13 MLP 的扩维与门控

> 也叫 FFN，Feed-Forward Network。

- **输入**：z `(1, 26, 768)`。
- **权重**：$W_{gate}$ `(768, 2432)`，$W_{up}$ `(768, 2432)`。
- **计算**：gate 分支像一个可学习的开关，控制 up 分支中的信息通过多少。
    - 升维 $g=zW_{gate},u=zW_{up}$。
    - 对 gate 分支使用 SiLU，$\text{SiLU}(g)=g\cdot \sigma(g)$，$\sigma$ 是 Sigmoid（$\sigma(x)=\frac{1}{1+e^{-x}}$）。
    - 两个分支逐元素相乘，$m=\text{SiLU}{g}\odot u$。
- **输出**：m `(1, 26, 2432)`。

### 1.14 down_proj 压回 768 维

- **输入**：m `(1, 26, 2432)`。
- **权重**：$W_{down}$ `(2432, 768)`。
- **计算**：$o=mW_{down}$。
- **输出**：o `(1, 26, 768)`。

### 1.15 Dropout 和残差连接

- **输入**：o `(1, 26, 768)`，之前的 h `(1, 26, 768)`。
- **计算**：
    - Dropout 在推理中不使用。
    - $y=h+o$。
- **输出**：`(1, 26, 768)`。

### 1.16 Final RMSNorm

经过 8 个 Transformer Block 后，得到 hidden_states `(1, 26, 768)`。经过 RMSNorm 后形状不变。

### 1.17 LM Head 映射到词表

- **输入**：hidden_states `(1, 26, 768)`。
- **权重**：$W_{lm}$ `(6400, 768)`，MiniMind 设置了 tie_word_embeddings = True 所以这里的矩阵和 Embedding 共享权重。
- **计算**：LM Head 是一个线性层，把每个 token 的 768 维隐藏向量映射到词表大小 6400。$\text{logits}=XW_{lm}^T$。
- **输出**：`(1, 26, 6400)`

### 1.18 生成下一个 token

- **输入**：`(1, 26, 6400)`。
- **计算**：
    - 只取最后一个位置的 logits，形状变为 `(6400)`。
    - 用 Softmax 转换为概率。
    - 选择下一个 token。方法很多，比如随机采样、Temperature、Top-k 和 Top-p。
- **输出**：下一个 token。接到输入后面，再次调用模型。

## 2 流程图

```mermaid
%%{init: {'flowchart': {'curve': 'basis'}}}%%
flowchart BT
	Input("Token ID")
	Input --(26,) INT64--> Embed
	Embed("Embedding<br>(6400,768) INT8")
	Embed --(26,768) INT16--> Norm1
	Embed --(26,768) INT16--> Res1
	subgraph Transformer
		Norm1("RMSNorm<br>(768,) INT16")
		Norm1 --(26,768) INT16--> Qproj & Kproj & Vproj
		Qproj("Q_proj<br>(768,768) INT8")
		Kproj("K_proj<br>(768,384) INT8")
		Vproj("V_proj<br>(768,384) INT8")
		Qproj --(8,26,96) INT16--> NormQ
		Kproj --(4,26,96) INT16--> NormK
		Vproj --(4,26,96) INT16----> ExtV
		NormQ("RMSNorm<br>(768,) INT16")
		NormK("RMSNorm<br>(768,) INT16")
		NormQ --(8,26,96) INT16--> RoPEQ
		NormK --(4,26,96) INT16--> RoPEK
		RoPEQ("RoPE")
		RoPEK("RoPE")
		RoPEK --(4,26,96) INT16--> ExtK
		ExtK("EXT")
		ExtV("EXT")
		ExtK --(8,26,96) INT16--> QK
		QK("Score & Mask")
		RoPEQ --(8,26,96) INT16---> QK
		QK --(8,26,26) INT16--> Soft
		Soft("SoftMax")
		Soft --(8,26,26) UINT16--> Att
		ExtV --(8,26,96)----> Att
		Att("Attention")
		Att --(26,768) INT16--> Oproj
		Oproj("O_proj<br>(768,768) INT8")
		Res1("Residual")
		Oproj --(26,768) INT16--> Res1
		
		Res1 --(26,768) INT16--> Norm2
		Norm2("RMSNorm<br>(768,) INT16")
		Norm2 --(26,768) INT16--> Upproj & Gateproj
		Upproj("Up_proj<br>(768,2432) INT8")
		Gateproj("Gate_proj<br>(768,2432) INT8")
		Gateproj --(26,2432) INT16--> SiLU
		Upproj --(26,2432) INT16---> Mul
		SiLU("SiLU")
		Mul("Point-wise Mul")
		SiLU --(26,2432) INT16--> Mul
		Mul --(26,2432) INT16--> Downproj
		Downproj("Down_proj<br>(2432,768) INT8")
		Res1 & Downproj --(26,768) INT16--> Res2
		Res2("Residual")
	end
	
	Res2 --(1,768) INT16--> NormF
	NormF("RMSNorm<br>(768,) INT16")
	NormF --(1,768) INT16--> LM
	LM("LM Head<br>(768,6400) INT8")
	LM --(1,6400) INT16--> SoftF --(1,6400)--> Logits
	SoftF("Softmax")
```

## 3 GEMM

> `T` 表示 token 个数，`t < 256`。

| 名称      | 激活 X     | 权重 W       | 数量 |
| --------- | ---------- | ------------ | ---- |
| Q         | `(T,768)`  | `(768,768)`  | 1    |
| K         | `(T,768)`  | `(768,384)`  | 1    |
| V         | `(T,768)`  | `(768,384)`  | 1    |
| Score(QK) | `(T,96)`   | `(96,T)`     | 8    |
| Attention | `(T,T)`    | `(T,96)`     | 8    |
| O         | `(T,768)`  | `(768,768)`  | 1    |
| Gate      | `(T,768)`  | `(768,2432)` | 1    |
| Up        | `(T,768)`  | `(768,2432)` | 1    |
| Down      | `(T,2432)` | `(2432,768)` | 1    |
| lm_head   | `(T,768)`  | `(768,6400)` | 1    |

## 4 架构与软硬件交互

![软硬件交互](assets/软硬交互.png)

### 4.1 软硬件交互

PS 是整个计算流程的调度者，PL 中的 DMA 和 GEMM Engine 被动等待 PS 下发命令。AXI-Lite 只传输配置和状态，不传输矩阵数据；矩阵数据由 DMA 经 HP 口在 DDR 与片上 Buffer 之间搬运。

PS 通过 AXI-Lite 读写两组寄存器：

| 模块 | 主要配置 | 控制与状态 |
| ---- | -------- | ---------- |
| DMA | DDR 地址、Buffer、方向、长度和 Stride | `START/BUSY/DONE/ERROR` |
| GEMM | `K_LEN`、`VALID_ROWS`、`VALID_COLS` | `START/BUSY/DONE/ERROR` |

一次 Tile 计算由 PS 按以下顺序调度：

1. 配置 DMA，将 X 从 DDR 搬入 X Buffer，等待 `DMA_DONE`。
2. 配置 DMA，将 W 从 DDR 搬入 Weight Buffer，等待 `DMA_DONE`。
3. 配置并启动 GEMM，等待 `GEMM_DONE`，此时 O Buffer 中的结果有效。
4. 配置 DMA，将 O Buffer 写回 DDR，等待 `DMA_DONE`。

PL 不会自动发起下一次搬运或计算。GEMV 复用相同流程，只需设置 `VALID_ROWS=1`，此时 X Buffer 和 O Buffer 只有一行有效。

### 4.2 GEMM 数据通路

![tinyTPU 架构图](assets/architecture.jpg)

$$
O=XW
$$

16×16 阵列每次计算一个最多 16×16 的输出 Tile。PS 负责沿 M、N 维分块，PL GEMM CTRL 负责一次 Tile 内 K 维的乘加、流水线排空和结果写入 O Buffer。

1. DMA 加载最多16行 X 到 X Row Buffer。
2. DMA 加载当前输出 Tile 对应的权重到 Weight Buffer。
3. GEMM CTRL 连续执行 K 次乘加。
4. 有效结果写入 O Buffer。
5. PS 启动 DMA，将有效结果写回 DDR。

| 操作 | 源 | 目标 | 说明 |
| ---- | -- | ---- | ---- |
| `dma_ddr_to_x` | DDR | X Row Buffer | PS 配置、启动并等待完成 |
| `dma_ddr_to_w` | DDR | Weight Buffer | PS 配置、启动并等待完成 |
| `gemm_tile` | X/Weight Buffer | O Buffer | 计算一个最多16×16的 Tile |
| `dma_o_to_ddr` | O Buffer | DDR | PS 配置、启动并等待完成 |

```python
def gemm(X_addr, W_addr, O_addr, M, K, N):
    for row_base in range(0, M, 16):
        valid_rows = min(16, M - row_base)

        dma_ddr_to_x(X_addr, row_base, valid_rows)
        wait_dma_done()

        for col_base in range(0, N, 16):
            valid_cols = min(16, N - col_base)

            dma_ddr_to_w(W_addr, col_base, valid_cols)
            wait_dma_done()

            gemm_tile(K, valid_rows, valid_cols)
            wait_gemm_done()

            dma_out_to_ddr(O_addr, row_base, col_base,
                           valid_rows, valid_cols)
            wait_dma_done()
```

- **X Row Buffer**：304kb，RAM
    - depth：2432
    - width：16×8
- **Output Row Buffer**：304kb
    - depth：2432
    - width：16×8
- **Weight Buffer**：FIFO
    - depth：2432
    - width：16×32

## 5 当前存在的问题

使用 KV Cache 后，Prefill 仍以 GEMM 为主，但 Decode 每轮只计算一个新 token。Q/K/V/O Projection、FFN 和 LM Head 都变成 `M=1` 的 GEMV；在当前16×16阵列上只有一行 MAC 有效，计算量明显下降。

Decode 的主要压力随之从计算转向数据搬运：

- 模型权重仍需从 DDR 搬入 Weight Buffer；
- 每层需要读取此前所有 token 的 K、V；
- KV Cache 读取量随上下文长度 `L` 线性增长；
- 同一序列的下一个 token 依赖当前结果，难以通过并行多个 token 隐藏访存延迟。

MiniMind 有8层、4个 KV Head，Head Dim 为96。若 K/V 使用 INT8，每生成一个 token，仅遍历一次全部 KV Cache 就需要读取：

$$
2\times8\times L\times4\times96=6144L\ \text{Byte}
$$

Zynq-7020 提供4个64-bit HP口，但它们分成两组独立通道。对于当前设计，一条数据流每拍可利用的最大并行数据宽度按128 bit计算：

$$
128\ \text{bit}=16\ \text{Byte}=16\times\text{INT8}
$$

这里的128 bit是单拍并行数据宽度；实际带宽还要乘以HP口频率，并扣除AXI握手、突发边界和仲裁开销。因此，仅KV Cache读取所需的理论最少传输拍数为：

$$
\frac{6144L}{16}=384L
$$

例如 `L=256` 时，每生成一个 token 需要读取约1.5 MiB KV数据，对应至少98,304个128-bit传输拍，尚未计入模型权重和中间结果。因此 Decode 阶段容易受到 DDR/HP 带宽限制，而不是受 MAC 数量限制。

第一版先复用现有阵列完成 GEMV，接受 `1/16` 的行方向 MAC 利用率。后续可由学生继续优化多 HP 口并行、KV Cache 分块、片上缓存、Ping-Pong Buffer、权重驻留或多请求 Batch。
