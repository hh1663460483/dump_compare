# dump_data_ascend 数据保存说明文档

## 1. 概述

### 1.1 目的

在 HunYuan V3 模型的 vLLM-Ascend 推理流程中，于模型各层关键位置插入了 dump 代码，用于保存中间张量数据至 `/mnt/dump_data/` 目录。这些数据可用于**精度对比**和**问题定位**。

### 1.2 保存机制：`tp_save`

三个源文件中各自定义了相同的 `tp_save` 函数：

```python
def tp_save(tensor, save_path):
    tp_rank = get_tensor_model_parallel_rank()
    if tp_rank == 0:
        torch.save(tensor.cpu(), save_path)
```

**核心特性：**
- **仅 TP rank 0 保存**：避免张量并行场景下的重复写入
- **CPU 转存**：保存前将张量从设备(NPU/GPU)搬运至 CPU
- **文件格式**：虽然扩展名为 `.bin`，实际使用 `torch.save` 序列化，可通过 `torch.load("xxx.bin")` 直接加载
- **保存路径**：统一写入 `/mnt/dump_data/` 目录

### 1.3 触发条件

所有 dump 操作均受 **batch size 条件守卫**，不会在正常推理时触发：

| 条件 | 适用范围 | 说明 |
|------|---------|------|
| `hidden_states.shape[0] == 10` | 绝大多数 dump 点 | 主条件 |
| `final_hidden_states[0].shape[0] == 12` | 共享/融合专家输出 | 因 shared expert 拼接导致维度变化 |

### 1.4 层号标识

| 源文件 | 标识变量 | 含义 |
|--------|---------|------|
| `hunyuan_v3.py` | `{idx}` | DecoderLayer 的层索引，由 forward 参数传入 |
| `experts_selector.py` | `{COUNT}` | 独立全局计数器，初始值=1，每经过一个 MoE 层 +1 |
| `fused_moe.py` | `{COUNT}` | 独立全局计数器，初始值=1，每经过一个 MoE 层 +1 |

> **注意**：`experts_selector.py` 和 `fused_moe.py` 中的 `COUNT` 是**各自独立**的全局变量，但因 MoE 层数量相同，两者数值保持同步递增。

### 1.5 涉及源文件

| 文件 | tp_save 调用数 | 说明 |
|------|---------------|------|
| `vllm/model_executor/models/hunyuan_v3.py` | 15 | 模型主体各层关键位置 |
| `vllm_ascend/ops/fused_moe/experts_selector.py` | 10 | MoE 专家路由选择过程 |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 2 | MoE 专家计算前的路由结果 |

---

## 2. HunYuan V3 模型架构概览

```
Input IDs
  │
  ▼
┌─────────────────────────┐
│  VocabParallelEmbedding │  (embed_tokens)
└────────────┬────────────┘
             │
  ┌──────────▼──────────┐
  │   DecoderLayer × N  │  (每层结构见下方)
  └──────────┬──────────┘
             │
  ┌──────────▼──────────┐
  │      RMSNorm        │  (最终层归一化)
  └──────────┬──────────┘
             │
  ┌──────────▼──────────┐
  │   ParallelLMHead    │  (语言模型头)
  └──────────┬──────────┘
             │
             ▼
         Output Logits
```

每个 **DecoderLayer** 包含：
- `input_layernorm` (RMSNorm) → `self_attn` (Attention) → 残差连接
- `post_attention_layernorm` (RMSNorm) → `mlp` → 残差连接

其中 `mlp` 根据层索引不同：
- **Dense 层** (`idx < first_k_dense_replace`)：使用 `HYV3FeedForward`
- **MoE 层** (`idx >= first_k_dense_replace`)：使用 `HYV3MoEFused`

---

## 3. 全部 dump 文件清单

### 3.1 模型主体层（hunyuan_v3.py）— 15 个 dump 点

#### DecoderLayer 级别

| 序号 | 文件名 | 数据含义 | 模型位置 | 源码行 |
|:----:|--------|---------|---------|:------:|
| 1 | `layer{idx}_input.bin` | Decoder 层的原始输入 hidden_states | DecoderLayer 入口 | L590 |
| 2 | `layer{idx}_after_input_layernorm.bin` | input_layernorm (RMSNorm) 输出 | input_layernorm 之后 | L601 |

#### Attention 子层

| 序号 | 文件名 | 数据含义 | 模型位置 | 源码行 |
|:----:|--------|---------|---------|:------:|
| 3 | `layer{idx}_qkv_proj_input.bin` | Attention 输入 hidden_states | HYV3Attention 入口 | L471 |
| 4 | `layer{idx}_qkv_proj_weight.bin` | QKV 投影权重矩阵 | qkv_proj.weight | L472 |
| 5 | `layer{idx}_after_qkv_proj.bin` | QKV 投影输出（Q/K/V 拼接张量） | qkv_proj 之后 | L476 |
| 6 | `layer{idx}_after_attn_op.bin` | 注意力计算输出（attention operation） | attn 计算之后 | L498 |
| 7 | `layer{idx}_after_attn_layer.bin` | 注意力子层输出（含残差连接） | self_attn + 残差之后 | L611 |

#### Dense FFN 子层（仅 Dense 层）

| 序号 | 文件名 | 数据含义 | 模型位置 | 源码行 |
|:----:|--------|---------|---------|:------:|
| 8a | `layer{idx}_after_ffn_gate_up_proj.bin` | gate_up_proj 线性投影输出 | FeedForward.gate_up_proj 之后 | L133 |
| 8b | `layer{idx}_after_ffn_act_fn.bin` | 激活函数 (SiluAndMul) 输出 | FeedForward.act_fn 之后 | L137 |

#### MoE 子层（仅 MoE 层）

| 序号 | 文件名 | 数据含义 | 模型位置 | 源码行 |
|:----:|--------|---------|---------|:------:|
| 8c | `layer{idx}_experts_input.bin` | MoE 层输入 hidden_states | HYV3MoEFused 入口 | L280 |
| 8d | `layer{idx}_experts_e_score_correction_bias.bin` | 专家分数修正偏置向量 | experts.e_score_correction_bias | L290 |
| 8e | `layer{idx}_after_router.bin` | 路由器 (gate) 输出的 logits | gate 线性层之后 | L291 |
| 8f | `layer{idx}_after_experts_shared.bin` | 共享专家 (shared_mlp) 输出 | shared_mlp 之后 | L299 |
| 8g | `layer{idx}_after_experts_fused.bin` | 路由专家融合计算输出 | fused experts 之后 | L300 |

#### MLP/MoE 结束

| 序号 | 文件名 | 数据含义 | 模型位置 | 源码行 |
|:----:|--------|---------|---------|:------:|
| 9 | `layer{idx}_after_mlp_layer.bin` | MLP/MoE 子层输出（含残差连接） | mlp + 残差之后 | L626 |

---

### 3.2 专家路由选择（experts_selector.py）— 10 个 dump 点

以下 dump 发生在 MoE 层的**专家选择**阶段，记录路由权重从原始 logits 到最终 top-k 权重的完整计算过程：

| 序号 | 文件名 | 数据含义 | 所在函数 | 源码行 |
|:----:|--------|---------|---------|:------:|
| R1 | `layer{COUNT}_grouped_topk_input.bin` | 专家选择器的输入 hidden_states | `_native_select_experts` | L321 |
| R2 | `layer{COUNT}_grouped_topk_gating_output.bin` | 路由 logits（原始门控输出） | `_native_select_experts` | L322 |
| R3 | `layer{COUNT}_grouped_topk_scores.bin` | sigmoid 激活后的路由分数 | `_native_select_experts` | L328 |
| R4 | `layer{COUNT}_grouped_topk_scores_with_bias.bin` | 加上 e_score_correction_bias 后的分数 | `_select_expert_use_group_topk` | L205 |
| R5 | `layer{COUNT}_grouped_topk_after_bias_group_scores.bin` | 按专家组 top-2 求和后的分组权重 | `_native_grouped_topk` | L157 |
| R6 | `layer{COUNT}_grouped_topk_group_idx.bin` | 选中的专家组索引 (top-k group) | `_native_grouped_topk` | L166 |
| R7 | `layer{COUNT}_grouped_topk_tmp_scores.bin` | 组掩码后的分数（未选中组设为 -inf） | `_native_grouped_topk` | L175 |
| R8 | `layer{COUNT}_grouped_topk_topk_weights_1.bin` | top-k 选择后用原始无偏分数聚集的权重 | `_select_expert_use_group_topk` | L222 |
| R9 | `layer{COUNT}_grouped_topk_topk_weights_2.bin` | 归一化 (renormalize) 后的权重 | `_select_expert_use_group_topk` | L231 |
| R10 | `layer{COUNT}_grouped_topk_topk_weights_3.bin` | 乘以缩放因子 (×2.826) 后的最终路由权重 | `_select_expert_use_group_topk` | L235 |

> **COUNT 递增时机**：在 `_select_expert_use_group_topk` 末尾（L238），当 `hidden_states.shape[0] == 10` 时执行 `COUNT += 1`。

---

### 3.3 MoE 融合计算入口（fused_moe.py）— 2 个 dump 点

以下 dump 发生在专家选择完成后、实际专家计算 (`fused_experts`) 之前：

| 序号 | 文件名 | 数据含义 | 所在函数 | 源码行 |
|:----:|--------|---------|---------|:------:|
| F1 | `layer{COUNT}_fused_topk_weights.bin` | 传入 fused_experts 的最终路由权重 | `AscendUnquantizedFusedMoEMethod.apply` | L125 |
| F2 | `layer{COUNT}_fused_topk_ids.bin` | 传入 fused_experts 的最终专家 ID | `AscendUnquantizedFusedMoEMethod.apply` | L126 |

> **COUNT 递增时机**：在 L127 保存完两个文件后立即 `COUNT += 1`。

---

## 4. 执行顺序流程图

### 4.1 Dense 层（idx < first_k_dense_replace）— 每层 9 个 dump 文件

```
┌───────────────────────────────────────────────────────────────────┐
│                        DecoderLayer                               │
│                                                                   │
│  ① layer{idx}_input.bin                                          │
│  │                                                                │
│  ▼                                                                │
│  input_layernorm (RMSNorm)                                       │
│  │                                                                │
│  ② layer{idx}_after_input_layernorm.bin                          │
│  │                                                                │
│  ▼                                                                │
│  ┌─────────────────── Attention ───────────────────┐             │
│  │                                                  │             │
│  │  ③ layer{idx}_qkv_proj_input.bin                │             │
│  │  ④ layer{idx}_qkv_proj_weight.bin               │             │
│  │  │                                               │             │
│  │  ▼                                               │             │
│  │  qkv_proj (QKVParallelLinear)                   │             │
│  │  │                                               │             │
│  │  ⑤ layer{idx}_after_qkv_proj.bin               │             │
│  │  │                                               │             │
│  │  ▼                                               │             │
│  │  RoPE → Attention 计算 → o_proj                  │             │
│  │  │                                               │             │
│  │  ⑥ layer{idx}_after_attn_op.bin                 │             │
│  └──┼──────────────────────────────────────────────┘             │
│     │                                                             │
│     ▼  (+残差连接)                                                │
│                                                                   │
│  ⑦ layer{idx}_after_attn_layer.bin                               │
│  │                                                                │
│  ▼                                                                │
│  post_attention_layernorm (RMSNorm)                              │
│  │                                                                │
│  ▼                                                                │
│  ┌─────────── FeedForward (Dense FFN) ────────────┐             │
│  │                                                  │             │
│  │  gate_up_proj (MergedColumnParallelLinear)      │             │
│  │  │                                               │             │
│  │  ⑧ layer{idx}_after_ffn_gate_up_proj.bin       │             │
│  │  │                                               │             │
│  │  ▼                                               │             │
│  │  act_fn (SiluAndMul)                            │             │
│  │  │                                               │             │
│  │  ⑨ layer{idx}_after_ffn_act_fn.bin             │             │
│  │  │                                               │             │
│  │  ▼                                               │             │
│  │  down_proj (RowParallelLinear)                  │             │
│  └──┼──────────────────────────────────────────────┘             │
│     │                                                             │
│     ▼  (+残差连接)                                                │
│                                                                   │
│  ⑩ layer{idx}_after_mlp_layer.bin                                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

> 注：Dense 层实际为 9 个 dump 文件（序号 ③④ 在同一位置但保存不同数据）。

---

### 4.2 MoE 层（idx >= first_k_dense_replace）— 每层最多 19 个 dump 文件

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            DecoderLayer                                   │
│                                                                           │
│  ① layer{idx}_input.bin                                                  │
│  ② layer{idx}_after_input_layernorm.bin                                  │
│  │                                                                        │
│  ▼                                                                        │
│  ┌─────────── Attention（同 Dense 层）──────────┐                        │
│  │  ③ qkv_proj_input → ④ qkv_proj_weight        │                        │
│  │  → ⑤ after_qkv_proj → ⑥ after_attn_op        │                        │
│  └───────────────────────────────────────────────┘                        │
│  │                                                                        │
│  ⑦ layer{idx}_after_attn_layer.bin                                       │
│  │                                                                        │
│  ▼                                                                        │
│  post_attention_layernorm (RMSNorm)                                      │
│  │                                                                        │
│  ▼                                                                        │
│  ┌──────────────────── HYV3MoEFused ──────────────────────────────┐      │
│  │                                                                  │      │
│  │  ⑧ layer{idx}_experts_input.bin          [MoE入口]             │      │
│  │  ⑨ layer{idx}_experts_e_score_correction_bias.bin              │      │
│  │  │                                                               │      │
│  │  ▼  gate (ReplicatedLinear)                                     │      │
│  │  │                                                               │      │
│  │  ⑩ layer{idx}_after_router.bin           [路由器logits]        │      │
│  │  │                                                               │      │
│  │  ▼                                                               │      │
│  │  ┌──────── experts_selector.py: 专家选择 ────────────────┐     │      │
│  │  │                                                         │     │      │
│  │  │  R1  grouped_topk_input.bin          [选择器输入]      │     │      │
│  │  │  R2  grouped_topk_gating_output.bin  [门控输出]        │     │      │
│  │  │  │                                                      │     │      │
│  │  │  ▼  sigmoid()                                          │     │      │
│  │  │  │                                                      │     │      │
│  │  │  R3  grouped_topk_scores.bin         [sigmoid分数]     │     │      │
│  │  │  │                                                      │     │      │
│  │  │  ▼  + e_score_correction_bias                          │     │      │
│  │  │  │                                                      │     │      │
│  │  │  R4  grouped_topk_scores_with_bias.bin [加偏置分数]    │     │      │
│  │  │  │                                                      │     │      │
│  │  │  ▼  _native_grouped_topk()                             │     │      │
│  │  │  │                                                      │     │      │
│  │  │  R5  grouped_topk_after_bias_group_scores.bin          │     │      │
│  │  │  │   [按组top-2求和的分组权重]                          │     │      │
│  │  │  R6  grouped_topk_group_idx.bin                        │     │      │
│  │  │  │   [选中的专家组索引]                                 │     │      │
│  │  │  R7  grouped_topk_tmp_scores.bin                       │     │      │
│  │  │  │   [组掩码后分数]                                     │     │      │
│  │  │  │                                                      │     │      │
│  │  │  ▼  topk() 选择 + 用原始分数聚集                       │     │      │
│  │  │  │                                                      │     │      │
│  │  │  R8  grouped_topk_topk_weights_1.bin [无偏聚集权重]    │     │      │
│  │  │  │                                                      │     │      │
│  │  │  ▼  renormalize                                        │     │      │
│  │  │  │                                                      │     │      │
│  │  │  R9  grouped_topk_topk_weights_2.bin [归一化权重]      │     │      │
│  │  │  │                                                      │     │      │
│  │  │  ▼  × 2.826 (缩放因子)                                │     │      │
│  │  │  │                                                      │     │      │
│  │  │  R10 grouped_topk_topk_weights_3.bin [最终路由权重]    │     │      │
│  │  │  │                                                      │     │      │
│  │  │  ▼  COUNT++ (experts_selector)                         │     │      │
│  │  └──┼─────────────────────────────────────────────────────┘     │      │
│  │     │                                                            │      │
│  │     ▼                                                            │      │
│  │  ┌──────── fused_moe.py: 专家计算入口 ──────────────────┐      │      │
│  │  │                                                        │      │      │
│  │  │  F1  fused_topk_weights.bin   [最终路由权重]          │      │      │
│  │  │  F2  fused_topk_ids.bin       [最终专家ID]            │      │      │
│  │  │  │                                                     │      │      │
│  │  │  ▼  COUNT++ (fused_moe)                               │      │      │
│  │  │  │                                                     │      │      │
│  │  │  ▼  fused_experts() 实际专家计算                      │      │      │
│  │  └──┼─────────────────────────────────────────────────────┘      │      │
│  │     │                                                            │      │
│  │  ⑪ layer{idx}_after_experts_shared.bin  [共享专家输出]          │      │
│  │  ⑫ layer{idx}_after_experts_fused.bin   [路由专家输出]          │      │
│  │                                                                  │      │
│  └──┼──────────────────────────────────────────────────────────────┘      │
│     │                                                                      │
│     ▼  (+残差连接)                                                        │
│                                                                           │
│  ⑬ layer{idx}_after_mlp_layer.bin                                        │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 调用链与数据流向

```
HYV3DecoderLayer.forward()                        [hunyuan_v3.py]
├── dump: input, after_input_layernorm
├── HYV3Attention.forward()
│   ├── dump: qkv_proj_input, qkv_proj_weight, after_qkv_proj, after_attn_op
├── dump: after_attn_layer
│
├── [Dense 层] HYV3FeedForward.forward()
│   ├── dump: after_ffn_gate_up_proj, after_ffn_act_fn
│
├── [MoE 层] HYV3MoEFused.forward()
│   ├── dump: experts_input, e_score_correction_bias, after_router
│   ├── self.experts()  →  AscendSharedFusedMoE / AscendFusedMoE
│   │   ├── select_experts()                      [experts_selector.py]
│   │   │   └── _native_select_experts()
│   │   │       ├── dump: R1(input), R2(gating_output), R3(scores)
│   │   │       └── _select_expert_use_group_topk()
│   │   │           ├── dump: R4(scores_with_bias)
│   │   │           ├── _native_grouped_topk()
│   │   │           │   └── dump: R5(group_scores), R6(group_idx), R7(tmp_scores)
│   │   │           ├── dump: R8(topk_weights_1), R9(topk_weights_2), R10(topk_weights_3)
│   │   │           └── COUNT++ (experts_selector)
│   │   │
│   │   └── AscendUnquantizedFusedMoEMethod.apply()  [fused_moe.py]
│   │       ├── dump: F1(topk_weights), F2(topk_ids)
│   │       ├── COUNT++ (fused_moe)
│   │       └── fused_experts()  →  实际专家 MLP 计算
│   │
│   ├── dump: after_experts_shared, after_experts_fused
│
└── dump: after_mlp_layer
```

---

## 6. 注意事项

### 6.1 文件格式

- 扩展名为 `.bin`，但底层使用 `torch.save` 序列化
- 加载方式：`tensor = torch.load("/mnt/dump_data/layer0_input.bin")`
- 所有张量在保存前已转换为 CPU 张量

### 6.2 COUNT 计数器

`experts_selector.py` 和 `fused_moe.py` 各维护独立的 `COUNT` 全局变量：
- 两者初始值均为 `1`
- 每经过一个 MoE 层各自递增 `+1`
- 由于每个 MoE 层都会依次调用两个文件，两者的 COUNT 值始终保持同步
- COUNT 值与 `hunyuan_v3.py` 中的 `idx` 的对应关系为：`COUNT = idx - first_k_dense_replace + 1`（即 COUNT 从第一个 MoE 层开始计数）

### 6.3 条件性 dump

部分 dump 点有额外条件：
- **R3** (`grouped_topk_scores.bin`)：仅在 `scoring_func == "sigmoid"` 时保存
- **R4** (`scores_with_bias.bin`)：仅在 `e_score_correction_bias is not None` 时保存
- **R5** (`after_bias_group_scores.bin`)：仅在有 bias 时保存（使用 top-2 求和而非 max）
- **R8** (`topk_weights_1.bin`)：仅在 `e_score_correction_bias is not None` 时保存

### 6.4 Dense 层 vs MoE 层

| 特性 | Dense 层 | MoE 层 |
|------|---------|--------|
| 层索引范围 | `idx < first_k_dense_replace` | `idx >= first_k_dense_replace` |
| MLP 类型 | HYV3FeedForward | HYV3MoEFused |
| dump 文件数/层 | 9 个 | 最多 19 个 (7 + 10 + 2) |
| 路由相关 dump | 无 | 10 个 (experts_selector) + 2 个 (fused_moe) |
