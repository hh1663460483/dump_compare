# Dump Data Precision Comparison

HunYuan V3 模型推理过程中 Ascend 950 (NPU) 与 H20 (GPU) 的逐层 dump 数据精度对比工具集。

## 项目结构

```
dump_compare/
├── compare_dump_precision.py          # 主精度对比脚本（dump 数据）
├── compare_layer_precision.py         # 逐层逐算子精度对比脚本
├── compv8.py                          # MoE 专家选择 (topk_ids) 分析脚本
├── precision_result.xlsx              # Excel 报告模板
├── dump_data_ascend_说明文档.md        # dump 数据保存点说明文档
├── dump_data_ascend/                  # Ascend 950 dump 数据目录
├── dump_data_h20/                     # H20 GPU dump 数据目录
├── dump_precision_report.md           # [生成] Markdown 精度报告
├── dump_precision_report.xlsx         # [生成] Excel 精度报告
└── dump_charts/                       # [生成] 可视化图表
    ├── snr_by_layer.png
    ├── within_1ulp_by_layer.png
    ├── snr_heatmap.png
    └── error_accumulation.png
```

## 环境要求

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- openpyxl

```bash
pip install torch numpy matplotlib openpyxl
```

## 使用方法

### 主脚本：compare_dump_precision.py

对比两个 dump 目录下的所有 `.bin` 文件，按模型执行顺序计算精度指标，输出 Markdown 报告、Excel 表格和可视化图表。

```bash
python compare_dump_precision.py \
    --baseline-dir ./dump_data_h20 \
    --test-dir ./dump_data_ascend \
    --output-report ./dump_precision_report.md \
    --output-charts-dir ./dump_charts \
    --output-excel ./precision_result_output.xlsx
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--baseline-dir` | `./dump_data_h20` | Baseline 数据目录 (H20 GPU) |
| `--test-dir` | `./dump_data_ascend` | Test 数据目录 (Ascend 950) |
| `--output-report` | 无（仅 stdout） | Markdown 报告输出路径 |
| `--output-charts-dir` | 自动 | 图表输出目录 |
| `--output-excel` | 自动 | Excel 报告输出路径 |

### 其他脚本

- **compare_layer_precision.py**：逐层逐算子（InputLN / Attn / PostLN / MLP）精度对比，需要按 `layer_*/` 子目录组织的数据
- **compv8.py**：分析 MoE 专家路由选择的差异（topk_ids 匹配率、分数边界分析）

## 输出说明

### Markdown 报告

包含 6 个章节：
1. **配置信息** — 对比目录、文件数、层范围
2. **按 Dump 顺序的完整比对结果** — 每层一个子表，列出所有 dump 点的精度指标
3. **按功能阶段汇总** — 按 Input LN / Attention / MoE Routing 等阶段统计
4. **按层汇总** — 每层的平均/最小 SNR 和 ULP
5. **质量评估** — 总体和各阶段的质量等级
6. **异常点检测** — 列出 SNR < 50 dB 或 within_1ulp < 95% 的 dump 点

### Excel 报告

按 `precision_result.xlsx` 模板格式，单 sheet 内按层分组，包含：
- H20 / Ascend 各自的 mean / min / max 统计
- 绝对误差 (abs error)、相对误差 (rel error)
- ULP 指标 (mean_ulp, within_1ulp%, within_2ulp%)

### 可视化图表

| 图表 | 说明 |
|------|------|
| `snr_by_layer.png` | 关键 dump 点的 SNR 随层变化折线图 |
| `within_1ulp_by_layer.png` | 关键 dump 点的 within_1ULP 随层变化折线图 |
| `snr_heatmap.png` | 全量 SNR 热力图 (层 x dump 点) |
| `error_accumulation.png` | 按执行顺序的误差累积图 |

## 精度指标说明

### ULP (Unit in Last Place)

衡量两个浮点数在 BF16 精度下的差异。BF16 有 7 位尾数，ULP = 2^(floor(log2(|x|)) - 7)。

| 指标 | 含义 |
|------|------|
| mean_ulp | 平均 ULP 偏差 |
| within_1ulp% | 差异在 1 ULP 以内的元素占比 |
| within_2ulp% | 差异在 2 ULP 以内的元素占比 |

### SNR (Signal-to-Noise Ratio)

信噪比，单位 dB。SNR = 10 * log10(signal_power / noise_power)。

### 质量阈值

| 等级 | SNR | within_1ulp |
|------|-----|-------------|
| Excellent | >= 60 dB | >= 99% |
| Good | >= 50 dB | >= 95% |
| Needs Investigation | < 50 dB | < 95% |

## Dump 文件命名规则

每个 dump 文件命名格式为 `layer{idx}_{dump_name}.bin`，其中 `idx` 为层索引（0~39）。

### 每层关键 dump 点（按执行顺序）

**Attention 子层：**
`input` → `after_input_layernorm` → `qkv_proj_input` / `qkv_proj_weight` → `after_qkv_proj` → `after_self_attn` → `o_proj_weight` → `after_o_proj` → `after_attn_add_residual` → `after_post_attention_layernorm`

**Dense FFN 子层（仅 layer 0）：**
`gate_up_proj_weight` → `after_ffn_gate_up_proj` → `after_ffn_act_fn` → `down_proj_weight` → `after_mlp_layer`

**MoE 子层（layer 1~39）：**
`experts_input` → `after_router` → `grouped_topk_*`（10 个路由选择 dump） → `fused_topk_*`（2 个） → `after_experts_shared` / `after_experts_fused` → `after_mlp_layer`

## 数据格式

Dump 文件扩展名为 `.bin`，实际使用 `torch.save()` 序列化，加载方式：

```python
tensor = torch.load("layer0_input.bin", map_location="cpu", weights_only=True)
```

详细的 dump 点说明见 `dump_data_ascend_说明文档.md`。
