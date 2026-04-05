#!/usr/bin/env python3
"""
逐层逐算子精度比对脚本 — 对比两组逐层数据的精度差异，生成报告和图表
=====================================================================
加载 collect_layer_outputs.py 采集的两组逐层数据（如 H20 vs Ascend），
对每层的每个算子计算 ULP 和 SNR 指标，生成 Markdown 报告和可视化图表。

图表包括:
  1. SNR 随层变化折线图（4 条线对应 4 个算子）
  2. within_1ulp 随层变化折线图
  3. 误差热力图（x=算子, y=层号, 颜色=SNR）
  4. mean_abs_err 随层变化折线图

用法:
  python compare_layer_precision.py \
      --baseline-dir ./data/h20_layerwise \
      --test-dir ./data/ascend_layerwise \
      --output-report ./reports/layerwise_h20_vs_ascend.md \
      --output-charts-dir ./reports/charts

环境要求: torch, numpy, matplotlib
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Import core metric functions from compare_precision.py
from compare_precision import (
    compute_bf16_ulp,
    compute_ulp_metrics,
    compute_snr_db,
    compute_abs_error_metrics,
)

# Must match the operators in collect_layer_outputs.py
OPERATOR_NAMES = [
    "input_layernorm",
    "self_attn",
    "post_attention_layernorm",
    "mlp",
]

# Friendly display names for charts
OPERATOR_DISPLAY_NAMES = {
    "input_layernorm": "Input LN",
    "self_attn": "Self Attn",
    "post_attention_layernorm": "Post-Attn LN",
    "mlp": "MLP",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OperatorMetrics:
    """Metrics for a single operator in a single layer."""
    layer_idx: int
    op_name: str
    data_type: str  # "input" or "output"
    # ULP
    mean_ulp: float
    max_ulp: float
    within_1ulp_pct: float
    within_2ulp_pct: float
    # SNR
    snr_db: float
    # Abs error
    mean_abs_err: float
    max_abs_err: float
    # Shape info
    baseline_shape: list
    test_shape: list


@dataclass
class LayerSummary:
    """Summary for one layer across all operators."""
    layer_idx: int
    operators: dict  # op_name -> {"input": OperatorMetrics, "output": OperatorMetrics}


@dataclass
class LayerwiseReport:
    """Full layerwise comparison report."""
    num_layers: int
    layers: list  # list of LayerSummary
    baseline_meta: Optional[dict]
    test_meta: Optional[dict]


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_layerwise(
    baseline_dir: str,
    test_dir: str,
) -> LayerwiseReport:
    """Compare layerwise data between two directories.

    Args:
        baseline_dir: Path to baseline layerwise data (e.g. h20_layerwise).
        test_dir: Path to test layerwise data (e.g. ascend_layerwise).

    Returns:
        LayerwiseReport with per-layer per-operator metrics.
    """
    # Load metadata
    baseline_meta = _load_metadata(baseline_dir)
    test_meta = _load_metadata(test_dir)

    # Determine number of layers
    num_layers = 0
    for d in sorted(Path(baseline_dir).iterdir()):
        if d.is_dir() and d.name.startswith("layer_"):
            num_layers += 1

    if num_layers == 0:
        raise FileNotFoundError(f"No layer_* directories found in {baseline_dir}")

    print(f"Found {num_layers} layers to compare")

    layers = []

    for layer_idx in range(num_layers):
        bl_layer_dir = os.path.join(baseline_dir, f"layer_{layer_idx:03d}")
        tl_layer_dir = os.path.join(test_dir, f"layer_{layer_idx:03d}")

        if not os.path.isdir(bl_layer_dir) or not os.path.isdir(tl_layer_dir):
            print(f"  WARNING: layer {layer_idx} missing in one of the dirs, skipping")
            continue

        op_results = {}

        for op_name in OPERATOR_NAMES:
            op_metrics = {}

            for data_type in ["input", "output"]:
                filename = f"{op_name}_{data_type}.pt"
                bl_file = os.path.join(bl_layer_dir, filename)
                tl_file = os.path.join(tl_layer_dir, filename)

                if not os.path.exists(bl_file) or not os.path.exists(tl_file):
                    continue

                bl_tensor = torch.load(
                    bl_file, map_location="cpu", weights_only=True
                ).float()
                tl_tensor = torch.load(
                    tl_file, map_location="cpu", weights_only=True
                ).float()

                orig_bl_shape = list(bl_tensor.shape)
                orig_tl_shape = list(tl_tensor.shape)

                # Handle shape mismatch
                if bl_tensor.shape != tl_tensor.shape:
                    print(
                        f"  WARNING: shape mismatch at layer {layer_idx}/{op_name}_{data_type}: "
                        f"{bl_tensor.shape} vs {tl_tensor.shape}"
                    )
                    min_tokens = min(bl_tensor.shape[0], tl_tensor.shape[0])
                    if bl_tensor.dim() > 1:
                        min_dim = min(bl_tensor.shape[1], tl_tensor.shape[1])
                        bl_tensor = bl_tensor[:min_tokens, :min_dim]
                        tl_tensor = tl_tensor[:min_tokens, :min_dim]
                    else:
                        bl_tensor = bl_tensor[:min_tokens]
                        tl_tensor = tl_tensor[:min_tokens]

                ulp = compute_ulp_metrics(bl_tensor, tl_tensor)
                snr = compute_snr_db(bl_tensor, tl_tensor)
                abs_err = compute_abs_error_metrics(bl_tensor, tl_tensor)

                op_metrics[data_type] = OperatorMetrics(
                    layer_idx=layer_idx,
                    op_name=op_name,
                    data_type=data_type,
                    mean_ulp=ulp["mean_ulp"],
                    max_ulp=ulp["max_ulp"],
                    within_1ulp_pct=ulp["within_1ulp_pct"],
                    within_2ulp_pct=ulp["within_2ulp_pct"],
                    snr_db=snr,
                    mean_abs_err=abs_err["mean_abs_err"],
                    max_abs_err=abs_err["max_abs_err"],
                    baseline_shape=orig_bl_shape,
                    test_shape=orig_tl_shape,
                )

            op_results[op_name] = op_metrics

        layers.append(LayerSummary(layer_idx=layer_idx, operators=op_results))

        if layer_idx % 10 == 0:
            print(f"  Processed layer {layer_idx}/{num_layers}")

    return LayerwiseReport(
        num_layers=num_layers,
        layers=layers,
        baseline_meta=baseline_meta,
        test_meta=test_meta,
    )


# ---------------------------------------------------------------------------
# Report generation (Markdown)
# ---------------------------------------------------------------------------

def _load_metadata(directory: str) -> Optional[dict]:
    meta_path = os.path.join(directory, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


def _fmt(val: float, fmt: str = ".4f") -> str:
    if math.isinf(val):
        return "inf"
    if math.isnan(val):
        return "NaN"
    return f"{val:{fmt}}"


def generate_markdown_report(report: LayerwiseReport, baseline_dir: str, test_dir: str) -> str:
    """Generate a Markdown report for layerwise comparison."""
    lines = []
    lines.append("# Layerwise Operator Precision Comparison Report\n")

    # Configuration
    lines.append("## Configuration\n")
    lines.append("| Item | Baseline | Test |")
    lines.append("|---|---|---|")
    lines.append(f"| Directory | `{baseline_dir}` | `{test_dir}` |")

    bl_meta = report.baseline_meta
    tl_meta = report.test_meta
    if bl_meta and tl_meta:
        for key, label in [
            ("device_info", "Device"),
            ("vllm_version", "vLLM Version"),
            ("model", "Model"),
            ("tp_size", "TP Size"),
            ("dtype", "Dtype"),
            ("prompt", "Prompt"),
            ("token_count", "Token Count"),
            ("num_decoder_layers", "Decoder Layers"),
            ("replace_input", "Replace Input"),
        ]:
            bl_val = bl_meta.get(key, "N/A")
            tl_val = tl_meta.get(key, "N/A")
            # Truncate long prompts
            if key == "prompt":
                bl_val = str(bl_val)[:60] + ("..." if len(str(bl_val)) > 60 else "")
                tl_val = str(tl_val)[:60] + ("..." if len(str(tl_val)) > 60 else "")
            lines.append(f"| {label} | {bl_val} | {tl_val} |")
    lines.append("")

    # Per-operator output comparison table (main focus)
    lines.append("## Per-Layer Operator Output Comparison\n")
    lines.append(
        "| Layer | "
        "InputLN SNR | InputLN w1ulp | "
        "Attn SNR | Attn w1ulp | "
        "PostLN SNR | PostLN w1ulp | "
        "MLP SNR | MLP w1ulp |"
    )
    lines.append("|" + "---|" * 9)

    for ls in report.layers:
        row = f"| {ls.layer_idx:3d} |"
        for op_name in OPERATOR_NAMES:
            op_data = ls.operators.get(op_name, {})
            out_m = op_data.get("output")
            if out_m:
                row += f" {_fmt(out_m.snr_db, '.2f')} | {_fmt(out_m.within_1ulp_pct, '.1f')}% |"
            else:
                row += " N/A | N/A |"
        lines.append(row)
    lines.append("")

    # Per-operator input comparison table
    lines.append("## Per-Layer Operator Input Comparison\n")
    lines.append(
        "| Layer | "
        "InputLN SNR | InputLN w1ulp | "
        "Attn SNR | Attn w1ulp | "
        "PostLN SNR | PostLN w1ulp | "
        "MLP SNR | MLP w1ulp |"
    )
    lines.append("|" + "---|" * 9)

    for ls in report.layers:
        row = f"| {ls.layer_idx:3d} |"
        for op_name in OPERATOR_NAMES:
            op_data = ls.operators.get(op_name, {})
            in_m = op_data.get("input")
            if in_m:
                row += f" {_fmt(in_m.snr_db, '.2f')} | {_fmt(in_m.within_1ulp_pct, '.1f')}% |"
            else:
                row += " N/A | N/A |"
        lines.append(row)
    lines.append("")

    # Summary statistics
    lines.append("## Summary Statistics\n")
    lines.append("### Output SNR by Operator (avg across layers)\n")
    lines.append("| Operator | Avg SNR (dB) | Min SNR (dB) | Avg within_1ulp | Min within_1ulp |")
    lines.append("|---|---|---|---|---|")

    for op_name in OPERATOR_NAMES:
        snrs = []
        w1ulps = []
        for ls in report.layers:
            out_m = ls.operators.get(op_name, {}).get("output")
            if out_m and math.isfinite(out_m.snr_db):
                snrs.append(out_m.snr_db)
                w1ulps.append(out_m.within_1ulp_pct)

        if snrs:
            avg_snr = sum(snrs) / len(snrs)
            min_snr = min(snrs)
            avg_w1 = sum(w1ulps) / len(w1ulps)
            min_w1 = min(w1ulps)
            lines.append(
                f"| {OPERATOR_DISPLAY_NAMES[op_name]} | "
                f"{_fmt(avg_snr, '.2f')} | {_fmt(min_snr, '.2f')} | "
                f"{_fmt(avg_w1, '.2f')}% | {_fmt(min_w1, '.2f')}% |"
            )
        else:
            lines.append(f"| {OPERATOR_DISPLAY_NAMES[op_name]} | N/A | N/A | N/A | N/A |")
    lines.append("")

    # Quality assessment
    lines.append("## Quality Assessment\n")
    for op_name in OPERATOR_NAMES:
        snrs = []
        for ls in report.layers:
            out_m = ls.operators.get(op_name, {}).get("output")
            if out_m and math.isfinite(out_m.snr_db):
                snrs.append(out_m.snr_db)
        if snrs:
            avg = sum(snrs) / len(snrs)
            if avg >= 60:
                status = "Excellent"
            elif avg >= 50:
                status = "Good"
            else:
                status = "Needs investigation"
            lines.append(f"- **{OPERATOR_DISPLAY_NAMES[op_name]}**: {status} (avg SNR = {avg:.2f} dB)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_charts(report: LayerwiseReport, output_dir: str) -> list:
    """Generate precision analysis charts.

    Returns:
        List of saved chart file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("WARNING: matplotlib not available, skipping chart generation")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    num_layers = len(report.layers)
    layer_indices = [ls.layer_idx for ls in report.layers]

    # Collect data for charts
    # output metrics: {op_name: {"snr": [...], "w1ulp": [...], "mean_abs_err": [...]}}
    output_data = {op: {"snr": [], "w1ulp": [], "mean_abs_err": []} for op in OPERATOR_NAMES}
    input_data = {op: {"snr": [], "w1ulp": [], "mean_abs_err": []} for op in OPERATOR_NAMES}

    for ls in report.layers:
        for op_name in OPERATOR_NAMES:
            op = ls.operators.get(op_name, {})
            for data_type, data_dict in [("output", output_data), ("input", input_data)]:
                m = op.get(data_type)
                if m:
                    data_dict[op_name]["snr"].append(m.snr_db if math.isfinite(m.snr_db) else 100.0)
                    data_dict[op_name]["w1ulp"].append(m.within_1ulp_pct)
                    data_dict[op_name]["mean_abs_err"].append(m.mean_abs_err)
                else:
                    data_dict[op_name]["snr"].append(float("nan"))
                    data_dict[op_name]["w1ulp"].append(float("nan"))
                    data_dict[op_name]["mean_abs_err"].append(float("nan"))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    display_names = [OPERATOR_DISPLAY_NAMES[op] for op in OPERATOR_NAMES]

    # ── Chart 1: Output SNR by Layer ──
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, op_name in enumerate(OPERATOR_NAMES):
        ax.plot(
            layer_indices, output_data[op_name]["snr"],
            marker="o", markersize=3, linewidth=1.5,
            color=colors[i], label=display_names[i],
        )
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("SNR (dB)", fontsize=12)
    ax.set_title("Output SNR by Layer (Higher = Better)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color="green", linestyle="--", alpha=0.5, label="Good threshold (50 dB)")
    ax.axhline(y=60, color="blue", linestyle="--", alpha=0.3, label="Excellent threshold (60 dB)")
    fig.tight_layout()
    path = os.path.join(output_dir, "output_snr_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")

    # ── Chart 2: Output within_1ulp by Layer ──
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, op_name in enumerate(OPERATOR_NAMES):
        ax.plot(
            layer_indices, output_data[op_name]["w1ulp"],
            marker="o", markersize=3, linewidth=1.5,
            color=colors[i], label=display_names[i],
        )
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("within_1ulp (%)", fontsize=12)
    ax.set_title("Output within_1ULP by Layer (Higher = Better)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=95, color="green", linestyle="--", alpha=0.5, label="Good threshold (95%)")
    ax.axhline(y=99, color="blue", linestyle="--", alpha=0.3, label="Excellent threshold (99%)")
    ax.set_ylim(bottom=0, top=105)
    fig.tight_layout()
    path = os.path.join(output_dir, "output_w1ulp_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")

    # ── Chart 3: SNR Heatmap (Output) ──
    fig, ax = plt.subplots(figsize=(8, 14))
    heatmap_data = np.array([output_data[op]["snr"] for op in OPERATOR_NAMES]).T
    # Replace nan with 0 for display
    heatmap_data = np.nan_to_num(heatmap_data, nan=0.0)

    im = ax.imshow(
        heatmap_data, aspect="auto", cmap="RdYlGn",
        norm=Normalize(vmin=0, vmax=max(70, np.nanmax(heatmap_data))),
    )
    ax.set_xticks(range(len(OPERATOR_NAMES)))
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel("Layer Index", fontsize=12)
    ax.set_title("Output SNR Heatmap (dB)", fontsize=14)

    # Show layer ticks at intervals
    tick_interval = max(1, num_layers // 20)
    ax.set_yticks(range(0, num_layers, tick_interval))
    ax.set_yticklabels(range(0, num_layers, tick_interval))

    fig.colorbar(im, ax=ax, label="SNR (dB)", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(output_dir, "output_snr_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")

    # ── Chart 4: Mean Absolute Error by Layer ──
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, op_name in enumerate(OPERATOR_NAMES):
        values = output_data[op_name]["mean_abs_err"]
        ax.semilogy(
            layer_indices, values,
            marker="o", markersize=3, linewidth=1.5,
            color=colors[i], label=display_names[i],
        )
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (log scale)", fontsize=12)
    ax.set_title("Output Mean Absolute Error by Layer", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "output_mean_abs_err_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")

    # ── Chart 5: Input vs Output SNR comparison (combined) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, op_name in enumerate(OPERATOR_NAMES):
        ax = axes[idx // 2][idx % 2]
        in_snr = input_data[op_name]["snr"]
        out_snr = output_data[op_name]["snr"]
        ax.plot(layer_indices, in_snr, "b-o", markersize=2, linewidth=1, label="Input", alpha=0.8)
        ax.plot(layer_indices, out_snr, "r-s", markersize=2, linewidth=1, label="Output", alpha=0.8)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("SNR (dB)", fontsize=10)
        ax.set_title(f"{OPERATOR_DISPLAY_NAMES[op_name]} - Input vs Output SNR", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color="green", linestyle="--", alpha=0.3)

    fig.suptitle("Per-Operator Input vs Output SNR Comparison", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "input_vs_output_snr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")

    # ── Chart 6: Error accumulation summary ──
    fig, ax = plt.subplots(figsize=(14, 6))
    # Plot the output SNR at each "stage" going through each layer
    # Stage order: input_layernorm_out -> self_attn_out -> post_attn_ln_out -> mlp_out
    stages = []
    stage_labels = []
    for ls in report.layers:
        for op_name in OPERATOR_NAMES:
            out_m = ls.operators.get(op_name, {}).get("output")
            if out_m and math.isfinite(out_m.snr_db):
                stages.append(out_m.snr_db)
            else:
                stages.append(float("nan"))
            stage_labels.append(f"L{ls.layer_idx}")

    ax.plot(range(len(stages)), stages, linewidth=0.8, alpha=0.8, color="#1f77b4")
    ax.set_xlabel("Processing Stage (4 ops per layer)", fontsize=12)
    ax.set_ylabel("SNR (dB)", fontsize=12)
    ax.set_title("Error Accumulation Through All Layers and Operators", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color="green", linestyle="--", alpha=0.3)

    # Mark layer boundaries
    for i in range(0, len(stages), len(OPERATOR_NAMES)):
        if i > 0:
            ax.axvline(x=i, color="gray", linestyle=":", alpha=0.2)

    fig.tight_layout()
    path = os.path.join(output_dir, "error_accumulation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")

    return saved_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare per-layer per-operator precision between two runs"
    )
    parser.add_argument(
        "--baseline-dir", type=str, required=True,
        help="Directory with baseline layerwise data (e.g., H20)",
    )
    parser.add_argument(
        "--test-dir", type=str, required=True,
        help="Directory with test layerwise data (e.g., Ascend 950)",
    )
    parser.add_argument(
        "--output-report", "-o", type=str, default=None,
        help="Path to save markdown report (default: stdout only)",
    )
    parser.add_argument(
        "--output-charts-dir", type=str, default=None,
        help="Directory to save charts (default: same as report dir / no charts)",
    )
    args = parser.parse_args()

    print(f"Layerwise precision comparison:")
    print(f"  Baseline: {args.baseline_dir}")
    print(f"  Test:     {args.test_dir}")
    print()

    # Run comparison
    report = compare_layerwise(args.baseline_dir, args.test_dir)

    # Generate markdown report
    md_report = generate_markdown_report(report, args.baseline_dir, args.test_dir)

    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w") as f:
            f.write(md_report)
        print(f"\nReport saved to {args.output_report}")

    print(md_report)

    # Generate charts
    charts_dir = args.output_charts_dir
    if not charts_dir and args.output_report:
        charts_dir = os.path.join(
            os.path.dirname(args.output_report) or ".", "charts"
        )

    if charts_dir:
        print(f"\nGenerating charts to {charts_dir}...")
        saved = generate_charts(report, charts_dir)
        print(f"  Generated {len(saved)} charts")


if __name__ == "__main__":
    main()
