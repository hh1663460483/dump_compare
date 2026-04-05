import os
import torch
import torch.nn.functional as F
import numpy as np


def load_tensor(file_path, cast_float32=True):
    try:
        tensor = torch.load(file_path, map_location='cpu', weights_only=False)
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
    except Exception:
        tensor_np = np.fromfile(file_path, dtype=np.float32)
        tensor = torch.from_numpy(tensor_np)
    if cast_float32:
        return tensor.float()
    return tensor


def analyze_topk_diff(router_logits_a, router_logits_b,
                      bias_a, bias_b,
                      topk_ids_a, topk_ids_b,
                      router_logits_a_raw, router_logits_b_raw,
                      bias_a_raw, bias_b_raw,
                      top_k=8, layer_idx=0):
    """Analyze why topk_ids differ between GPU and NPU."""
    print(f"\n{'='*70}")
    print(f"Layer {layer_idx} - TopK IDs Analysis")
    print(f"{'='*70}")

    # 0. Show original dtypes
    print(f"\n[0] Original dtypes:")
    print(f"  GPU router_logits: {router_logits_a_raw.dtype}, "
          f"NPU router_logits: {router_logits_b_raw.dtype}")
    print(f"  GPU bias: {bias_a_raw.dtype}, NPU bias: {bias_b_raw.dtype}")

    # 1. Check if inputs match
    router_diff = (router_logits_a - router_logits_b).abs()
    bias_diff = (bias_a - bias_b).abs()
    print(f"\n[1] Input Comparison (float32):")
    print(f"  router_logits max_diff={router_diff.max().item():.6e}, "
          f"mean_diff={router_diff.mean().item():.6e}")
    print(f"  bias max_diff={bias_diff.max().item():.6e}, "
          f"mean_diff={bias_diff.mean().item():.6e}")

    # 2. Compute scores in float32
    scores_a = router_logits_a.sigmoid() + bias_a.unsqueeze(0)
    scores_b = router_logits_b.sigmoid() + bias_b.unsqueeze(0)
    scores_diff = (scores_a - scores_b).abs()
    print(f"\n[2] Scores (sigmoid + bias) in float32:")
    print(f"  max_diff={scores_diff.max().item():.6e}, "
          f"mean_diff={scores_diff.mean().item():.6e}")

    # 3. Recompute topk_ids in float32
    recomputed_ids_a_f32 = torch.topk(scores_a, k=top_k, dim=-1, sorted=True)[1]
    recomputed_ids_b_f32 = torch.topk(scores_b, k=top_k, dim=-1, sorted=True)[1]

    # 3b. Recompute topk_ids in ORIGINAL dtype
    scores_a_native = router_logits_a_raw.sigmoid() + bias_a_raw.unsqueeze(0)
    scores_b_native = router_logits_b_raw.sigmoid() + bias_b_raw.unsqueeze(0)
    recomputed_ids_a_native = torch.topk(
        scores_a_native, k=top_k, dim=-1, sorted=True)[1]
    recomputed_ids_b_native = torch.topk(
        scores_b_native, k=top_k, dim=-1, sorted=True)[1]

    topk_ids_a_int = topk_ids_a.to(torch.long)
    topk_ids_b_int = topk_ids_b.to(torch.long)

    num_tokens = topk_ids_a_int.shape[0]
    print(f"\n[3] TopK IDs Match (element-wise, {num_tokens} tokens x {top_k} experts):")
    exact_match = (topk_ids_a_int == topk_ids_b_int)
    print(f"  GPU vs NPU topk_ids: {exact_match.sum().item()}/{exact_match.numel()} "
          f"({exact_match.float().mean().item()*100:.1f}%)")

    # Check set-wise match (same experts, different order)
    set_match_count = 0
    for t in range(num_tokens):
        set_a = set(topk_ids_a_int[t].tolist())
        set_b = set(topk_ids_b_int[t].tolist())
        if set_a == set_b:
            set_match_count += 1
    print(f"  GPU vs NPU topk_ids (set match): {set_match_count}/{num_tokens} "
          f"({set_match_count/num_tokens*100:.1f}%)")

    # 4. Check if recomputed matches dumped
    recomp_match_a_f32 = (recomputed_ids_a_f32.to(torch.long) == topk_ids_a_int)
    recomp_match_b_f32 = (recomputed_ids_b_f32.to(torch.long) == topk_ids_b_int)
    recomp_match_a_native = (recomputed_ids_a_native.to(torch.long) == topk_ids_a_int)
    recomp_match_b_native = (recomputed_ids_b_native.to(torch.long) == topk_ids_b_int)
    print(f"\n[4] Recomputed vs Dumped TopK IDs:")
    print(f"  float32 recompute:")
    print(f"    GPU recomputed vs dumped: {recomp_match_a_f32.sum().item()}/{recomp_match_a_f32.numel()} "
          f"({recomp_match_a_f32.float().mean().item()*100:.1f}%)")
    print(f"    NPU recomputed vs dumped: {recomp_match_b_f32.sum().item()}/{recomp_match_b_f32.numel()} "
          f"({recomp_match_b_f32.float().mean().item()*100:.1f}%)")
    print(f"  native dtype ({router_logits_a_raw.dtype} / {router_logits_b_raw.dtype}) recompute:")
    print(f"    GPU recomputed vs dumped: {recomp_match_a_native.sum().item()}/{recomp_match_a_native.numel()} "
          f"({recomp_match_a_native.float().mean().item()*100:.1f}%)")
    print(f"    NPU recomputed vs dumped: {recomp_match_b_native.sum().item()}/{recomp_match_b_native.numel()} "
          f"({recomp_match_b_native.float().mean().item()*100:.1f}%)")

    # 5. Analyze mismatched tokens in detail
    token_mismatch = ~exact_match.all(dim=-1)  # [num_tokens]
    mismatch_indices = token_mismatch.nonzero(as_tuple=True)[0]
    num_show = min(5, len(mismatch_indices))
    if num_show > 0:
        print(f"\n[5] Detailed Mismatch Analysis (showing first {num_show} of "
              f"{len(mismatch_indices)} mismatched tokens):")
        for i in range(num_show):
            t = mismatch_indices[i].item()
            ids_a = topk_ids_a_int[t].tolist()
            ids_b = topk_ids_b_int[t].tolist()
            # Get the scores for selected experts
            s_a = scores_a[t]
            s_b = scores_b[t]

            print(f"\n  Token {t}:")
            print(f"    GPU topk_ids: {ids_a}")
            print(f"    NPU topk_ids: {ids_b}")

            # Show scores for experts that differ
            only_in_a = set(ids_a) - set(ids_b)
            only_in_b = set(ids_b) - set(ids_a)
            if only_in_a or only_in_b:
                print(f"    Only in GPU: {only_in_a}")
                print(f"    Only in NPU: {only_in_b}")
                # Show scores for disputed experts
                disputed = list(only_in_a | only_in_b)
                disputed.sort()
                print(f"    Scores for disputed experts:")
                for e in disputed:
                    print(f"      Expert {e}: GPU_score={s_a[e].item():.8f}, "
                          f"NPU_score={s_b[e].item():.8f}, "
                          f"diff={abs(s_a[e].item()-s_b[e].item()):.8e}")

                # Show the boundary: score of the last selected vs first not selected
                sorted_scores_a = torch.sort(s_a, descending=True)
                kth_score_a = sorted_scores_a.values[top_k - 1].item()
                k1th_score_a = sorted_scores_a.values[top_k].item()
                print(f"    GPU: {top_k}th score={kth_score_a:.8f}, "
                      f"{top_k+1}th score={k1th_score_a:.8f}, "
                      f"gap={kth_score_a - k1th_score_a:.8e}")

                sorted_scores_b = torch.sort(s_b, descending=True)
                kth_score_b = sorted_scores_b.values[top_k - 1].item()
                k1th_score_b = sorted_scores_b.values[top_k].item()
                print(f"    NPU: {top_k}th score={kth_score_b:.8f}, "
                      f"{top_k+1}th score={k1th_score_b:.8f}, "
                      f"gap={kth_score_b - k1th_score_b:.8e}")
    else:
        print(f"\n[5] No mismatched tokens found!")

    # 6. Global score gap statistics at the top_k boundary
    print(f"\n[6] Score Gap at Top-{top_k} Boundary (all tokens):")
    sorted_scores_a_all = torch.sort(scores_a, dim=-1, descending=True).values
    gaps_a = sorted_scores_a_all[:, top_k - 1] - sorted_scores_a_all[:, top_k]
    print(f"  GPU: min_gap={gaps_a.min().item():.8e}, "
          f"mean_gap={gaps_a.mean().item():.8e}, "
          f"median_gap={gaps_a.median().item():.8e}")
    print(f"  GPU: tokens with gap < 1e-4: {(gaps_a < 1e-4).sum().item()}/{num_tokens}")
    print(f"  GPU: tokens with gap < 1e-3: {(gaps_a < 1e-3).sum().item()}/{num_tokens}")


def main():
    path_a = "/mnt/dump_data_h20"
    path_b = "/mnt/dump_data_ascend"
    num_layers = 40
    top_k = 8

    for layer_idx in range(num_layers):
        if layer_idx == 0:
            # Layer 0 doesn't have after_router / MoE experts
            continue

        router_file = f"layer{layer_idx}_after_router.bin"
        bias_file = f"layer{layer_idx}_experts_e_score_correction_bias.bin"
        topk_ids_file = f"layer{layer_idx}_fused_topk_ids.bin"

        files = [
            os.path.join(path_a, router_file),
            os.path.join(path_b, router_file),
            os.path.join(path_a, bias_file),
            os.path.join(path_b, bias_file),
            os.path.join(path_a, topk_ids_file),
            os.path.join(path_b, topk_ids_file),
        ]

        if not all(os.path.exists(f) for f in files):
            missing = [f for f in files if not os.path.exists(f)]
            print(f"[Skip] Layer {layer_idx} - missing: {missing}")
            continue

        router_logits_a = load_tensor(os.path.join(path_a, router_file))
        router_logits_b = load_tensor(os.path.join(path_b, router_file))
        bias_a = load_tensor(os.path.join(path_a, bias_file))
        bias_b = load_tensor(os.path.join(path_b, bias_file))
        topk_ids_a = load_tensor(os.path.join(path_a, topk_ids_file))
        topk_ids_b = load_tensor(os.path.join(path_b, topk_ids_file))

        # Also load in original dtype
        router_logits_a_raw = load_tensor(os.path.join(path_a, router_file), cast_float32=False)
        router_logits_b_raw = load_tensor(os.path.join(path_b, router_file), cast_float32=False)
        bias_a_raw = load_tensor(os.path.join(path_a, bias_file), cast_float32=False)
        bias_b_raw = load_tensor(os.path.join(path_b, bias_file), cast_float32=False)

        analyze_topk_diff(
            router_logits_a, router_logits_b,
            bias_a, bias_b,
            topk_ids_a, topk_ids_b,
            router_logits_a_raw, router_logits_b_raw,
            bias_a_raw, bias_b_raw,
            top_k=top_k,
            layer_idx=layer_idx,
        )


if __name__ == "__main__":
    main()
