"""
Comprehensive Quantization Benchmark & Comparison:
1. Top-1, Top-3, Top-5 Accuracy for Baseline (Float32).
2. Top-1, Top-3, Top-5 Accuracy for Paper Method (Power-of-2 PTQ).
3. Top-1, Top-3, Top-5 Accuracy for Linear Quantization (Scale & Multiplier).
"""

import json
import numpy as np
from tqdm import tqdm

def top_k_accuracy(logits, targets, k):
    """Check whether the ground truth label is within the top-k predictions."""
    # Sort prediction indices in descending order
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    # Check if ground truth label is in top-k
    correct = [targets[i] in top_k_preds[i] for i in range(len(targets))]
    return float(np.mean(correct))

def pad_same(x, kh, kw):
    ph1 = (kh - 1) // 2
    ph2 = kh - 1 - ph1
    pw1 = (kw - 1) // 2
    pw2 = kw - 1 - pw1
    return np.pad(x, ((0, 0), (ph1, ph2), (pw1, pw2), (0, 0)), mode='constant')

def maxpool2d_same_2x2(x):
    N, H, W, C = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0,0), (0, pad_h), (0, pad_w), (0,0)), mode='constant')
    out_h = (H + pad_h) // 2
    out_w = (W + pad_w) // 2
    x_reshaped = x.reshape(N, out_h, 2, out_w, 2, C)
    return x_reshaped.max(axis=(2, 4))

def maxpool2d_1x5(x):
    N, H, W, C = x.shape
    out = np.zeros((N, H, W - 5 + 1, C), dtype=x.dtype)
    for j in range(W - 5 + 1):
        out[:, :, j, :] = x[:, :, j:j+5, :].max(axis=2)
    return out

# -------------------------------------------------------------------
# 1. Floating-Point 32 Baseline Inference
# -------------------------------------------------------------------
def conv2d_float(x, w, b):
    N, H, W, C_in = x.shape
    KH, KW, _, C_out = w.shape
    x_pad = pad_same(x, KH, KW)
    out = np.zeros((N, H, W, C_out), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            patch = x_pad[:, i:i+KH, j:j+KW, :]
            out[:, i, j, :] = np.tensordot(patch, w, axes=([1, 2, 3], [0, 1, 2])) + b
    return out

def forward_float32(x, weights):
    # Conv1 + ReLU + Pool1
    x = conv2d_float(x, weights['conv1_w'], weights['conv1_b'])
    x = np.maximum(0, x)
    x = maxpool2d_same_2x2(x)
    # Conv2 + ReLU + Pool2
    x = conv2d_float(x, weights['conv2_w'], weights['conv2_b'])
    x = np.maximum(0, x)
    x = maxpool2d_same_2x2(x)
    # Conv3 + ReLU + Pool3
    x = conv2d_float(x, weights['conv3_w'], weights['conv3_b'])
    x = np.maximum(0, x)
    x = maxpool2d_1x5(x)
    # FC
    x = x.reshape(x.shape[0], -1)
    x = np.dot(x, weights['fc_w']) + weights['fc_b']
    return x

# -------------------------------------------------------------------
# 2. Paper Method: Power-of-2 Fixed-Point PTQ
# -------------------------------------------------------------------
from ptq_framework import forward_int8

# -------------------------------------------------------------------
# 3. Alternative Method: Linear Quantization (Scale-Multiplier)
# S = max(|x|) / 127  =>  q = round(x / S)  =>  x_dequant = q * S
# -------------------------------------------------------------------
def linear_quantize(tensor, n_bits=8):
    qmax = (1 << (n_bits - 1)) - 1 # 127
    max_val = np.max(np.abs(tensor))
    scale = max_val / qmax if max_val > 0 else 1.0
    q_tensor = np.clip(np.round(tensor / scale), -qmax - 1, qmax).astype(np.int8)
    return q_tensor, scale

def forward_linear_quant(x, weights, amax_dict):
    # Layer-by-layer linear quantization using exact scale factors
    # 1. Input quantization
    s_in = amax_dict['in'] / 127.0
    x_q = np.clip(np.round(x / s_in), -128, 127).astype(np.float32) * s_in
    
    # Layer 1
    w1_q, s_w1 = linear_quantize(weights['conv1_w'])
    w1_deq = w1_q.astype(np.float32) * s_w1
    c1 = conv2d_float(x_q, w1_deq, weights['conv1_b'])
    c1 = np.maximum(0, c1)
    s_a1 = amax_dict['conv1'] / 127.0
    c1_q = np.clip(np.round(c1 / s_a1), -128, 127).astype(np.float32) * s_a1
    p1 = maxpool2d_same_2x2(c1_q)
    
    # Layer 2
    w2_q, s_w2 = linear_quantize(weights['conv2_w'])
    w2_deq = w2_q.astype(np.float32) * s_w2
    c2 = conv2d_float(p1, w2_deq, weights['conv2_b'])
    c2 = np.maximum(0, c2)
    s_a2 = amax_dict['conv2'] / 127.0
    c2_q = np.clip(np.round(c2 / s_a2), -128, 127).astype(np.float32) * s_a2
    p2 = maxpool2d_same_2x2(c2_q)
    
    # Layer 3
    w3_q, s_w3 = linear_quantize(weights['conv3_w'])
    w3_deq = w3_q.astype(np.float32) * s_w3
    c3 = conv2d_float(p2, w3_deq, weights['conv3_b'])
    c3 = np.maximum(0, c3)
    s_a3 = amax_dict['conv3'] / 127.0
    c3_q = np.clip(np.round(c3 / s_a3), -128, 127).astype(np.float32) * s_a3
    p3 = maxpool2d_1x5(c3_q)
    
    # Layer 4 (FC)
    w_fc_q, s_wfc = linear_quantize(weights['fc_w'])
    w_fc_deq = w_fc_q.astype(np.float32) * s_wfc
    p3_flat = p3.reshape(p3.shape[0], -1)
    fc_out = np.dot(p3_flat, w_fc_deq) + weights['fc_b']
    return fc_out

def run_evaluation():
    raw_weights = np.load('export/weights_float32.npz')
    calib = np.load('export/calibration.npz')
    test = np.load('export/test.npz')
    q_data = np.load('export/quantized_model_int8.npz')
    
    x_test = test['x']
    y_test = test['y']
    batch_size = 128
    
    # Prepare paper quantized weights
    q_weights_paper = {
        'conv1_w': q_data['conv1_w'], 'conv1_b': q_data['conv1_b'],
        'conv2_w': q_data['conv2_w'], 'conv2_b': q_data['conv2_b'],
        'conv3_w': q_data['conv3_w'], 'conv3_b': q_data['conv3_b'],
        'fc_w': q_data['fc_w'], 'fc_b': q_data['fc_b']
    }
    shifts_paper = {
        'shift_in': int(q_data['shift_in']),
        'rs_conv1': int(q_data['rs_conv1']),
        'rs_conv2': int(q_data['rs_conv2']),
        'rs_conv3': int(q_data['rs_conv3']),
        'rs_fc': int(q_data['rs_fc'])
    }
    
    from ptq_framework import run_calibration
    amax_dict = run_calibration(calib['x'], raw_weights)
    
    print("\n" + "="*60)
    print("Running comprehensive evaluation on 6,885 test samples...")
    print("="*60)
    
    logits_fp32 = []
    logits_paper_ptq = []
    logits_linear_quant = []
    
    for start in tqdm(range(0, len(x_test), batch_size), desc="Testing"):
        bx = x_test[start:start+batch_size]
        # 1. Float32
        out_fp = forward_float32(bx, raw_weights)
        logits_fp32.append(out_fp)
        # 2. Paper Power-of-2 PTQ
        out_paper = forward_int8(bx, q_weights_paper, shifts_paper)
        logits_paper_ptq.append(out_paper)
        # 3. Linear Quantization
        out_linear = forward_linear_quant(bx, raw_weights, amax_dict)
        logits_linear_quant.append(out_linear)
        
    logits_fp32 = np.vstack(logits_fp32)
    logits_paper_ptq = np.vstack(logits_paper_ptq)
    logits_linear_quant = np.vstack(logits_linear_quant)
    
    # Compute evaluation metrics
    fp32_top1 = top_k_accuracy(logits_fp32, y_test, 1)
    fp32_top3 = top_k_accuracy(logits_fp32, y_test, 3)
    fp32_top5 = top_k_accuracy(logits_fp32, y_test, 5)
    
    paper_top1 = top_k_accuracy(logits_paper_ptq, y_test, 1)
    paper_top3 = top_k_accuracy(logits_paper_ptq, y_test, 3)
    paper_top5 = top_k_accuracy(logits_paper_ptq, y_test, 5)
    
    linear_top1 = top_k_accuracy(logits_linear_quant, y_test, 1)
    linear_top3 = top_k_accuracy(logits_linear_quant, y_test, 3)
    linear_top5 = top_k_accuracy(logits_linear_quant, y_test, 5)
    
    results = {
        "Float32": {"Top1": fp32_top1, "Top3": fp32_top3, "Top5": fp32_top5},
        "Paper_PowerOf2_PTQ": {"Top1": paper_top1, "Top3": paper_top3, "Top5": paper_top5},
        "Linear_Quantization": {"Top1": linear_top1, "Top3": linear_top3, "Top5": linear_top5}
    }
    
    with open('export/comparison_topk_linear.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*70)
    print(f"{'Model / Method':<28} | {'Top-1 Acc':<12} | {'Top-3 Acc':<12} | {'Top-5 Acc':<12}")
    print("="*70)
    print(f"{'1. Baseline (Float32)':<28} | {fp32_top1*100:>8.2f}%   | {fp32_top3*100:>8.2f}%   | {fp32_top5*100:>8.2f}%")
    print(f"{'2. Paper PTQ (Power-of-2)':<28} | {paper_top1*100:>8.2f}%   | {paper_top3*100:>8.2f}%   | {paper_top5*100:>8.2f}%")
    print(f"{'3. Linear Quantization':<28} | {linear_top1*100:>8.2f}%   | {linear_top3*100:>8.2f}%   | {linear_top5*100:>8.2f}%")
    print("="*70)

if __name__ == '__main__':
    run_evaluation()
