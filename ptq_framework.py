"""
Implementation of the paper's Post-Training Quantization (PTQ) algorithms:
Post-Training Quantization (PTQ) for Microcontroller (Sailesh et al. 2022)
Algorithms 1 & 2
"""

import json
import sys
import io
import numpy as np
from tqdm import tqdm

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def pad_same(x, kh, kw):
    ph1 = (kh - 1) // 2
    ph2 = kh - 1 - ph1
    pw1 = (kw - 1) // 2
    pw2 = kw - 1 - pw1
    return np.pad(x, ((0, 0), (ph1, ph2), (pw1, pw2), (0, 0)), mode='constant')

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

def next_pow2_shift(max_val, n_bits=8):
    """
    Compute power-of-two bit shift such that:
    Scale = 2^shift
    max_val * 2^shift <= 2^(n_bits - 1) - 1 (127 for int8)
    """
    if max_val <= 1e-9:
        return 0
    # Target condition: 2^shift <= 127 / max_val
    # shift = floor(log2(127 / max_val))
    shift = int(np.floor(np.log2(127.0 / max_val)))
    return shift

# -------------------------------------------------------------
# Algorithm 1: Activation Calibration
# -------------------------------------------------------------
def run_calibration(calib_x, weights):
    """
    Algorithm 1: Model Calibration
    Forward pass over 500 calibration samples to record maximum activation (Amax) per layer
    """
    print("Running calibration (Algorithm 1) on", len(calib_x), "samples...")
    
    # 1. Input Max
    max_in = float(np.max(np.abs(calib_x)))
    
    # Conv1 + ReLU + Pool1
    a1_conv = conv2d_float(calib_x, weights['conv1_w'], weights['conv1_b'])
    a1_relu = np.maximum(0, a1_conv)
    max_a1 = float(np.max(a1_relu))
    p1 = maxpool2d_same_2x2(a1_relu)
    
    # Conv2 + ReLU + Pool2
    a2_conv = conv2d_float(p1, weights['conv2_w'], weights['conv2_b'])
    a2_relu = np.maximum(0, a2_conv)
    max_a2 = float(np.max(a2_relu))
    p2 = maxpool2d_same_2x2(a2_relu)
    
    # Conv3 + ReLU + Pool3
    a3_conv = conv2d_float(p2, weights['conv3_w'], weights['conv3_b'])
    a3_relu = np.maximum(0, a3_conv)
    max_a3 = float(np.max(a3_relu))
    p3 = maxpool2d_1x5(a3_relu)
    
    # FC
    p3_flat = p3.reshape(p3.shape[0], -1)
    fc_out = np.dot(p3_flat, weights['fc_w']) + weights['fc_b']
    max_fc = float(np.max(np.abs(fc_out)))
    
    amax_dict = {
        'in': max_in,
        'conv1': max_a1,
        'conv2': max_a2,
        'conv3': max_a3,
        'fc': max_fc
    }
    
    print("Maximum layer activations (Amax):")
    for k, v in amax_dict.items():
        print(f"  - {k:8s}: {v:.4f}")
        
    return amax_dict

# -------------------------------------------------------------
# Algorithm 2: Compute Quantization Parameters
# -------------------------------------------------------------
def compute_quantization_params(weights, amax_dict):
    """
    Algorithm 2 & Equations (8)-(12):
    Compute power-of-two scales, quantize weights to int8 and biases to int32
    """
    print("\nComputing quantization parameters (Algorithm 2)...")
    
    # Power-of-two scales for Activations (QA = 2^shift)
    shift_in = next_pow2_shift(amax_dict['in'])
    shift_a1 = next_pow2_shift(amax_dict['conv1'])
    shift_a2 = next_pow2_shift(amax_dict['conv2'])
    shift_a3 = next_pow2_shift(amax_dict['conv3'])
    shift_fc = next_pow2_shift(amax_dict['fc'])
    
    # Power-of-two scales for Weights (QW = 2^shift)
    w_names = ['conv1', 'conv2', 'conv3', 'fc']
    w_shifts = {}
    for name in w_names:
        w_max = float(np.max(np.abs(weights[name + '_w'])))
        w_shifts[name] = next_pow2_shift(w_max)
        print(f"  - Weight {name:8s}: Max = {w_max:.4f}, Shift = {w_shifts[name]}")
        
    print(f"  - Input activation : Shift = {shift_in}")
    print(f"  - Conv1 activation : Shift = {shift_a1}")
    print(f"  - Conv2 activation : Shift = {shift_a2}")
    print(f"  - Conv3 activation : Shift = {shift_a3}")
    print(f"  - FC activation    : Shift = {shift_fc}")
    
    # Quantize weights to int8
    q_weights = {}
    for name in w_names:
        # W_int8 = round(W * 2^shift)
        w_scaled = np.round(weights[name + '_w'] * (2.0 ** w_shifts[name]))
        q_weights[name + '_w'] = np.clip(w_scaled, -128, 127).astype(np.int8)
        
    # Compute right shift amounts to requantize from int32 accumulator to int8:
    # (Input_int8 * Weight_int8) scale is 2^(shift_in + shift_w)
    # Bias is scaled identically: b_int32 = round(b * 2^(shift_in + shift_w))
    # Desired output scale is 2^(shift_out)
    # Thus right shift amount: rs = (shift_in + shift_w) - shift_out
    
    rs_conv1 = (shift_in + w_shifts['conv1']) - shift_a1
    b_scale1 = 2.0 ** (shift_in + w_shifts['conv1'])
    q_weights['conv1_b'] = np.round(weights['conv1_b'] * b_scale1).astype(np.int32)
    
    rs_conv2 = (shift_a1 + w_shifts['conv2']) - shift_a2
    b_scale2 = 2.0 ** (shift_a1 + w_shifts['conv2'])
    q_weights['conv2_b'] = np.round(weights['conv2_b'] * b_scale2).astype(np.int32)
    
    rs_conv3 = (shift_a2 + w_shifts['conv3']) - shift_a3
    b_scale3 = 2.0 ** (shift_a2 + w_shifts['conv3'])
    q_weights['conv3_b'] = np.round(weights['conv3_b'] * b_scale3).astype(np.int32)
    
    # For final FC layer (compute logits directly)
    rs_fc = (shift_a3 + w_shifts['fc']) - shift_fc
    b_scalefc = 2.0 ** (shift_a3 + w_shifts['fc'])
    q_weights['fc_b'] = np.round(weights['fc_b'] * b_scalefc).astype(np.int32)
    
    shifts = {
        'shift_in': shift_in,
        'shift_a1': shift_a1,
        'shift_a2': shift_a2,
        'shift_a3': shift_a3,
        'shift_fc': shift_fc,
        'w_shifts': w_shifts,
        'rs_conv1': rs_conv1,
        'rs_conv2': rs_conv2,
        'rs_conv3': rs_conv3,
        'rs_fc': rs_fc
    }
    
    print("\nRequantization right-shift values (rs):")
    print(f"  - Conv1 Right Shift: {rs_conv1}")
    print(f"  - Conv2 Right Shift: {rs_conv2}")
    print(f"  - Conv3 Right Shift: {rs_conv3}")
    print(f"  - FC    Right Shift: {rs_fc}")
    
    return q_weights, shifts

# -------------------------------------------------------------
# Integer INT8 Inference Engine
# Bit-exact emulation of CMSIS-NN MCU kernel routines
# -------------------------------------------------------------
def conv2d_int8(x_int8, w_int8, b_int32, right_shift):
    N, H, W, C_in = x_int8.shape
    KH, KW, _, C_out = w_int8.shape
    x_pad = pad_same(x_int8, KH, KW)
    
    # 32-bit integer accumulation
    acc32 = np.zeros((N, H, W, C_out), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            patch = x_pad[:, i:i+KH, j:j+KW, :].astype(np.int32)
            # tensordot using int32
            acc32[:, i, j, :] = np.tensordot(patch, w_int8.astype(np.int32), axes=([1, 2, 3], [0, 1, 2])) + b_int32
            
    # ReLU activation on int32 (in-place)
    acc32 = np.maximum(0, acc32)
    
    # Re-quantization: right bit-shift to int8 with rounding
    if right_shift > 0:
        # Rounding offset: + (1 << (rs - 1))
        half = 1 << (right_shift - 1)
        acc32 = (acc32 + half) >> right_shift
    elif right_shift < 0:
        acc32 = acc32 << (-right_shift)
        
    out_int8 = np.clip(acc32, -128, 127).astype(np.int8)
    return out_int8

def forward_int8(x_float, q_weights, shifts):
    # 1. Input quantization to int8
    scale_in = 2.0 ** shifts['shift_in']
    x_int8 = np.clip(np.round(x_float * scale_in), -128, 127).astype(np.int8)
    
    # Layer 1: Conv1 (int8) + ReLU + Pool1
    c1 = conv2d_int8(x_int8, q_weights['conv1_w'], q_weights['conv1_b'], shifts['rs_conv1'])
    p1 = maxpool2d_same_2x2(c1)
    
    # Layer 2: Conv2 (int8) + ReLU + Pool2
    c2 = conv2d_int8(p1, q_weights['conv2_w'], q_weights['conv2_b'], shifts['rs_conv2'])
    p2 = maxpool2d_same_2x2(c2)
    
    # Layer 3: Conv3 (int8) + ReLU + Pool3
    c3 = conv2d_int8(p2, q_weights['conv3_w'], q_weights['conv3_b'], shifts['rs_conv3'])
    p3 = maxpool2d_1x5(c3)
    
    # Layer 4: FC
    p3_flat = p3.reshape(p3.shape[0], -1).astype(np.int32)
    fc_acc = np.dot(p3_flat, q_weights['fc_w'].astype(np.int32)) + q_weights['fc_b']
    
    # FC output (or direct argmax on 32-bit accumulator since uniform scaling preserves rank)
    return fc_acc

# -------------------------------------------------------------
# Evaluation and memory footprint metrics
# -------------------------------------------------------------
def evaluate_quantization():
    raw_weights = np.load('export/weights_float32.npz')
    calib = np.load('export/calibration.npz')
    test = np.load('export/test.npz')
    with open('export/info.json', 'r') as f:
        info = json.load(f)
        
    amax_dict = run_calibration(calib['x'], raw_weights)
    q_weights, shifts = compute_quantization_params(raw_weights, amax_dict)
    
    # Save quantized model artifacts
    save_dict = {k: v for k, v in q_weights.items()}
    for k, v in shifts.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                save_dict[f"{k}_{sub_k}"] = sub_v
        else:
            save_dict[k] = v
    np.savez('export/quantized_model_int8.npz', **save_dict)
    
    print("\nEvaluating quantized model (INT8) on test set (6,885 samples)...")
    x_test = test['x']
    y_test = test['y']
    
    # Batch evaluation to optimize execution time and RAM
    batch_size = 128
    preds = []
    for start in tqdm(range(0, len(x_test), batch_size), desc="INT8 Evaluation"):
        batch_x = x_test[start:start+batch_size]
        logits = forward_int8(batch_x, q_weights, shifts)
        batch_preds = np.argmax(logits, axis=1)
        preds.extend(batch_preds)
        
    preds = np.array(preds)
    acc_int8 = float(np.mean(preds == y_test))
    acc_fp32 = info['test_accuracy_float32']
    
    # Model size comparison
    # Float32 weights: 4 bytes each | INT8 weights: 1 byte each
    # Float32 biases: 4 bytes each | INT32 biases: 4 bytes each
    n_weights = sum(raw_weights[name + '_w'].size for name in ['conv1', 'conv2', 'conv3', 'fc'])
    n_biases = sum(raw_weights[name + '_b'].size for name in ['conv1', 'conv2', 'conv3', 'fc'])
    
    size_fp32_bytes = (n_weights + n_biases) * 4
    size_int8_bytes = (n_weights * 1) + (n_biases * 4)
    
    size_fp32_kb = size_fp32_bytes / 1024.0
    size_int8_kb = size_int8_bytes / 1024.0
    compression_ratio = size_fp32_bytes / size_int8_bytes
    
    print("\n" + "="*50)
    print("Final Benchmark Results (Accuracy & Memory Footprint):")
    print("="*50)
    print(f"Total Parameters               : {n_weights + n_biases:,}")
    print(f"Baseline Model Size (Float32)  : {size_fp32_kb:.2f} KB")
    print(f"Quantized Model Size (INT8)    : {size_int8_kb:.2f} KB")
    print(f"Compression Ratio              : {compression_ratio:.2f}x ({(1 - 1/compression_ratio)*100:.1f}% reduction)")
    print("-" * 50)
    print(f"Float32 Baseline Accuracy      : {acc_fp32 * 100:.2f}%")
    print(f"INT8 PTQ Accuracy              : {acc_int8 * 100:.2f}%")
    print(f"Accuracy Drop                  : {(acc_fp32 - acc_int8) * 100:+.2f}%")
    print("="*50)
    
    # Save summary report to text file
    with open('export/ptq_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy_float32': acc_fp32,
            'accuracy_int8': acc_int8,
            'accuracy_drop': acc_fp32 - acc_int8,
            'size_float32_kb': size_fp32_kb,
            'size_int8_kb': size_int8_kb,
            'compression_ratio': compression_ratio,
            'n_params': n_weights + n_biases
        }, f, indent=2)

if __name__ == '__main__':
    evaluate_quantization()
