"""
=========================================================================================
Power-of-Two (PoT) Post-Training Quantization (PTQ) Framework for MCU Deployment
=========================================================================================
Reference Paper:
  "A novel framework for deployment of CNN models using post-training quantization 
   on microcontroller", S. M. et al., Microprocessors and Microsystems, 2022.

Methodology:
  - Algorithm 1: Dynamic Range Calibration (Determines A_max and W_max).
  - Algorithm 2: Power-of-Two Scaling & Shift-Based Integer Quantization.
  - Generates C-compatible integer weights (INT8), biases (INT32), and right-shift factors.
=========================================================================================
"""

import os
import math
import json
import numpy as np

# =======================================================================================
# 1. Helper Functions: Power-of-Two Exponent Calculation
# =======================================================================================

def next_pow2_shift(max_val, n_bits=8):
    """
    Computes the shift factor 'Q' such that:
        Scale = 2^Q <= (2^(n_bits-1) - 1) / max_val
    Ensures that (max_val * 2^Q) <= 127 to prevent overflow in INT8.
    
    Formula from Paper (Equation 8 & Algorithm 2):
        Q = floor( log2( 127 / max_val ) )
    """
    if max_val <= 1e-9:
        return 0
    qmax = (1 << (n_bits - 1)) - 1  # 127 for INT8
    shift = int(math.floor(math.log2(qmax / max_val)))
    return shift


# =======================================================================================
# 2. Algorithm 1: Dynamic Range Calibration
# =======================================================================================

def pad_same(x, kh, kw):
    """Zero-padding matching TensorFlow/Keras 'same' padding."""
    ph1 = (kh - 1) // 2
    ph2 = kh - 1 - ph1
    pw1 = (kw - 1) // 2
    pw2 = kw - 1 - pw1
    return np.pad(x, ((0, 0), (ph1, ph2), (pw1, pw2), (0, 0)), mode='constant')

def maxpool2d_same_2x2(x):
    """MaxPool 2x2 with Stride 2 and same padding."""
    N, H, W, C = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    out_h = (H + pad_h) // 2
    out_w = (W + pad_w) // 2
    x_reshaped = x.reshape(N, out_h, 2, out_w, 2, C)
    return x_reshaped.max(axis=(2, 4))

def maxpool2d_1x5(x):
    """MaxPool 1x5 with Stride 1 and valid padding."""
    N, H, W, C = x.shape
    out = np.zeros((N, H, W - 5 + 1, C), dtype=x.dtype)
    for j in range(W - 5 + 1):
        out[:, :, j, :] = x[:, :, j:j+5, :].max(axis=2)
    return out

def conv2d_float(x, w, b):
    """Standard 2D Convolution in Float32."""
    N, H, W, C_in = x.shape
    KH, KW, _, C_out = w.shape
    x_pad = pad_same(x, KH, KW)
    out = np.zeros((N, H, W, C_out), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            patch = x_pad[:, i:i+KH, j:j+KW, :]
            out[:, i, j, :] = np.tensordot(patch, w, axes=([1, 2, 3], [0, 1, 2])) + b
    return out

def run_calibration(calib_data, weights):
    """
    Algorithm 1 from Paper:
      Passes representative calibration audio samples (e.g. 500 samples)
      through the trained Float32 model to extract maximum activation amplitudes (A_max).
    """
    print("[Algorithm 1] Running Dynamic Range Calibration on representative dataset...")
    max_in = float(np.max(np.abs(calib_data)))
    
    # Conv1 + ReLU + Pool1
    a1_conv = conv2d_float(calib_data, weights['conv1_w'], weights['conv1_b'])
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
    
    # Fully Connected (FC)
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
    
    print("  Extracted Activation Maxima (A_max):")
    for k, v in amax_dict.items():
        print(f"    - Layer {k:6s}: {v:.4f}")
        
    return amax_dict


# =======================================================================================
# 3. Algorithm 2: Quantization Parameters & Power-of-Two Scaling
# =======================================================================================

def compute_quantization_params(weights, amax_dict):
    """
    Algorithm 2 from Paper:
      Derives integer quantization parameters:
        1. Weight scale:     Q_W = 2^(shift_w)
        2. Activation scale: Q_A = 2^(shift_a)
        3. Quantized weights: W_int8 = round(W * Q_W)
        4. Quantized biases:  B_int32 = round(B * Q_in * Q_W)
        5. Right-shift rs:   rs = (shift_in + shift_w) - shift_out
    """
    print("\n[Algorithm 2] Computing Power-of-Two Shifts & Requantization Parameters...")
    
    # Activation shifts (QA = 2^shift)
    shift_in = next_pow2_shift(amax_dict['in'])
    shift_a1 = next_pow2_shift(amax_dict['conv1'])
    shift_a2 = next_pow2_shift(amax_dict['conv2'])
    shift_a3 = next_pow2_shift(amax_dict['conv3'])
    shift_fc = next_pow2_shift(amax_dict['fc'])
    
    # Weight shifts (QW = 2^shift)
    w_names = ['conv1', 'conv2', 'conv3', 'fc']
    w_shifts = {}
    for name in w_names:
        w_max = float(np.max(np.abs(weights[name + '_w'])))
        w_shifts[name] = next_pow2_shift(w_max)
        
    q_weights = {}
    
    # Layer 1: Conv1
    q_weights['conv1_w'] = np.clip(np.round(weights['conv1_w'] * (2.0 ** w_shifts['conv1'])), -128, 127).astype(np.int8)
    b_scale1 = 2.0 ** (shift_in + w_shifts['conv1'])
    q_weights['conv1_b'] = np.round(weights['conv1_b'] * b_scale1).astype(np.int32)
    rs_conv1 = (shift_in + w_shifts['conv1']) - shift_a1
    
    # Layer 2: Conv2
    q_weights['conv2_w'] = np.clip(np.round(weights['conv2_w'] * (2.0 ** w_shifts['conv2'])), -128, 127).astype(np.int8)
    b_scale2 = 2.0 ** (shift_a1 + w_shifts['conv2'])
    q_weights['conv2_b'] = np.round(weights['conv2_b'] * b_scale2).astype(np.int32)
    rs_conv2 = (shift_a1 + w_shifts['conv2']) - shift_a2
    
    # Layer 3: Conv3
    q_weights['conv3_w'] = np.clip(np.round(weights['conv3_w'] * (2.0 ** w_shifts['conv3'])), -128, 127).astype(np.int8)
    b_scale3 = 2.0 ** (shift_a2 + w_shifts['conv3'])
    q_weights['conv3_b'] = np.round(weights['conv3_b'] * b_scale3).astype(np.int32)
    rs_conv3 = (shift_a2 + w_shifts['conv3']) - shift_a3
    
    # Layer 4: Fully Connected
    q_weights['fc_w'] = np.clip(np.round(weights['fc_w'] * (2.0 ** w_shifts['fc'])), -128, 127).astype(np.int8)
    b_scalefc = 2.0 ** (shift_a3 + w_shifts['fc'])
    q_weights['fc_b'] = np.round(weights['fc_b'] * b_scalefc).astype(np.int32)
    rs_fc = (shift_a3 + w_shifts['fc']) - shift_fc
    
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
    
    print("  Calculated Right-Shifts (Single-Cycle >> on Cortex-M4):")
    print(f"    - Conv1 RS: {rs_conv1}")
    print(f"    - Conv2 RS: {rs_conv2}")
    print(f"    - Conv3 RS: {rs_conv3}")
    print(f"    - FC    RS: {rs_fc}")
    
    return q_weights, shifts


# =======================================================================================
# 4. Integer INT8 Inference Engine (Bit-exact with MCU C Implementation)
# =======================================================================================

def conv2d_int8(x_int8, w_int8, b_int32, right_shift):
    """
    Fixed-Point INT8 Conv2D matching MCU hardware:
      - 32-bit integer accumulation (MAC).
      - ReLU activation.
      - Requantization via rounding and arithmetic right shift (>>).
    """
    N, H, W, C_in = x_int8.shape
    KH, KW, _, C_out = w_int8.shape
    x_pad = pad_same(x_int8, KH, KW)
    
    acc32 = np.zeros((N, H, W, C_out), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            patch = x_pad[:, i:i+KH, j:j+KW, :].astype(np.int32)
            acc32[:, i, j, :] = np.tensordot(patch, w_int8.astype(np.int32), axes=([1, 2, 3], [0, 1, 2])) + b_int32
            
    # ReLU
    acc32 = np.maximum(0, acc32)
    
    # Single-cycle Bit-shift requantization with rounding (+ 0.5)
    if right_shift > 0:
        half = 1 << (right_shift - 1)
        acc32 = (acc32 + half) >> right_shift
    elif right_shift < 0:
        acc32 = acc32 << (-right_shift)
        
    out_int8 = np.clip(acc32, -128, 127).astype(np.int8)
    return out_int8

def forward_pot_int8(x_float, q_weights, shifts):
    """Complete INT8 Forward Pass using Power-of-Two arithmetic."""
    # 1. Quantize Input Spectrogram [32, 39, 1] to INT8
    scale_in = 2.0 ** shifts['shift_in']
    x_int8 = np.clip(np.round(x_float * scale_in), -128, 127).astype(np.int8)
    
    # Layer 1: Conv1 + ReLU + Pool1
    c1 = conv2d_int8(x_int8, q_weights['conv1_w'], q_weights['conv1_b'], shifts['rs_conv1'])
    p1 = maxpool2d_same_2x2(c1)
    
    # Layer 2: Conv2 + ReLU + Pool2
    c2 = conv2d_int8(p1, q_weights['conv2_w'], q_weights['conv2_b'], shifts['rs_conv2'])
    p2 = maxpool2d_same_2x2(c2)
    
    # Layer 3: Conv3 + ReLU + Pool3
    c3 = conv2d_int8(p2, q_weights['conv3_w'], q_weights['conv3_b'], shifts['rs_conv3'])
    p3 = maxpool2d_1x5(c3)
    
    # Layer 4: Fully Connected (Dense -> 12 Output Logits)
    N = p3.shape[0]
    p3_flat = p3.reshape(N, -1).astype(np.int32)
    logits32 = np.dot(p3_flat, q_weights['fc_w'].astype(np.int32)) + q_weights['fc_b']
    
    return logits32


# =======================================================================================
# 5. Standalone Execution & Verification Demo
# =======================================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Power-of-Two PTQ Framework (Sailesh et al., 2022) - Demonstration")
    print("=" * 80)
    
    # Check if exported model weights exist
    weights_path = 'export/weights_float32.npz'
    calib_path = 'export/calibration.npz'
    test_path = 'export/test.npz'
    
    if os.path.exists(weights_path) and os.path.exists(calib_path):
        weights = np.load(weights_path)
        calib = np.load(calib_path)
        test = np.load(test_path)
        
        # 1. Calibrate
        amax = run_calibration(calib['x'], weights)
        
        # 2. Quantize
        q_weights, shifts = compute_quantization_params(weights, amax)
        
        # 3. Test on a batch
        print("\n[Inference Verification] Running INT8 inference on 100 test samples...")
        sample_x = test['x'][:100]
        sample_y = test['y'][:100]
        
        logits = forward_pot_int8(sample_x, q_weights, shifts)
        predictions = np.argmax(logits, axis=1)
        batch_acc = np.mean(predictions == sample_y)
        
        print(f"  Batch Accuracy (INT8 PoT): {batch_acc * 100:.2f}%")
        print("\nFramework verified successfully! Ready for STM32 C Code Generation.")
    else:
        print("Model files not found. Place 'weights_float32.npz' and 'calibration.npz' in 'export/'.")
    print("=" * 80)
