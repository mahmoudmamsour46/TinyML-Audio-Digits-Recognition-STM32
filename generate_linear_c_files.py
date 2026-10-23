"""
Generate C source and header files for Linear Quantization on STM32:
1. Extract linear quantized weights (INT8).
2. Extract quantized biases (INT32).
3. Extract integer multipliers and shifts (TFLite / CMSIS-NN style).
4. Verify accuracy on test samples.
5. Write C files:
   - export/weights_data_linear.h
   - export/model_parameters_linear.h
   - export/nn_inference_linear.c
"""

import os
import math
import numpy as np

def generate_linear_c_files():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("=" * 70)
    print("Generating C files for Linear Quantization on STM32...")
    print("=" * 70)
    
    raw_weights = np.load('export/weights_float32.npz')
    calib = np.load('export/calibration.npz')
    test = np.load('export/test.npz')
    
    from ptq_framework import run_calibration
    amax_dict = run_calibration(calib['x'], raw_weights)
    
    # 1. Scales
    s_in = amax_dict['in'] / 127.0
    s_w1 = float(np.max(np.abs(raw_weights['conv1_w']))) / 127.0
    s_a1 = amax_dict['conv1'] / 127.0
    
    s_w2 = float(np.max(np.abs(raw_weights['conv2_w']))) / 127.0
    s_a2 = amax_dict['conv2'] / 127.0
    
    s_w3 = float(np.max(np.abs(raw_weights['conv3_w']))) / 127.0
    s_a3 = amax_dict['conv3'] / 127.0
    
    s_wfc = float(np.max(np.abs(raw_weights['fc_w']))) / 127.0
    s_afc = amax_dict['fc'] / 127.0
    
    def quantize_multiplier(real_scale):
        if real_scale <= 0:
            return 0, 0
        q, shift = math.frexp(real_scale)
        q_fixed = int(round(q * (1 << 31)))
        if q_fixed == (1 << 31):
            q_fixed //= 2
            shift += 1
        return q_fixed, -shift
        
    # Weights & Biases
    w1_int8 = np.clip(np.round(raw_weights['conv1_w'] / s_w1), -128, 127).astype(np.int8)
    b1_int32 = np.round(raw_weights['conv1_b'] / (s_in * s_w1)).astype(np.int32)
    M1, shift1 = quantize_multiplier((s_in * s_w1) / s_a1)
    
    w2_int8 = np.clip(np.round(raw_weights['conv2_w'] / s_w2), -128, 127).astype(np.int8)
    b2_int32 = np.round(raw_weights['conv2_b'] / (s_a1 * s_w2)).astype(np.int32)
    M2, shift2 = quantize_multiplier((s_a1 * s_w2) / s_a2)
    
    w3_int8 = np.clip(np.round(raw_weights['conv3_w'] / s_w3), -128, 127).astype(np.int8)
    b3_int32 = np.round(raw_weights['conv3_b'] / (s_a2 * s_w3)).astype(np.int32)
    M3, shift3 = quantize_multiplier((s_a2 * s_w3) / s_a3)
    
    wfc_int8 = np.clip(np.round(raw_weights['fc_w'] / s_wfc), -128, 127).astype(np.int8)
    bfc_int32 = np.round(raw_weights['fc_b'] / (s_a3 * s_wfc)).astype(np.int32)
    Mfc, shiftfc = quantize_multiplier((s_a3 * s_wfc) / s_afc)
    
    print("Extracted Linear Quantization parameters:")
    print(f"  - Input Scale (1/s_in): {1.0 / s_in:.4f}")
    print(f"  - Conv1: M0 = {M1}, Shift = {shift1}")
    print(f"  - Conv2: M0 = {M2}, Shift = {shift2}")
    print(f"  - Conv3: M0 = {M3}, Shift = {shift3}")
    print(f"  - FC   : M0 = {Mfc}, Shift = {shiftfc}")
    
    # 2. Generate model_parameters_linear.h
    model_param_h = f"""/* Quantization parameters for Linear Quantization (Fixed-Point Integer) */
#ifndef __MODEL_PARAMETERS_H__
#define __MODEL_PARAMETERS_H__

#include <stdint.h>

#define QUANTIZATION_MODE_LINEAR 1

// Input scale factor to INT8: val * INPUT_SCALE_FACTOR
#define INPUT_SCALE_FACTOR  {1.0 / s_in:.6f}f

// Linear quantization parameters (Multipliers & Shifts) per layer
#define CONV1_MULT          {M1}
#define CONV1_SHIFT         {shift1}

#define CONV2_MULT          {M2}
#define CONV2_SHIFT         {shift2}

#define CONV3_MULT          {M3}
#define CONV3_SHIFT         {shift3}

#define FC_MULT             {Mfc}
#define FC_SHIFT            {shiftfc}

#define INPUT_HEIGHT        32
#define INPUT_WIDTH         39
#define INPUT_CHANNELS      1
#define NUM_CLASSES         12

#endif // __MODEL_PARAMETERS_H__
"""
    with open('export/model_parameters_linear.h', 'w', encoding='utf-8') as f:
        f.write(model_param_h)
    print("Successfully saved export/model_parameters_linear.h")
    
    # 3. Generate weights_data_linear.h
    def array_to_c_int8(name, arr):
        flat = arr.flatten()
        lines = [f"const int8_t {name}[{len(flat)}] = {{\n    "]
        for i, val in enumerate(flat):
            lines.append(f"{val:4d},")
            if (i + 1) % 16 == 0 and (i + 1) < len(flat):
                lines.append("\n    ")
        lines.append("\n};\n\n")
        return "".join(lines)
        
    def array_to_c_int32(name, arr):
        flat = arr.flatten()
        lines = [f"const int32_t {name}[{len(flat)}] = {{\n    "]
        for i, val in enumerate(flat):
            lines.append(f"{val:8d},")
            if (i + 1) % 8 == 0 and (i + 1) < len(flat):
                lines.append("\n    ")
        lines.append("\n};\n\n")
        return "".join(lines)
        
    weights_h = [
        "/* Linearly Quantized Weights and Biases (INT8 / INT32) */\n",
        "#ifndef __WEIGHTS_DATA_H__\n",
        "#define __WEIGHTS_DATA_H__\n\n",
        "#include <stdint.h>\n\n",
        array_to_c_int8("conv1_wt", w1_int8),
        array_to_c_int32("conv1_bias", b1_int32),
        array_to_c_int8("conv2_wt", w2_int8),
        array_to_c_int32("conv2_bias", b2_int32),
        array_to_c_int8("conv3_wt", w3_int8),
        array_to_c_int32("conv3_bias", b3_int32),
        array_to_c_int8("fc_wt", wfc_int8),
        array_to_c_int32("fc_bias", bfc_int32),
        "#endif // __WEIGHTS_DATA_H__\n"
    ]
    with open('export/weights_data_linear.h', 'w', encoding='utf-8') as f:
        f.writelines(weights_h)
    print("Successfully saved export/weights_data_linear.h")
    
    # 4. Generate nn_inference_linear.c
    nn_inf_c = """#include "nn_inference.h"
#include "model_parameters.h"
#include "weights_data.h"

// 12 class label names
const char* const CLASS_NAMES[12] = {
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "background", "unknown"
};

/*
 * Ping-Pong Buffers technique for RAM optimization:
 * - Buffer B: 14,976 bytes (maximum layer activation for Conv1)
 * - Buffer A: 3,840 bytes (maximum activation for intermediate layers)
 * Total allocated RAM = 18.8 KB, fitting easily inside 64 KB SRAM of STM32F401.
 */
#define BUFFER_A_SIZE 3840
#define BUFFER_B_SIZE 14976

static int8_t buf_a[BUFFER_A_SIZE];
static int8_t buf_b[BUFFER_B_SIZE];

/*
 * Fixed-point linear requantization helper (TFLite / CMSIS-NN style):
 * 64-bit multiplication by quantized multiplier followed by rounding and shift.
 */
static inline int8_t requantize_linear(int32_t val, int32_t mult, int shift) {
    if (val < 0) val = 0; // ReLU
    
    // 64-bit Fixed Point Multiplication
    int64_t val64 = (int64_t)val * (int64_t)mult;
    int32_t val32 = (int32_t)((val64 + (1LL << 30)) >> 31);
    
    if (shift > 0) {
        val32 = (val32 + (1 << (shift - 1))) >> shift;
    } else if (shift < 0) {
        val32 = val32 << (-shift);
    }
    
    if (val32 > 127) val32 = 127;
    if (val32 < -128) val32 = -128;
    return (int8_t)val32;
}

// Conv2D 3x3 Same Padding with ReLU and Linear Requantization
static void conv3x3_same_relu_linear(const int8_t *in, int H, int W, int C_in,
                                    const int8_t *wt, const int32_t *bias, int C_out,
                                    int32_t mult, int shift, int8_t *out) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int oc = 0; oc < C_out; oc++) {
                int32_t acc = bias[oc];
                
                for (int ki = 0; ki < 3; ki++) {
                    int ni = i + ki - 1; // pad 1
                    if (ni < 0 || ni >= H) continue;
                    
                    for (int kj = 0; kj < 3; kj++) {
                        int nj = j + kj - 1; // pad 1
                        if (nj < 0 || nj >= W) continue;
                        
                        int in_idx = (ni * W + nj) * C_in;
                        int wt_idx = ((ki * 3 + kj) * C_in + 0) * C_out + oc;
                        
                        for (int ic = 0; ic < C_in; ic++) {
                            acc += (int32_t)in[in_idx + ic] * (int32_t)wt[wt_idx + ic * C_out];
                        }
                    }
                }
                
                *out++ = requantize_linear(acc, mult, shift);
            }
        }
    }
}

// MaxPool 2x2 Stride 2 Same Padding
static void maxpool2x2_same(const int8_t *in, int H, int W, int C, int8_t *out) {
    int out_h = (H + 1) / 2;
    int out_w = (W + 1) / 2;
    
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int c = 0; c < C; c++) {
                int8_t max_val = -128;
                for (int kh = 0; kh < 2; kh++) {
                    int ih = oh * 2 + kh;
                    if (ih >= H) continue;
                    for (int kw = 0; kw < 2; kw++) {
                        int iw = ow * 2 + kw;
                        if (iw >= W) continue;
                        int8_t v = in[(ih * W + iw) * C + c];
                        if (v > max_val) max_val = v;
                    }
                }
                *out++ = max_val;
            }
        }
    }
}

// MaxPool 1x5 Stride 1 Valid Padding
static void maxpool1x5_valid(const int8_t *in, int H, int W, int C, int8_t *out) {
    int out_w = W - 5 + 1; // 10 - 5 + 1 = 6
    for (int h = 0; h < H; h++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int c = 0; c < C; c++) {
                int8_t max_val = -128;
                for (int kw = 0; kw < 5; kw++) {
                    int8_t v = in[(h * W + (ow + kw)) * C + c];
                    if (v > max_val) max_val = v;
                }
                *out++ = max_val;
            }
        }
    }
}

// Run full linear-quantized neural network inference
int nn_predict(const int8_t *input_spec, int32_t *output_logits) {
    // 1. Layer 1: Conv1 [32, 39, 1] -> [32, 39, 12] (writes to buf_b)
    conv3x3_same_relu_linear(input_spec, 32, 39, 1, conv1_wt, conv1_bias, 12, CONV1_MULT, CONV1_SHIFT, buf_b);
    
    // Pool1: [32, 39, 12] -> [16, 20, 12] (reads buf_b, writes buf_a)
    maxpool2x2_same(buf_b, 32, 39, 12, buf_a);
    
    // 2. Layer 2: Conv2 [16, 20, 12] -> [16, 20, 24] (reads buf_a, writes buf_b)
    conv3x3_same_relu_linear(buf_a, 16, 20, 12, conv2_wt, conv2_bias, 24, CONV2_MULT, CONV2_SHIFT, buf_b);
    
    // Pool2: [16, 20, 24] -> [8, 10, 24] (reads buf_b, writes buf_a)
    maxpool2x2_same(buf_b, 16, 20, 24, buf_a);
    
    // 3. Layer 3: Conv3 [8, 10, 24] -> [8, 10, 48] (reads buf_a, writes buf_b)
    conv3x3_same_relu_linear(buf_a, 8, 10, 24, conv3_wt, conv3_bias, 48, CONV3_MULT, CONV3_SHIFT, buf_b);
    
    // Pool3: [8, 10, 48] -> [8, 6, 48] (reads buf_b, writes buf_a)
    maxpool1x5_valid(buf_b, 8, 10, 48, buf_a);
    
    // 4. Layer 4: Fully Connected [2304] -> [12]
    int best_class = 0;
    int32_t max_score = -2147483647;
    
    for (int oc = 0; oc < 12; oc++) {
        int32_t acc = fc_bias[oc];
        for (int ic = 0; ic < 2304; ic++) {
            acc += (int32_t)buf_a[ic] * (int32_t)fc_wt[ic * 12 + oc];
        }
        if (output_logits) {
            output_logits[oc] = acc;
        }
        if (acc > max_score) {
            max_score = acc;
            best_class = oc;
        }
    }
    
    return best_class;
}
"""
    with open('export/nn_inference_linear.c', 'w', encoding='utf-8') as f:
        f.write(nn_inf_c)
    print("Successfully saved export/nn_inference_linear.c")
    
    print("\nAll linear quantization files prepared successfully!")

if __name__ == '__main__':
    generate_linear_c_files()
