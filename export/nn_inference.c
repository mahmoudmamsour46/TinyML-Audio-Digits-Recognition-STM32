#include "nn_inference.h"
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
 * Ping-pong buffer optimization: buffers alternate activations between consecutive layers:
 * - Buffer B: 14,976 bytes (maximum layer activation for Conv1)
 * - Buffer A: 3,840 bytes (maximum activation for intermediate layers)
 * Total allocated RAM = 18.8 KB, fitting comfortably in STM32F401 64 KB SRAM.
 */
#define BUFFER_A_SIZE 3840
#define BUFFER_B_SIZE 14976

static int8_t buf_a[BUFFER_A_SIZE];
static int8_t buf_b[BUFFER_B_SIZE];

// Integer round and right-shift helper
static inline int8_t requantize_shift(int32_t val, int shift) {
    if (val < 0) val = 0; // ReLU
    if (shift > 0) {
        val += (1 << (shift - 1)); // Rounding (+ 0.5)
        val >>= shift;
    }
    if (val > 127) val = 127;
    return (int8_t)val;
}

// Conv2D 3x3 Same Padding with ReLU and Power-of-Two Requantization
static void conv3x3_same_relu(const int8_t *in, int H, int W, int C_in,
                             const int8_t *wt, const int32_t *bias, int C_out,
                             int shift, int8_t *out) {
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
                
                *out++ = requantize_shift(acc, shift);
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

// Run complete end-to-end quantized inference
int nn_predict(const int8_t *input_spec, int32_t *output_logits) {
    // 1. Layer 1: Conv1 [32, 39, 1] -> [32, 39, 12] (writes to buf_b)
    conv3x3_same_relu(input_spec, 32, 39, 1, conv1_wt, conv1_bias, 12, CONV1_OUT_RSHIFT, buf_b);
    
    // Pool1: [32, 39, 12] -> [16, 20, 12] (reads buf_b, writes buf_a)
    maxpool2x2_same(buf_b, 32, 39, 12, buf_a);
    
    // 2. Layer 2: Conv2 [16, 20, 12] -> [16, 20, 24] (reads buf_a, writes buf_b)
    conv3x3_same_relu(buf_a, 16, 20, 12, conv2_wt, conv2_bias, 24, CONV2_OUT_RSHIFT, buf_b);
    
    // Pool2: [16, 20, 24] -> [8, 10, 24] (reads buf_b, writes buf_a)
    maxpool2x2_same(buf_b, 16, 20, 24, buf_a);
    
    // 3. Layer 3: Conv3 [8, 10, 24] -> [8, 10, 48] (reads buf_a, writes buf_b)
    conv3x3_same_relu(buf_a, 8, 10, 24, conv3_wt, conv3_bias, 48, CONV3_OUT_RSHIFT, buf_b);
    
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
