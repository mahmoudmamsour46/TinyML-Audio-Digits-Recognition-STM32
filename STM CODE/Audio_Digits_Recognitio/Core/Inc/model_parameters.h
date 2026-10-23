/* Quantization parameters for Linear Quantization (Fixed-Point Integer) */
#ifndef __MODEL_PARAMETERS_H__
#define __MODEL_PARAMETERS_H__

#include <stdint.h>

#define QUANTIZATION_MODE_LINEAR 1

// Input scale factor to INT8: val * INPUT_SCALE_FACTOR
#define INPUT_SCALE_FACTOR  36.315987f

// Linear quantization parameters (Multipliers & Shifts) per layer
#define CONV1_MULT          2086144783
#define CONV1_SHIFT         8

#define CONV2_MULT          1417338361
#define CONV2_SHIFT         7

#define CONV3_MULT          2075168700
#define CONV3_SHIFT         7

#define FC_MULT             1305291043
#define FC_SHIFT            10

#define INPUT_HEIGHT        32
#define INPUT_WIDTH         39
#define INPUT_CHANNELS      1
#define NUM_CLASSES         12

#endif // __MODEL_PARAMETERS_H__
