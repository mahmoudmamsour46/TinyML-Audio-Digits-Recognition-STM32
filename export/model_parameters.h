/* Quantization parameters and shifts */
#ifndef __MODEL_PARAMETERS_H__
#define __MODEL_PARAMETERS_H__

#define INPUT_SCALE_SHIFT   5
#define CONV1_OUT_RSHIFT    8
#define CONV2_OUT_RSHIFT    8
#define CONV3_OUT_RSHIFT    6
#define FC_OUT_RSHIFT       11

#define INPUT_HEIGHT        32
#define INPUT_WIDTH         39
#define INPUT_CHANNELS      1
#define NUM_CLASSES         12

#endif // __MODEL_PARAMETERS_H__
