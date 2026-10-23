"""
Generate C headers compatible with CMSIS-NN and STM32
based on Post-Training Quantization (PTQ) results
"""

import numpy as np

def generate_c_code():
    q_data = np.load('export/quantized_model_int8.npz')
    
    # 1. Generate weights_data.h
    with open('export/weights_data.h', 'w', encoding='utf-8') as f:
        f.write("/* Weights data for Microcontroller */\n")
        f.write("#ifndef __WEIGHTS_DATA_H__\n#define __WEIGHTS_DATA_H__\n\n#include <stdint.h>\n\n")
        
        # conv1_w
        w1 = q_data['conv1_w'].flatten()
        f.write(f"// Conv1 Weights: {len(w1)} elements\n")
        f.write("static const int8_t conv1_wt[" + str(len(w1)) + "] = {\n  " + ", ".join(map(str, w1)) + "\n};\n\n")
        
        # conv1_b
        b1 = q_data['conv1_b'].flatten()
        f.write(f"// Conv1 Biases: {len(b1)} elements\n")
        f.write("static const int32_t conv1_bias[" + str(len(b1)) + "] = {\n  " + ", ".join(map(str, b1)) + "\n};\n\n")
        
        # conv2_w
        w2 = q_data['conv2_w'].flatten()
        f.write(f"// Conv2 Weights: {len(w2)} elements\n")
        f.write("static const int8_t conv2_wt[" + str(len(w2)) + "] = {\n  " + ", ".join(map(str, w2)) + "\n};\n\n")
        
        # conv2_b
        b2 = q_data['conv2_b'].flatten()
        f.write(f"// Conv2 Biases: {len(b2)} elements\n")
        f.write("static const int32_t conv2_bias[" + str(len(b2)) + "] = {\n  " + ", ".join(map(str, b2)) + "\n};\n\n")
        
        # conv3_w
        w3 = q_data['conv3_w'].flatten()
        f.write(f"// Conv3 Weights: {len(w3)} elements\n")
        f.write("static const int8_t conv3_wt[" + str(len(w3)) + "] = {\n  " + ", ".join(map(str, w3)) + "\n};\n\n")
        
        # conv3_b
        b3 = q_data['conv3_b'].flatten()
        f.write(f"// Conv3 Biases: {len(b3)} elements\n")
        f.write("static const int32_t conv3_bias[" + str(len(b3)) + "] = {\n  " + ", ".join(map(str, b3)) + "\n};\n\n")
        
        # fc_w
        w_fc = q_data['fc_w'].flatten()
        f.write(f"// FC Weights: {len(w_fc)} elements\n")
        f.write("static const int8_t fc_wt[" + str(len(w_fc)) + "] = {\n  " + ", ".join(map(str, w_fc)) + "\n};\n\n")
        
        # fc_b
        b_fc = q_data['fc_b'].flatten()
        f.write(f"// FC Biases: {len(b_fc)} elements\n")
        f.write("static const int32_t fc_bias[" + str(len(b_fc)) + "] = {\n  " + ", ".join(map(str, b_fc)) + "\n};\n\n")
        
        f.write("#endif // __WEIGHTS_DATA_H__\n")
        
    # 2. Generate model_parameters.h
    with open('export/model_parameters.h', 'w', encoding='utf-8') as f:
        f.write("/* Quantization parameters and shifts */\n")
        f.write("#ifndef __MODEL_PARAMETERS_H__\n#define __MODEL_PARAMETERS_H__\n\n")
        
        f.write(f"#define INPUT_SCALE_SHIFT   {int(q_data['shift_in'])}\n")
        f.write(f"#define CONV1_OUT_RSHIFT    {int(q_data['rs_conv1'])}\n")
        f.write(f"#define CONV2_OUT_RSHIFT    {int(q_data['rs_conv2'])}\n")
        f.write(f"#define CONV3_OUT_RSHIFT    {int(q_data['rs_conv3'])}\n")
        f.write(f"#define FC_OUT_RSHIFT       {int(q_data['rs_fc'])}\n\n")
        
        f.write("#define INPUT_HEIGHT        32\n")
        f.write("#define INPUT_WIDTH         39\n")
        f.write("#define INPUT_CHANNELS      1\n")
        f.write("#define NUM_CLASSES         12\n\n")
        
        f.write("#endif // __MODEL_PARAMETERS_H__\n")
        
    print("C header files generated successfully in export directory:")
    print("  - export/weights_data.h")
    print("  - export/model_parameters.h")

if __name__ == '__main__':
    generate_c_code()
