import numpy as np

test_data = np.load('export/test.npz')
q_data = np.load('export/quantized_model_int8.npz')
scale_in = 2.0 ** float(q_data['shift_in'])

classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "background", "unknown"]

# Find first sample index for each digit class 0 to 9
samples = {}
for i in range(len(test_data['y'])):
    lbl = int(test_data['y'][i])
    if lbl not in samples and lbl < 10:
        samples[lbl] = i
    if len(samples) == 10:
        break

# Export sample for digit 'five' studied in paper
target_idx = samples[5] # Word 'five'
x_float = test_data['x'][target_idx]
y_true = int(test_data['y'][target_idx])
x_int8 = np.clip(np.round(x_float * scale_in), -128, 127).astype(np.int8).flatten()

with open('export/test_sample.h', 'w', encoding='utf-8') as f:
    f.write("/* Test sample of spoken digit 'FIVE' for MCU validation */\n")
    f.write("#ifndef __TEST_SAMPLE_H__\n#define __TEST_SAMPLE_H__\n\n#include <stdint.h>\n\n")
    f.write(f"// Ground Truth Class: {y_true} ({classes[y_true]})\n")
    f.write(f"#define EXPECTED_CLASS {y_true}\n")
    f.write(f"#define EXPECTED_CLASS_NAME \"{classes[y_true]}\"\n\n")
    f.write(f"static const int8_t sample_spectrogram[{len(x_int8)}] = {{\n  ")
    f.write(", ".join(map(str, x_int8)))
    f.write("\n};\n\n#endif // __TEST_SAMPLE_H__\n")

print(f"Exported sample for digit {y_true} ({classes[y_true]}) at index {target_idx}")
