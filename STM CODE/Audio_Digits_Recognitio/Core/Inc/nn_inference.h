#ifndef __NN_INFERENCE_H__
#define __NN_INFERENCE_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Run full quantized neural network inference
// Input: Quantized audio spectrogram (32 x 39 = 1248 bytes)
// Output: Logits/scores array for 12 classes
// Return: Predicted class index with highest score (0 to 11)
int nn_predict(const int8_t *input_spectrogram, int32_t *output_logits);

// Class label strings for debugging/display
extern const char* const CLASS_NAMES[12];

#ifdef __cplusplus
}
#endif

#endif // __NN_INFERENCE_H__
