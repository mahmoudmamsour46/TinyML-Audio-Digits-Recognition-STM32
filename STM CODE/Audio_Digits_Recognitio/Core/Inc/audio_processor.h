#ifndef __AUDIO_PROCESSOR_H__
#define __AUDIO_PROCESSOR_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Audio and spectrogram constants from paper
#define AUDIO_SAMPLE_RATE     8000
#define AUDIO_FRAME_LEN       240    // 30 ms frame
#define AUDIO_HOP_LEN         200    // 200-sample hop (40-sample overlap = 5 ms)
#define AUDIO_NUM_FRAMES      39     // 39 frames per 1-second clip
#define AUDIO_NUM_BINS        32     // 32 frequency bands
#define AUDIO_SPECTROGRAM_LEN (AUDIO_NUM_BINS * AUDIO_NUM_FRAMES) // 32 * 39 = 1248

// Initialize audio processing front-end
void audio_processor_init(void);

// Convert 8000 raw audio samples (1s) to quantized spectrogram (int8_t)
// audio_samples: 8000 raw ADC samples (0 to 255 range)
// out_spectrogram: 1248-byte output buffer (32x39) formatted for nn_predict
void compute_spectrogram(const uint8_t *audio_samples, int8_t *out_spectrogram);

// Voice Activity Detection (VAD):
// Evaluates audio energy to detect active speech vs silence
// Returns 1 if speech detected, 0 if silence
int detect_speech(const uint8_t *samples, int length, int threshold);

#ifdef __cplusplus
}
#endif

#endif // __AUDIO_PROCESSOR_H__
