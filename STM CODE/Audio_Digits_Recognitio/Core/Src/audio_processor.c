#include "audio_processor.h"
#include "model_parameters.h"
#include <math.h>
#include <string.h>

#define NFFT 256
#define PI 3.14159265358979323846f

// Constants derived from dataset in Python
#define NORM_MEAN  (-35.616352f)
#define NORM_STD   (24.129728f)
#define SCALE_IN   (36.3160f) // 2^5

static float twiddle_cos[NFFT / 2];
static float twiddle_sin[NFFT / 2];

void audio_processor_init(void) {
    for (int i = 0; i < NFFT / 2; i++) {
        twiddle_cos[i] = cosf(-2.0f * PI * (float)i / (float)NFFT);
        twiddle_sin[i] = sinf(-2.0f * PI * (float)i / (float)NFFT);
    }
}

// Fast Radix-2 Cooley-Tukey 256-point FFT
static void fft256(float *real, float *imag) {
    // 1. Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < NFFT - 1; i++) {
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
        int k = NFFT / 2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // 2. Butterfly computations
    for (int len = 2; len <= NFFT; len <<= 1) {
        int half = len >> 1;
        int step = NFFT / len;
        for (int i = 0; i < NFFT; i += len) {
            for (int k = 0; k < half; k++) {
                int idx = k * step;
                float cos_val = twiddle_cos[idx];
                float sin_val = twiddle_sin[idx];
                
                float u_r = real[i + k];
                float u_i = imag[i + k];
                
                float v_r = real[i + k + half] * cos_val - imag[i + k + half] * sin_val;
                float v_i = real[i + k + half] * sin_val + imag[i + k + half] * cos_val;
                
                real[i + k] = u_r + v_r;
                imag[i + k] = u_i + v_i;
                real[i + k + half] = u_r - v_r;
                imag[i + k + half] = u_i - v_i;
            }
        }
    }
}

// Compute full spectrogram from 8000 audio samples
void compute_spectrogram(const uint8_t *audio_samples, int8_t *out_spectrogram) {
    // 1. Compute DC bias mean of raw audio signal
    uint32_t sum = 0;
    for (int i = 0; i < 8000; i++) {
        sum += audio_samples[i];
    }
    float dc_bias = (float)sum / 8000.0f;

    float real[NFFT];
    float imag[NFFT];
    float mag[128];

    // 2. Process 39 frames (240 samples per frame, 200 sample hop)
    for (int f = 0; f < AUDIO_NUM_FRAMES; f++) {
        int start = f * AUDIO_HOP_LEN;

        // Zero-pad frame: 240 samples + 16 zeros to 256 points
        for (int i = 0; i < AUDIO_FRAME_LEN; i++) {
            // Normalize ADC sample from 0..255 to audio range [-1.0, 1.0]
            real[i] = ((float)audio_samples[start + i] - dc_bias) / 128.0f;
            imag[i] = 0.0f;
        }
        for (int i = AUDIO_FRAME_LEN; i < NFFT; i++) {
            real[i] = 0.0f;
            imag[i] = 0.0f;
        }

        // Compute FFT
        fft256(real, imag);

        // Compute magnitude for first 128 frequency bins
        for (int i = 0; i < 128; i++) {
            mag[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
        }

        // Average every 4 bins into 32 bands, compute log and normalize
        for (int b = 0; b < AUDIO_NUM_BINS; b++) {
            float avg_mag = (mag[b * 4] + mag[b * 4 + 1] + mag[b * 4 + 2] + mag[b * 4 + 3]) * 0.25f;
            float log_val = 20.0f * log10f(avg_mag + 1e-6f);
            
            // Feature Standardization
            float norm_val = (log_val - NORM_MEAN) / NORM_STD;
            
            // Quantize to int8 (scale factor x32 with rounding)
            int val_int = (int)roundf(norm_val * SCALE_IN);
            if (val_int > 127) val_int = 127;
            if (val_int < -128) val_int = -128;

            // Tensor layout: Height (32 bins) x Width (39 frames)
            out_spectrogram[b * AUDIO_NUM_FRAMES + f] = (int8_t)val_int;
        }
    }
}

// Voice Activity Detection (VAD)
// Measures average absolute deviation from DC center
int detect_speech(const uint8_t *samples, int length, int threshold) {
    if (length <= 0) return 0;
    
    // Calculate sample mean
    uint32_t sum = 0;
    for (int i = 0; i < length; i++) {
        sum += samples[i];
    }
    int mid = sum / length;
    
    // Calculate frame energy (mean absolute deviation)
    uint32_t energy = 0;
    for (int i = 0; i < length; i++) {
        int diff = (int)samples[i] - mid;
        if (diff < 0) diff = -diff;
        energy += diff;
    }
    
    uint32_t avg_energy = energy / length;
    return (avg_energy > (uint32_t)threshold) ? 1 : 0;
}
