# TinyML Audio Digits Recognition on ARM Cortex-M4 (STM32)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Target: STM32F4](https://img.shields.io/badge/Target-STM32F401%20%2F%20STM32F411-orange.svg)](https://www.st.com/en/microcontrollers-microprocessors/stm32f4-series.html)
[![TinyML: INT8 PTQ](https://img.shields.io/badge/Quantization-INT8%20PTQ-purple.svg)](https://en.wikipedia.org/wiki/Quantization_(signal_processing))

An end-to-end TinyML audio digit recognition system (digits 0–9) trained on Google Speech Commands and deployed onto an ARM Cortex-M4 microcontroller (**STM32F401RE / STM32F411RE**).

This project is an independent practical implementation and edge deployment of the methodology presented in the Elsevier (2022) research paper:
> **"A novel framework for deployment of CNN models using post-training quantization on microcontroller"** *(Microprocessors and Microsystems, 2022)*

### 🎯 Research Implementation & Scope
- **Paper Reproduction on a More Constrained Target:** While the original paper targeted an **STM32F407** microcontroller (192 KB RAM, 1 MB Flash), this project adapts and deploys the framework onto the more constrained **STM32F401RE** (64 KB RAM).
- **Memory Footprint Optimization:** To fit within the 64 KB RAM constraint of the STM32F401 (where unbuffered intermediate activations in the paper required ~110 KB), layer-by-layer **Ping-Pong activation buffers** were employed, keeping peak SRAM under **18.8 KB**.
- **Accuracy Parity:** Preserves the paper's reported classification performance (~85% Top-1 accuracy) with zero structural drift.

---

## 📌 Key Highlights

- **Ultra-low Memory Footprint:** Compressed model weights from **161.4 KB (Float32)** down to **45.4 KB (INT8)** — a **3.6× reduction** in Flash/ROM storage.
- **Optimized SRAM Usage (18.8 KB Peak RAM):** Implements dynamic layer-by-layer **Ping-Pong activation buffers**, enabling full inference within microcontrollers with as little as 64 KB RAM.
- **Negligible Accuracy Degradation:** 
  - Float32 Baseline: **85.96%** Top-1 Accuracy (98.42% Top-5)
  - INT8 Quantized: **85.03%** Top-1 Accuracy (98.39% Top-5)
  - Accuracy drop is only **0.93%**, with McNemar's test p-value = **0.8173** (confirming no statistically significant degradation).
- **Zero-Dependency Bare-Metal C Inference:** Custom C inference engine (`nn_inference_linear.c` / `nn_inference.c`) written without heavy external runtimes (no TensorFlow Lite Micro overhead).
- **Complete Embedded Firmware:** Ready-to-flash STM32CubeIDE project featuring ADC DMA audio acquisition, Voice Activity Detection (VAD), 256-point Radix-2 FFT spectrogram extraction, and 7-segment display output.

---

## 🏗️ Architecture & Pipeline

```mermaid
flowchart LR
    A["Raw Audio (1s @ 16 kHz)"] --> B["ADC + DMA Double Buffer"]
    B --> C["VAD Thresholding"]
    C --> D["256-pt Radix-2 FFT"]
    D --> E["32x39 Spectrogram"]
    E --> F["3-Layer CNN (INT8)"]
    F --> G["7-Segment Display / UART"]
```

### 1. Audio Processing Front-End
- **Input:** 1 second audio sampled at 16 kHz via electret microphone / ADC.
- **Windowing:** Frame length of 256 samples with 50% overlap.
- **Spectral Feature Extraction:** Radix-2 Fast Fourier Transform (FFT) computing 32 frequency bins across 39 temporal frames to produce a `32 × 39 × 1` spectrogram tensor.

### 2. CNN Model Architecture (40,812 Parameters)
| Layer | Operation | Output Shape | Parameters |
| :--- | :--- | :--- | :--- |
| **Input** | Audio Spectrogram | `32 × 39 × 1` | 0 |
| **Conv2D_1** | 16 filters (3×3), ReLU, Stride 1 | `30 × 37 × 16` | 160 |
| **Pool_1** | MaxPooling (2×2), Stride 2 | `15 × 18 × 16` | 0 |
| **Conv2D_2** | 32 filters (3×3), ReLU, Stride 1 | `13 × 16 × 32` | 4,640 |
| **Pool_2** | MaxPooling (2×2), Stride 2 | `6 × 8 × 32` | 0 |
| **Conv2D_3** | 64 filters (3×3), ReLU, Stride 1 | `4 × 6 × 64` | 18,496 |
| **Pool_3** | MaxPooling (2×2), Stride 2 | `2 × 3 × 64` | 0 |
| **Flatten** | Reshape | `384` | 0 |
| **Dense** | Fully Connected + Softmax | `10` (Digits 0–9) | 3,850 |
| **Total** | | | **40,812 parameters** |

---

## 📊 Quantization & Benchmark Results

### Quantization Methodologies
1. **Power-of-Two (PoT) Bit-Shift PTQ:** Quantizes scaling factors to powers of two ($2^{-n}$), replacing hardware multiplications with simple bit-shift operations (`>> n`).
2. **True Integer Linear Quantization:** Standard affine integer quantization ($q = \text{round}(x/S) + Z$) using 32-bit integer accumulation and fixed-point multipliers.

### Rigorous Evaluation (3,005 Test Samples)
Evaluated using `verify_all_rigorous.py` against unseen spoken digits:

| Metric | Float32 Baseline | INT8 Linear PTQ | Difference |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **85.96%** (2583/3005) | **85.03%** (2555/3005) | -0.93% |
| **Top-3 Accuracy** | **96.08%** (2887/3005) | **95.82%** (2879/3005) | -0.26% |
| **Top-5 Accuracy** | **98.42%** (2957/3005) | **98.39%** (2956/3005) | -0.03% |
| **Flash Memory** | ~165 KB | **45.4 KB** | **-72.5% (3.6× smaller)** |
| **Peak RAM Usage** | > 100 KB (unbuffered) | **18.8 KB** (Ping-Pong) | **Fits STM32F401 (64KB)** |

#### Statistical Significance (McNemar's Test)
- Both models correct: 2,470 samples
- Both models incorrect: 387 samples
- Float32 correct / INT8 incorrect: 87 samples
- INT8 correct / Float32 incorrect: 61 samples
- **p-value = 0.8173** ($\gg 0.05$): There is no statistically significant degradation in predictive performance after INT8 post-training quantization.

---

## 📁 Repository Structure

```text
├── Audio_Digits_Recognitio.ipynb   # Complete pipeline notebook (Train, Quantize, Evaluate)
├── ptq_framework.py                # Calibration and quantization parameter calculation
├── power_of_two_ptq.py             # Power-of-Two bit-shift quantization engine
├── compare_quantization_metrics.py # Top-K accuracy comparison (Float vs INT8)
├── verify_all_rigorous.py          # Rigorous statistical evaluation & McNemar test
├── generate_linear_c_files.py      # Automated C source/header generator
├── generate_c_headers.py           # Header generator for PoT weights
├── switch_firmware_mode.py         # Utility to toggle Linear vs PoT in firmware
├── requirements.txt                # Python environment dependencies
│
├── export/                         # Deployment-ready C inference kernels & weights
│   ├── audio_processor.c/h         # 256-point Radix-2 FFT & spectrogram calculation
│   ├── nn_inference_linear.c/h     # Fixed-point integer C inference engine
│   ├── weights_data_linear.h       # INT8 model weights and bias arrays
│   ├── model_parameters_linear.h   # Layer shapes, scales, and zero-points
│   ├── test_sample.h               # Embedded verification test sample
│   └── quantized_model_int8.npz    # Exported quantized weights & parameters
│
└── STM CODE/
    └── Audio_Digits_Recognitio/    # STM32CubeIDE complete embedded project
        ├── Core/Src/               # main.c, ADC DMA handler, 7-segment display driver
        ├── Core/Inc/               # Hardware headers & peripheral configs
        └── Audio_Digits_Recognitio.ioc # STM32CubeMX device pinout configuration
```

---

## 🚀 Getting Started

### 1. Python Environment Setup
```bash
git clone https://github.com/mahmoudmamsour46/TinyML-Audio-Digits-Recognition-STM32.git
cd TinyML-Audio-Digits-Recognition-STM32

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Quantized Model Performance
Run the rigorous verification script over the test dataset:
```bash
python verify_all_rigorous.py
```

### 3. Generate C Firmware Headers
To regenerate the C headers and quantized weights from Python:
```bash
python generate_linear_c_files.py
```

### 4. Deploy to STM32 Microcontroller
1. Open **STM32CubeIDE**.
2. Select **File > Open Projects from File System...** and browse to `STM CODE/Audio_Digits_Recognitio`.
3. Connect your **STM32F401RE / STM32F411RE Nucleo board** via USB.
4. Build the project (`Ctrl + B`) and click **Run** (`Ctrl + F11`) to flash the firmware.
5. Speak digits (0–9) into the microphone — the recognized digit will immediately light up on the 7-segment display and transmit debug probabilities over USART.

---

## 🔬 Hardware Setup & Pinout

| Peripheral | STM32 Pin | Function |
| :--- | :--- | :--- |
| **Microphone ADC** | `PA0` (ADC1_IN0) | Continuous DMA audio sampling @ 16 kHz |
| **7-Segment Display** | `PB0` - `PB6` | Segments A through G |
| **UART Debug** | `PA2` (TX), `PA3` (RX) | Baud rate: 115200 (logs predictions & latency) |
| **Status LED** | `PA5` | Toggles upon active voice trigger (VAD) |

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author
**Mahmoud Mansour**  
- GitHub: [@mahmoudmamsour46](https://github.com/mahmoudmamsour46)
