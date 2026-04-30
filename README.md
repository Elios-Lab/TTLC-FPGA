# Time-to-Lane-Change Estimation on FPGA

The code provides ready-to-synthesize Vitis HLS implementations of three machine-learning models for **Time-to-Lane-Change (TTLC)** estimation, each available in multiple numerical precisions to support resource-accuracy trade-off analysis on FPGA targets.

---

## Table of Contents

- [Task Description](#task-description)
- [Models and Precisions](#models-and-precisions)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Task Description

**Time-to-Lane-Change (TTLC)** estimation is a regression task in automated driving and advanced driver-assistance systems. Given a sliding window of observed vehicle kinematic features, the model predicts the time remaining (in seconds) until the ego vehicle performs a lane change maneuver.

Each input sample consists of a sequence of **50 timesteps × 30 features**, capturing longitudinal and lateral dynamics of the ego vehicle and surrounding traffic participants. The output is a single scalar representing the predicted TTLC value.

The dataset used for training and evaluation is the [**Lane Change Intention Recognition Dataset**](https://elios-lab.github.io/LC-Intention-Framework/).

---

## Models and Precisions

| Model | Precision | Description |
|---|---|---|
| 1D-CNN | FP32 | Single-precision floating-point |
| 1D-CNN | INT8 | Fully-integer with per-tensor/per-channel quantization |
| Transformer | FP32 | Single-precision floating-point |
| Transformer | INT8 | INT8 weights, INT32 accumulation, float scales |
| Transformer | Binary | XNOR-popcount binarized weights and activations with α-scaling |
| XGBoost | FP32 | Tree ensemble with float thresholds |
| XGBoost | INT8 | Quantized thresholds with inverse-scale dequantization |

### 1D-CNN Architecture

```
Input [50 × 30]
  → Conv1D(32 filters, k=5, same, ReLU) → MaxPool1D(stride=2)
  → Conv1D(32 filters, k=7, same, ReLU) → MaxPool1D(stride=2)
  → Flatten [384]
  → Dense(32, ReLU) → Dense(1, Linear)
  → Output scalar
```

### Transformer Architecture

```
Input [50 × 30]
  → Positional Embedding
  → 2 × Encoder Block [LayerNorm → MHA(head_size=128) → LayerNorm → FFN(dim=128)]
  → Global Average Pooling
  → MLP Head: Dense(160, ReLU) → Dense(1, Linear)
  → Output scalar
```

### XGBoost Architecture

- 100 gradient-boosted regression trees
- Maximum tree depth: 16
- Input features: 1500 (50 timesteps × 30 channels, flattened)
- Base score: 2.94

---

## Repository Structure

```
.
├── README.md
├── 1D-CNN/
│   ├── fp32/                        # 1D-CNN – FP32 implementation
│   │   ├── 1dcnn_fp32.h             # Architecture defines and top-function declaration
│   │   ├── 1dcnn_fp32.cpp           # HLS top-level implementation
│   │   ├── 1dcnn_fp32_tb.cpp        # Testbench
│   │   └── 1dcnn_weights_fp32.h     # Pre-trained FP32 weights
│   └── int8/                        # 1D-CNN – INT8 implementation
│       ├── 1dcnn_int8.h             # Architecture defines and quantization parameters
│       ├── 1dcnn_int8.cpp           # HLS top-level implementation
│       ├── 1dcnn_int8_tb.cpp        # Testbench
│       ├── 1dcnn_weights_int8.h     # Pre-trained INT8 weights
│       └── 1dcnn_quant_params.h     # Quantization scales and zero-points
├── Transformer/
│   ├── trans_fp32/                  # Transformer – FP32 implementation
│   │   ├── trans_fp32.h             # Architecture defines and top-function declaration
│   │   ├── trans_fp32.cpp           # HLS top-level implementation
│   │   ├── trans_fp32_tb.cpp        # Testbench
│   │   └── trans_weights_fp32.h     # Pre-trained FP32 weights (exported from TFLite)
│   ├── trans_int8/                  # Transformer – INT8 implementation
│   │   ├── trans_int8.h             # Architecture defines and quantization structures
│   │   ├── trans_int8.cpp           # HLS top-level implementation
│   │   ├── trans_int8_tb.cpp        # Testbench
│   │   ├── trans_weights_int8.h     # Pre-trained INT8 weights
│   │   └── trans_weights_fp32.h     # Auxiliary float weights (LayerNorm, biases)
│   └── trans_bin/                   # Transformer – Binary implementation
│       ├── trans_bin.h              # Architecture defines and binary weight structures
│       ├── trans_bin.cpp            # HLS top-level implementation (XNOR-popcount)
│       ├── trans_bin_tb.cpp         # Testbench
│       └── trans_weights_bin.h      # Pre-trained binarised weights with α-scales
└── XGBoost/
    ├── fp32/                        # XGBoost – FP32 implementation
    │   ├── xgboost_fp32.h           # Defines and top-function declaration
    │   ├── xgboost_fp32.cpp         # HLS top-level implementation
    │   └── xgboost_fp32_tb.cpp      # Testbench
    └── int8/                        # XGBoost – INT8 implementation
        ├── xgboost_int8.h           # Defines and top-function declaration
        ├── xgboost_int8.cpp         # HLS top-level implementation
        └── xgboost_int8_tb.cpp      # Testbench
```

---

## Requirements

### Software

| Tool | Version |
|---|---|
| AMD Vitis HLS (or Vitis Unified IDE) | 2022.2 or later |
| AMD Vivado (for bitstream generation) | 2022.2 or later |
| C++ compiler (for host-side testbenches) | C++11 or later |

> **Note:** The HLS implementations make use of Vitis HLS-specific headers (`ap_int.h`, `hls_math.h`) and synthesis pragmas (`#pragma HLS`). These are only available within the Vitis HLS environment and are not compatible with standard C++ compilers out of the box.

### Hardware

Any Xilinx/AMD FPGA with sufficient LUT, DSP, and BRAM resources. The models have been developed and tested targeting **Xilinx Zynq Ultrascale+** devices, but can be retargeted to other families via the Vitis HLS project settings.

---

## Usage

### 1. Create a Vitis HLS Project

1. Open **Vitis HLS** and select *Create Project*.
2. Set the project name and location.
3. Select the target FPGA part (e.g., `xczu7ev-ffvc1156-2-e`).
4. Set the clock period according to your timing requirements (e.g., 10 ns for 100 MHz).

### 2. Add Source Files

For each model and precision, add the corresponding files as follows:

- Add the `.cpp` implementation file as a **Design Source**.
- Add the `.h` header(s) and weight header(s) as **Header Files** (place them in the same source directory or add the directory to the include path).
- Add the `_tb.cpp` testbench file as a **Testbench Source**.

**Example – 1D-CNN FP32:**

| File | Role |
|---|---|
| `1D-CNN/fp32/1dcnn_fp32.cpp` | Design source |
| `1D-CNN/fp32/1dcnn_fp32.h` | Header |
| `1D-CNN/fp32/weights.h` | Weight header |
| `1D-CNN/fp32/1dcnn_fp32_tb.cpp` | Testbench |

### 3. Set the Top Function

In the Vitis HLS project settings, specify the **top function** for synthesis:

| Model / Precision | Top Function |
|---|---|
| 1D-CNN FP32 | `cnn1d_fp32` |
| 1D-CNN INT8 | `cnn1d_int8` |
| Transformer FP32 | `trans_fp32` |
| Transformer INT8 | `trans_int8` |
| Transformer Binary | `trans_bin` |
| XGBoost FP32 | `xgboost_fp32` |
| XGBoost INT8 | `xgboost_int8` |

### 4. Run C Simulation

Before synthesis, validate the implementation by running **C Simulation** (*Solution → Run C Simulation*). The testbench will execute inference on embedded test vectors and report the predicted output alongside the expected ground-truth value.

### 5. Run C Synthesis

Run **C Synthesis** (*Solution → Run C Synthesis*) to obtain the RTL implementation and the resource/timing report. The synthesis report provides estimates for:

- Latency (clock cycles)
- Initiation Interval (II)
- LUT, FF, DSP, and BRAM utilisation

### 6. Run Co-Simulation (Optional)

Run **C/RTL Co-Simulation** to verify that the synthesised RTL produces bit-accurate outputs matching the C simulation.

### 7. Export RTL / Implement on FPGA

Export the synthesized IP core (*Solution → Export RTL*) and integrate it into a Vivado block design for full implementation and bitstream generation.

---

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{ttlc_fpga_2025,
  title   = {[Title to be added upon publication]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  volume  = {[Volume]},
  number  = {[Number]},
  pages   = {[Pages]},
  doi     = {[DOI]}
}
```

This section will be updated upon publication of the associated article.
