
#include "1dcnn_fp32.h"
#include "1dcnn_weights_fp32.h"

static inline float relu(float x) { return x > 0.f ? x : 0.f; }
static inline float fmax2(float a, float b) { return a > b ? a : b; }

void cnn1d_fp32(const float input[IN_TIMESTEPS][IN_CHANNELS], float output[D2_UNITS]) {
  // Buffers
  float c1[L1_LEN][C1_OUT_CH];
  float p1[L1P_LEN][C1_OUT_CH];
  float c2[L2_LEN][C2_OUT_CH];
  float p2[L2P_LEN][C2_OUT_CH];
  float flat[FLAT_SIZE];
  float d1[D1_UNITS];

#pragma HLS BIND_STORAGE variable=c1   type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=p1   type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=c2   type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=p2   type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=flat type=ram_2p impl=bram

#pragma HLS ARRAY_PARTITION variable=c1 dim=2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=p1 dim=2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=c2 dim=2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=p2 dim=2 cyclic factor=4

  // Conv1D #1 (k=5, same), ReLU
  for (int t = 0; t < L1_LEN; ++t) {
    for (int oc = 0; oc < C1_OUT_CH; ++oc) {
#pragma HLS PIPELINE II=1
      float acc = bias[oc];
      for (int k = 0; k < C1_K; ++k) {
        int ti = t + k - (C1_K/2);
        if (ti < 0 || ti >= IN_TIMESTEPS) continue;
        for (int ic = 0; ic < IN_CHANNELS; ++ic) {
          // kernel layout: [K, InC, OutC]
          int idx = ((k * IN_CHANNELS) + ic) * C1_OUT_CH + oc;
          acc += kernel[idx] * input[ti][ic];
        }
      }
      c1[t][oc] = relu(acc);
    }
  }

  // MaxPool1D(2, stride=2, valid)
  for (int t = 0; t < L1P_LEN; ++t) {
    int t0 = 2*t, t1 = 2*t + 1;
    for (int oc = 0; oc < C1_OUT_CH; ++oc) {
#pragma HLS PIPELINE II=1
      p1[t][oc] = fmax2(c1[t0][oc], c1[t1][oc]);
    }
  }

  // Conv1D #2 (k=7, same), ReLU
  for (int t = 0; t < L2_LEN; ++t) {
    for (int oc = 0; oc < C2_OUT_CH; ++oc) {
#pragma HLS PIPELINE II=1
      float acc = bias_1[oc];
      for (int k = 0; k < C2_K; ++k) {
        int ti = t + k - (C2_K/2);
        if (ti < 0 || ti >= L1P_LEN) continue;
        for (int ic = 0; ic < C1_OUT_CH; ++ic) {
          // kernel_1 layout: [K, InC, OutC]
          int idx = ((k * C1_OUT_CH) + ic) * C2_OUT_CH + oc;
          acc += kernel_1[idx] * p1[ti][ic];
        }
      }
      c2[t][oc] = relu(acc);
    }
  }

  // MaxPool1D(2, stride=2, valid)
  for (int t = 0; t < L2P_LEN; ++t) {
    int t0 = 2*t, t1 = 2*t + 1;
    for (int oc = 0; oc < C2_OUT_CH; ++oc) {
#pragma HLS PIPELINE II=1
      p2[t][oc] = fmax2(c2[t0][oc], c2[t1][oc]);
    }
  }

  // Flatten [L2P_LEN x C2_OUT_CH] -> [FLAT_SIZE]
  for (int t = 0; t < L2P_LEN; ++t) {
    for (int c = 0; c < C2_OUT_CH; ++c) {
#pragma HLS PIPELINE II=1
      flat[t*C2_OUT_CH + c] = p2[t][c];
    }
  }

  // Dense(384 -> 32), ReLU; kernel_2 layout: [in, out]
  for (int o = 0; o < D1_UNITS; ++o) {
    float acc = bias_2[o];
    for (int i = 0; i < FLAT_SIZE; ++i) {
#pragma HLS PIPELINE II=3
      acc += kernel_2[i*D1_UNITS + o] * flat[i];
    }
    d1[o] = relu(acc);
  }

  // Dense(32 -> 1), Linear; kernel_3 layout: [in]
  float y = bias_3[0];
  for (int i = 0; i < D1_UNITS; ++i) {
#pragma HLS PIPELINE II=3
    y += kernel_3[i] * d1[i];
  }
  output[0] = y;
}
