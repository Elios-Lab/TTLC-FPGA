
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include "1dcnn_int8.h"
#include "1dcnn_weights_int8.h"
#include "quant_params.h"

#define USE_PER_CHANNEL_MS_L1
#define USE_PER_CHANNEL_MS_L2
#define USE_PER_CHANNEL_MS_D1
#define USE_PER_CHANNEL_MS_D2

// Saturating clamp to int8 range
static inline int8_t sat8(int32_t x) {
  if (x > 127) return 127;
  if (x < -128) return -128;
  return (int8_t)x;
}

// Requantize with rounding and optional ReLU, then clamp to int8
static inline int8_t requantize_and_clamp(int32_t acc, int32_t M, int S, int relu) {
  // 64-bit to avoid overflow on multiply
  int64_t prod = (int64_t)acc * (int64_t)M;
  if (S > 0) {
    int64_t rnd = (int64_t)1 << (S - 1);
    // Symmetric rounding to nearest
    prod = (prod >= 0) ? (prod + rnd) : (prod - rnd);
    prod >>= S;
  }
  int32_t res = (int32_t)prod;
  if (relu && res < 0) res = 0;
  return sat8(res);
}

static inline int8_t imax2(int8_t a, int8_t b) { return a > b ? a : b; }

void cnn1d_int8(const float input[IN_TIMESTEPS][IN_CHANNELS], float output[D2_UNITS]) {
  // Local integer activation buffers
  int8_t x0[IN_TIMESTEPS][IN_CHANNELS];
  int8_t c1[L1_LEN][C1_OUT_CH];
  int8_t p1[L1P_LEN][C1_OUT_CH];
  int8_t c2[L2_LEN][C2_OUT_CH];
  int8_t p2[L2P_LEN][C2_OUT_CH];
  int8_t flat[FLAT_SIZE];
  int8_t d1[D1_UNITS];

  // Storage binding hints
  #pragma HLS BIND_STORAGE variable=c1  type=ram_2p impl=bram
  #pragma HLS BIND_STORAGE variable=p1  type=ram_2p impl=bram
  #pragma HLS BIND_STORAGE variable=c2  type=ram_2p impl=bram
  #pragma HLS BIND_STORAGE variable=p2  type=ram_2p impl=bram
  #pragma HLS BIND_STORAGE variable=flat type=ram_2p impl=bram
  #pragma HLS BIND_STORAGE variable=d1  type=ram_2p impl=bram

  // Modest partition over channels to balance throughput/resources
  #pragma HLS ARRAY_PARTITION variable=c1 dim=2 cyclic factor=4
  #pragma HLS ARRAY_PARTITION variable=p1 dim=2 cyclic factor=4
  #pragma HLS ARRAY_PARTITION variable=c2 dim=2 cyclic factor=4
  #pragma HLS ARRAY_PARTITION variable=p2 dim=2 cyclic factor=4

  // 0) Quantize input to int8 (symmetric, zp=0)
  for (int t = 0; t < IN_TIMESTEPS; ++t) {
    for (int ic = 0; ic < IN_CHANNELS; ++ic) {
      float v = input[t][ic] / INPUT_SCALE;
      // Round to nearest and clamp
      int32_t q = (int32_t)lrintf(v);
      if (q > 127) q = 127; else if (q < -128) q = -128;
      x0[t][ic] = (int8_t)q;
    }
  }

  // 1) Conv1D #1 (k=5, SAME), ReLU; weights: kernel_int8 [K, InC, OutC]
  for (int t = 0; t < L1_LEN; ++t) {
    for (int oc = 0; oc < C1_OUT_CH; ++oc) {
      #pragma HLS PIPELINE II=1
      int32_t acc = BIAS1_ACC32[oc];
      for (int k = 0; k < C1_K; ++k) {
        int ti = t + k - (C1_K/2);
        if (ti < 0 || ti >= IN_TIMESTEPS) continue;
        for (int ic = 0; ic < IN_CHANNELS; ++ic) {
          int idx = ((k * IN_CHANNELS) + ic) * C1_OUT_CH + oc; // [K, InC, OutC]
          int16_t prod = (int16_t)kernel_int8[idx] * (int16_t)x0[ti][ic];
          acc += (int32_t)prod;
        }
      }
      #ifdef USE_PER_CHANNEL_MS_L1
        int8_t y = requantize_and_clamp(acc, C1_M_ARR[oc], C1_S_ARR[oc], /*relu=*/1);
      #else
        int8_t y = requantize_and_clamp(acc, C1_M, C1_S, /*relu=*/1);
      #endif
      c1[t][oc] = y;
    }
  }

  // 2) MaxPool1D(2)
  for (int t = 0; t < L1P_LEN; ++t) {
    int t0 = 2*t, t1 = 2*t + 1;
    for (int oc = 0; oc < C1_OUT_CH; ++oc) {
      #pragma HLS PIPELINE II=1
      p1[t][oc] = imax2(c1[t0][oc], c1[t1][oc]);
    }
  }

  // 3) Conv1D #2 (k=7, SAME), ReLU; weights: kernel_1_int8 [K, InC, OutC]
  for (int t = 0; t < L2_LEN; ++t) {
    for (int oc = 0; oc < C2_OUT_CH; ++oc) {
      #pragma HLS PIPELINE II=1
      int32_t acc = BIAS2_ACC32[oc];
      for (int k = 0; k < C2_K; ++k) {
        int ti = t + k - (C2_K/2);
        if (ti < 0 || ti >= L1P_LEN) continue;
        for (int ic = 0; ic < C1_OUT_CH; ++ic) {
          int idx = ((k * C1_OUT_CH) + ic) * C2_OUT_CH + oc; // [K, InC, OutC]
          int16_t prod = (int16_t)kernel_1_int8[idx] * (int16_t)p1[ti][ic];
          acc += (int32_t)prod;
        }
      }
      #ifdef USE_PER_CHANNEL_MS_L2
        int8_t y = requantize_and_clamp(acc, C2_M_ARR[oc], C2_S_ARR[oc], /*relu=*/1);
      #else
        int8_t y = requantize_and_clamp(acc, C2_M, C2_S, /*relu=*/1);
      #endif
      c2[t][oc] = y;
    }
  }

  // 4) MaxPool1D(2)
  for (int t = 0; t < L2P_LEN; ++t) {
    int t0 = 2*t, t1 = 2*t + 1;
    for (int oc = 0; oc < C2_OUT_CH; ++oc) {
      #pragma HLS PIPELINE II=1
      p2[t][oc] = imax2(c2[t0][oc], c2[t1][oc]);
    }
  }

  // 5) Flatten [L2P_LEN, C2_OUT_CH] -> [FLAT_SIZE]
  for (int t = 0; t < L2P_LEN; ++t) {
    for (int c = 0; c < C2_OUT_CH; ++c) {
      #pragma HLS PIPELINE II=1
      flat[t*C2_OUT_CH + c] = p2[t][c];
    }
  }

  // 6) Dense(384 -> 32), ReLU; weights: kernel_2_int8 [in, out]
  for (int o = 0; o < D1_UNITS; ++o) {
    int32_t acc = BIAS_D1_ACC32[o];
    for (int i = 0; i < FLAT_SIZE; ++i) {
      #pragma HLS PIPELINE II=1
      int16_t prod = (int16_t)kernel_2_int8[i*D1_UNITS + o] * (int16_t)flat[i];
      acc += (int32_t)prod;
    }
    #ifdef USE_PER_CHANNEL_MS_D1
      d1[o] = requantize_and_clamp(acc, D1_M_ARR[o], D1_S_ARR[o], /*relu=*/1);
    #else
      d1[o] = requantize_and_clamp(acc, D1_M, D1_S, /*relu=*/1);
    #endif
  }

  // 7) Dense(32 -> 1), Linear; weights: kernel_3_int8 [32]
  int32_t acc = BIAS_D2_ACC32[0];
  for (int i = 0; i < D1_UNITS; ++i) {
    #pragma HLS PIPELINE II=1
    int16_t prod = (int16_t)kernel_3_int8[i] * (int16_t)d1[i];
    acc += (int32_t)prod;
  }
  #ifdef USE_PER_CHANNEL_MS_D2
    int8_t y_q = requantize_and_clamp(acc, D2_M_ARR[0], D2_S_ARR[0], /*relu=*/0);
  #else
    int8_t y_q = requantize_and_clamp(acc, D2_M, D2_S, /*relu=*/0);
  #endif

  // Dequantize to float output using OUTPUT_SCALE
  output[0] = (float)y_q * OUTPUT_SCALE;
}
