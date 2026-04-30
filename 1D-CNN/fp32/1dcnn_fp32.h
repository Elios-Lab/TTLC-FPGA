
#pragma once
#include <stdint.h>

#define IN_TIMESTEPS 50
#define IN_CHANNELS  30
#define C1_OUT_CH    32
#define C1_K         5
#define C2_OUT_CH    32
#define C2_K         7
#define P_STRIDE     2
#define L1_LEN       IN_TIMESTEPS
#define L1P_LEN      (L1_LEN / P_STRIDE)
#define L2_LEN       L1P_LEN
#define L2P_LEN      (L2_LEN / P_STRIDE)
#define FLAT_SIZE    (L2P_LEN * C2_OUT_CH)
#define D1_UNITS     32
#define D2_UNITS     1

#ifdef __cplusplus
extern "C" {
#endif

// Top-level function for HLS
void cnn1d_fp32(const float input[IN_TIMESTEPS][IN_CHANNELS], float output[D2_UNITS]);

#ifdef __cplusplus
}
#endif
