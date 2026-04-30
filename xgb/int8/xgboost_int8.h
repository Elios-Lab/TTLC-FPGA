#pragma once
#include <ap_int.h>

#define XGB_NUM_FEATURES 1500
#define XGB_NUM_TREES 100
#define XGB_TOTAL_NODES 11534
#define NUM_PARALLEL_TREES 20  // Adjust based on resources
#define XGB_MAX_DEPTH 16 // Adjust based on resources

#define SEQ_LEN  50
#define EMBED_DIM 30

extern const float INV_SCALE[XGB_NUM_FEATURES];

extern "C" void xgboost_kernel(const float* x, float* y);
