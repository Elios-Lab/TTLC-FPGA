#pragma once
// Quantized INT8 version of the transformer HLS model.
// Heavy matrix multiplications use INT8 weights and INT8 activations,
// with INT32 accumulation and float scales. Final outputs are dequantized
// back to float only at the very end.

#include <stdint.h>
#include "ap_int.h"

// Re-use the same model geometry as the float design.
constexpr int SEQ_LEN       = 50;
constexpr int EMBED_DIM     = 30;    // Input dimension (30 features)
constexpr int HEAD_SIZE     = 128;   // Attention head size
constexpr int NUM_HEADS     = 1;     // Single head attention
constexpr int FF_DIM        = 128;   // Feed-forward dimension
constexpr int NUM_BLOCKS    = 2;     // Number of transformer blocks
constexpr int MLP_UNITS     = 160;   // MLP head units
constexpr int FINAL_NEURONS = 1;     // Regression output

// Quantized integer types
typedef ap_int<8>   qint8_t;
typedef ap_int<16>  qint16_t;
typedef ap_int<32>  qint32_t;

// -------------------- INT8 Weight containers --------------------

// Positional embedding (already pre-projected to EMBED_DIM).
// Stored and consumed in the same activation scale domain as x.
struct PositionalEmbeddingWeightsInt8 {
    qint8_t table[SEQ_LEN][EMBED_DIM];
    float   scale[EMBED_DIM];   // per-channel
};


// One encoder block worth of weights
struct EncoderBlockWeightsInt8 {
    // Self-attention projections
    qint8_t W_q[EMBED_DIM][HEAD_SIZE];
    float   W_q_scale[HEAD_SIZE];
    float   b_q[HEAD_SIZE];

    qint8_t W_k[EMBED_DIM][HEAD_SIZE];
    float   W_k_scale[HEAD_SIZE];
    float   b_k[HEAD_SIZE];

    qint8_t W_v[EMBED_DIM][HEAD_SIZE];
    float   W_v_scale[HEAD_SIZE];
    float   b_v[HEAD_SIZE];

    // Output projection
    qint8_t W_o[HEAD_SIZE][EMBED_DIM];
    float   W_o_scale[EMBED_DIM];
    float   b_o[EMBED_DIM];

    // LayerNorm 1 (kept in float)
    float ln1_gamma[EMBED_DIM];
    float ln1_beta[EMBED_DIM];

    // Feed-forward (FFN)
    qint8_t W_ff1[EMBED_DIM][FF_DIM];
    float   W_ff1_scale[FF_DIM];
    float   b_ff1[FF_DIM];

    qint8_t W_ff2[FF_DIM][EMBED_DIM];
    float   W_ff2_scale[EMBED_DIM];
    float   b_ff2[EMBED_DIM];

    // LayerNorm 2 (kept in float)
    float ln2_gamma[EMBED_DIM];
    float ln2_beta[EMBED_DIM];
};

// Final MLP head
struct MLPWeightsInt8 {
    qint8_t W1[EMBED_DIM][MLP_UNITS];
    float   W1_scale[MLP_UNITS];
    float   b1[MLP_UNITS];

    qint8_t W2[MLP_UNITS][FINAL_NEURONS];
    float   W2_scale[FINAL_NEURONS];
    float   b2[FINAL_NEURONS];
};

// Full model
struct TransformerModelInt8 {
    PositionalEmbeddingWeightsInt8 pos_emb;
    EncoderBlockWeightsInt8        encoders[NUM_BLOCKS];
    MLPWeightsInt8                 mlp;
};

// Initialize model weights
void init_model_weights_int8(TransformerModelInt8 &model);

// Top-level HLS function
extern "C" void trans_int8(
    const float input[SEQ_LEN][EMBED_DIM],
    float output[FINAL_NEURONS]
);