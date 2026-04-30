#pragma once
// FP32 Transformer for Vitis HLS
// Standard floating-point weights and activations

#include <stdint.h>

constexpr int SEQ_LEN       = 50;
constexpr int EMBED_DIM     = 30;    // Input dimension (30 features)
constexpr int HEAD_SIZE     = 128;   // Attention head size
constexpr int NUM_HEADS     = 1;     // Single head attention
constexpr int FF_DIM        = 128;   // Feed-forward dimension
constexpr int NUM_BLOCKS    = 2;     // Number of transformer blocks
constexpr int MLP_UNITS     = 160;   // MLP head units
constexpr float LN_EPSILON  = 1e-6f; // LayerNorm epsilon
constexpr int FINAL_NEURONS = 1;     // Regression output

// -------------------- Float Weight containers --------------------
struct PositionalEmbeddingWeights {
    float position_embedding[SEQ_LEN][EMBED_DIM];
};

struct LayerNormWeights {
    float gamma[EMBED_DIM];
    float beta[EMBED_DIM];
};

struct MHAWeights {
    float W_q[EMBED_DIM][HEAD_SIZE];
    float b_q[HEAD_SIZE];

    float W_k[EMBED_DIM][HEAD_SIZE];
    float b_k[HEAD_SIZE];

    float W_v[EMBED_DIM][HEAD_SIZE];
    float b_v[HEAD_SIZE];

    float W_o[HEAD_SIZE][EMBED_DIM];
    float b_o[EMBED_DIM];
};

struct FFNWeights {
    float W1[EMBED_DIM][FF_DIM];
    float b1[FF_DIM];

    float W2[FF_DIM][EMBED_DIM];
    float b2[EMBED_DIM];
};

struct EncoderBlockWeights {
    LayerNormWeights ln1;
    MHAWeights mha;
    LayerNormWeights ln2;
    FFNWeights ffn;
};

struct MLPWeights {
    float W1[EMBED_DIM][MLP_UNITS];
    float b1[MLP_UNITS];

    float W2[MLP_UNITS][FINAL_NEURONS];
    float b2[FINAL_NEURONS];
};

struct TransformerModel {
    PositionalEmbeddingWeights pos_emb;
    EncoderBlockWeights encoders[NUM_BLOCKS];
    MLPWeights mlp;
};

// Initialize model weights from trans_weights_fp32.h
void init_model_weights(TransformerModel &model);

// -------------------- HLS top --------------------
// Recommended as top function in Vitis HLS.
extern "C" void trans_fp32(
    const float input[SEQ_LEN][EMBED_DIM],
    float output[FINAL_NEURONS]
);
