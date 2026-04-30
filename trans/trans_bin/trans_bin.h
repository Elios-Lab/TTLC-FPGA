#pragma once
// XNOR-Popcount Binary Transformer for Vitis HLS
// Binarized activations & weights using XNOR-popcount
// Alpha scaling factors trained alongside weights

#include <stdint.h>
#include <ap_int.h>

constexpr int SEQ_LEN       = 50;
constexpr int EMBED_DIM     = 30;    // Input dimension (30 features)
constexpr int HEAD_SIZE     = 128;   // Attention head size
constexpr int NUM_HEADS     = 1;     // Single head attention
constexpr int FF_DIM        = 128;   // Feed-forward dimension
constexpr int NUM_BLOCKS    = 2;     // Number of transformer blocks
constexpr int MLP_UNITS     = 160;   // MLP head units
constexpr float LN_EPSILON  = 1e-6f; // LayerNorm epsilon
constexpr int FINAL_NEURONS = 1;     // Regression output
constexpr int WORD_BITS     = 32;    // For packing binary weights

// -------------------- Binary Weight Containers --------------------
struct LayerNormWeights {
    float gamma[EMBED_DIM];
    float beta[EMBED_DIM];
};

// Binary attention weights (packed into 32-bit words)
struct BinaryMHAWeights {
    // Q/K/V projections: 30 input bits fit in 1x32-bit word
    ap_uint<32> Wq_bits[HEAD_SIZE];  // [128][1] packed
    ap_uint<32> Wk_bits[HEAD_SIZE];
    ap_uint<32> Wv_bits[HEAD_SIZE];
    
    // Alpha scaling factors (trained)
    float alpha_q[HEAD_SIZE];
    float alpha_k[HEAD_SIZE];
    float alpha_v[HEAD_SIZE];
    
    // Biases (float)
    float b_q[HEAD_SIZE];
    float b_k[HEAD_SIZE];
    float b_v[HEAD_SIZE];
    
    // Output projection: 128 input bits need 4x32-bit words = 128 bits
    ap_uint<32> Wo_bits[EMBED_DIM][4];  // 128 bits = 4 words
    
    float alpha_o[EMBED_DIM];
    float b_o[EMBED_DIM];
};

// Binary FFN weights
struct BinaryFFNWeights {
    // First projection: 30 → 128
    ap_uint<32> W1_bits[FF_DIM];      // [128][1] packed
    
    // Second projection: 128 → 30  
    ap_uint<32> W2_bits[EMBED_DIM][4];  // 128 bits = 4 words
    
    // Alpha scaling factors
    float alpha1[FF_DIM];
    float alpha2[EMBED_DIM];
    
    // Biases
    float b1[FF_DIM];
    float b2[EMBED_DIM];
};

struct EncoderBlockWeights {
    LayerNormWeights ln1;
    BinaryMHAWeights mha;
    LayerNormWeights ln2;
    BinaryFFNWeights ffn;
};

struct PositionalEmbeddingWeights {
    float position_embedding[SEQ_LEN][EMBED_DIM];
};

// MLP Head stays FP32
struct MLPWeights {
    float W1[EMBED_DIM][MLP_UNITS];
    float b1[MLP_UNITS];
    
    float W2[MLP_UNITS][FINAL_NEURONS];
    float b2[FINAL_NEURONS];
};

struct XNORTransformerModel {
    PositionalEmbeddingWeights pos_emb;
    EncoderBlockWeights encoders[NUM_BLOCKS];
    MLPWeights mlp;
};

// Initialize model weights from trans_weights_bin.h
void init_xnor_model_weights(XNORTransformerModel &model);

// Top-level HLS function
extern "C" void trans_bin(
    const float input[SEQ_LEN][EMBED_DIM],
    float output[FINAL_NEURONS]
);