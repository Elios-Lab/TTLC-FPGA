#include "trans_bin.h"
#include "trans_weights_bin.h"
#include <hls_math.h>

// --------------------------------------------------
// Helper functions
// --------------------------------------------------

// RelU - IDENTICAL to FP32 version
static inline float relu(float x) { 
#pragma HLS INLINE
    return (x > 0.0f) ? x : 0.0f; 
}

// LayerNorm - IDENTICAL to FP32 version
static void layer_norm(
    float x[][EMBED_DIM],
    const float* gamma,
    const float* beta,
    int len,
    int dim
) {
#pragma HLS INLINE off
    
LN_TOKENS:
    for (int i = 0; i < len; ++i) {
        float mean = 0.0f;
        
    LN_MEAN:
        for (int j = 0; j < dim; ++j) {
#pragma HLS PIPELINE II=3
#pragma HLS DEPENDENCE variable=mean inter false
            mean += x[i][j];
        }
        mean /= (float)dim;

        float var = 0.0f;
        
    LN_VAR:
        for (int j = 0; j < dim; ++j) {
#pragma HLS PIPELINE II=3
#pragma HLS DEPENDENCE variable=var inter false
            float diff = x[i][j] - mean;
            var += diff * diff;
        }
        var /= (float)dim;

        float inv_std = 1.0f / hls::sqrtf(var + LN_EPSILON);

    LN_APPLY:
        for (int j = 0; j < dim; ++j) {
#pragma HLS PIPELINE II=1
            float norm = (x[i][j] - mean) * inv_std;
            x[i][j] = norm * gamma[j] + beta[j];
        }
    }
}

// Softmax - IDENTICAL to FP32 version
static void softmax(const float* x, float* out, int len) {
#pragma HLS INLINE off
    
    float max_val = x[0];
SOFTMAX_MAX:
    for (int i = 1; i < len; ++i) {
#pragma HLS PIPELINE II=1
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
SOFTMAX_EXP:
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=3
#pragma HLS DEPENDENCE variable=sum inter false
        float exp_val = hls::expf(x[i] - max_val);
        out[i] = exp_val;
        sum += exp_val;
    }

    float inv_sum = 1.0f / (sum + 1e-8f);
SOFTMAX_NORM:
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        out[i] *= inv_sum;
    }
}

// popcount tree
template<int N>
struct Popcount {
    static int eval(ap_uint<N> x) {
#pragma HLS INLINE
        const int HALF = N / 2;
        ap_uint<HALF>     lo = x.range(HALF - 1, 0);
        ap_uint<N - HALF> hi = x.range(N - 1, HALF);
        return Popcount<HALF>::eval(lo) + Popcount<N - HALF>::eval(hi);
    }
};

template<>
struct Popcount<1> {
    static int eval(ap_uint<1> x) {
#pragma HLS INLINE
        return (int)x[0];
    }
};

template<int N>
static int popcount_hierarchical(ap_uint<N> x) {
#pragma HLS INLINE
    return Popcount<N>::eval(x);
}

// XNOR-Popcount Dense Layer (30→128)
static void xnor_dense_30_to_128(
    const float*       x_bin,       // {0,1}
    const ap_uint<32>* W_bits,      // [out_dim]
    const float*       alpha,       // [out_dim]
    const float*       bias,        // [out_dim] or nullptr
    float*             y,           // [seq_len*out_dim]
    int                seq_len,
    int                out_dim
) {
#pragma HLS INLINE off

    const ap_uint<32> MASK30 = (ap_uint<32>(1) << EMBED_DIM) - 1;

XNOR_30_128_ROWS:
    for (int i = 0; i < seq_len; ++i) {

        // 1) Load 30 floats sequentially
        float xb_local[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=xb_local complete

    LOAD_XB:
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            xb_local[d] = x_bin[i * EMBED_DIM + d];
        }

        // 2) Pack to bits (combinational after load)
        ap_uint<32> x_bits = 0;
        
    PACK_BITS:
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS UNROLL
            x_bits[d] = (xb_local[d] != 0.0f);
        }

        // 3) Compute outputs
    XNOR_30_128_COLS:
        for (int o = 0; o < out_dim; ++o) {
#pragma HLS PIPELINE II=1

            ap_uint<32> xnor_result  = ~(x_bits ^ W_bits[o]);
            xnor_result             &= MASK30; 
            int count                = popcount_hierarchical<32>(xnor_result);

            float dot                = 2.0f * (float)count - (float)EMBED_DIM;
            float result             = alpha[o] * dot + ((bias != nullptr) ? bias[o] : 0.0f);

            y[i * out_dim + o]       = result;
        }
    }
}

// XNOR-Popcount Dense Layer (128→30)
static void xnor_dense_128_to_30(
    const float*       x_bin,           // [seq_len,128] values {0,1}
    const ap_uint<32>  W_bits[][4],     // [out_dim][4] packed weights
    const float*       alpha,
    const float*       bias,
    float*             y,
    int                seq_len,
    int                out_dim
) {
#pragma HLS INLINE off

    for (int i = 0; i < seq_len; ++i) {
        ap_uint<32> xw[4];
#pragma HLS ARRAY_PARTITION variable=xw complete dim=1

        // Pack x into 4x32-bit
        for (int w = 0; w < 4; ++w) {
#pragma HLS UNROLL
            ap_uint<32> tmp = 0;
            for (int b = 0; b < 32; ++b) {
#pragma HLS UNROLL
                int d  = w * 32 + b;
                float v = x_bin[i * 128 + d];
                tmp[b]  = (v >= 0.5f) ? 1 : 0;
            }
            xw[w] = tmp;
        }

        for (int o = 0; o < out_dim; ++o) {
#pragma HLS PIPELINE II=2
            int count = 0;

            // XNOR+popcount across 4 words
            for (int w = 0; w < 4; ++w) {
#pragma HLS UNROLL
                ap_uint<32> xnor_word = ~(xw[w] ^ W_bits[o][w]);
                // popcount 32 bits
                for (int b = 0; b < 32; ++b) {
#pragma HLS UNROLL
                    count += (int)xnor_word[b];
                }
            }

            float dot          = 2.0f * (float)count - 128.0f;
            float result       = alpha[o] * dot + ((bias != nullptr) ? bias[o] : 0.0f);
            y[i * out_dim + o] = result;
        }
    }
}

// FP32 Dense (for MLP head) - IDENTICAL to FP32 version
static void dense_vec(
    const float* x,
    const float* W,
    const float* b,
    float*       y,
    int          in_dim,
    int          out_dim
) {
#pragma HLS INLINE off

DENSE_VEC_OUT:
    for (int o = 0; o < out_dim; ++o) {
        float acc0 = (b != nullptr) ? b[o] : 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

    DENSE_VEC_IN:
        for (int d = 0; d < in_dim; ++d) {
#pragma HLS PIPELINE II=3
            float prod = x[d] * W[d * out_dim + o];

            switch (d & 3) {
                case 0: acc0 += prod; break;
                case 1: acc1 += prod; break;
                case 2: acc2 += prod; break;
                default: acc3 += prod; break;
            }
        }

        y[o] = (acc0 + acc1) + (acc2 + acc3);
    }
}

// Global pooling - IDENTICAL to FP32 version
static void global_average_pooling_1d(float x[][EMBED_DIM], float* out) {
#pragma HLS INLINE off
    
GAP_DIM:
    for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE off
        float sum = 0.0f;
        
    GAP_SEQ:
        for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=3
#pragma HLS DEPENDENCE variable=sum inter false
            sum += x[i][d];
        }
        out[d] = sum / (float)SEQ_LEN;
    }
}

// Positional embedding - IDENTICAL to FP32 version
static void apply_positional_embedding(
    float x[][EMBED_DIM],
    const PositionalEmbeddingWeights& w
) {
#pragma HLS INLINE off
    
POS_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
    POS_D:
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x[i][d] += w.position_embedding[i][d];
        }
    }
}

// Binary activation: FP32 → {0,1}
static void binary_activation(float x[][EMBED_DIM]) {
#pragma HLS INLINE off
BIN_ACT_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x[i][d] = (x[i][d] >= 0.0f) ? 1.0f : 0.0f;
        }
    }
}

// --------------------------------------------------
// Binary projection 30->128 but output ONLY sign bits
// bit[o] = 1 if (alpha[o]*(2*popcount - 30) + bias[o]) >= 0 else 0
// This matches the thresholding rule (>=0.0f).
// --------------------------------------------------
static void xnor_dense_30_to_128_bits(
    const float*       x_bin,
    const ap_uint<32>* W_bits,
    const float*       alpha,
    const float*       bias,
    ap_uint<128>*      out_bits,
    int                seq_len
) {
#pragma HLS INLINE off

    const ap_uint<32> MASK30 = (ap_uint<32>(1) << EMBED_DIM) - 1; // EMBED_DIM=30

ROWS_QK:
    for (int i = 0; i < seq_len; ++i) {

        ap_uint<32> x_bits = 0;
        
PACK_IN_30:
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS UNROLL
            x_bits[d] = (x_bin[i * EMBED_DIM + d] != 0.0f);
        }

        ap_uint<128> qb = 0;

COLS_QK:
        for (int o = 0; o < HEAD_SIZE; ++o) {
#pragma HLS PIPELINE II=1
            ap_uint<32> xnor_word = ~(x_bits ^ W_bits[o]);

            // only count real 30 dims
            xnor_word &= MASK30;

            int count = popcount_hierarchical<32>(xnor_word);

            float dot30 = 2.0f * (float)count - (float)EMBED_DIM;
            float qf    = alpha[o] * dot30 + bias[o];
            qb[o]       = (qf >= 0.0f);
        }

        out_bits[i] = qb;
    }
}

// Binary Multi-Head Attention (single head)
static void binary_multi_head_attention_soft(
    float x_norm[][EMBED_DIM],      // FP32 input (will be binarized)
    const BinaryMHAWeights& w,
    float attn_out[][EMBED_DIM]
) {
#pragma HLS INLINE off

    // V must stay FP32 for context accumulation
    float V[SEQ_LEN][HEAD_SIZE];
#pragma HLS ARRAY_PARTITION variable=V cyclic factor=2 dim=2

    // Binarize input for projections
    float x_bin[SEQ_LEN][EMBED_DIM];
    
BIN_ATTN_INPUT:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x_bin[i][d] = (x_norm[i][d] >= 0.0f) ? 1.0f : 0.0f;
        }
    }

    // Compute Q_bits and K_bits directly (NO FP32 Q/K arrays, NO packing pass)
    ap_uint<128> Q_bits[SEQ_LEN];
    ap_uint<128> K_bits[SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=Q_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=K_bits complete dim=1

    xnor_dense_30_to_128_bits(&x_bin[0][0], w.Wq_bits, w.alpha_q, w.b_q, Q_bits, SEQ_LEN);
    xnor_dense_30_to_128_bits(&x_bin[0][0], w.Wk_bits, w.alpha_k, w.b_k, K_bits, SEQ_LEN);

    // V projection stays float projection
    xnor_dense_30_to_128(&x_bin[0][0], w.Wv_bits, w.alpha_v, w.b_v, &V[0][0], SEQ_LEN, HEAD_SIZE);

    // Attention scores
    float scores[SEQ_LEN][SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=scores cyclic factor=4 dim=2

    const float inv_sqrt_dk = 1.0f / hls::sqrtf((float)HEAD_SIZE);

ATTN_SCORE_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
    ATTN_SCORE_J:
        for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=1
            ap_uint<128> xnor = ~(Q_bits[i] ^ K_bits[j]);
            int count         = popcount_hierarchical<128>(xnor);              // [0..128]
            float dot         = 2.0f * (float)count - (float)HEAD_SIZE;        // [-128..+128]
            scores[i][j]      = dot * inv_sqrt_dk;                             // keep original scale
        }
    }

    // Softmax
    float attn_weights[SEQ_LEN][SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=attn_weights cyclic factor=4 dim=2

ATTN_SOFTMAX:
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE off
        softmax(scores[i], attn_weights[i], SEQ_LEN);
    }

    // Context
    float context[SEQ_LEN][HEAD_SIZE];
#pragma HLS ARRAY_PARTITION variable=context cyclic factor=2 dim=2

ATTN_CTX_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
    ATTN_CTX_D:
        for (int d = 0; d < HEAD_SIZE; ++d) {
#pragma HLS PIPELINE II=1
            float sum = 0.0f;

        ATTN_CTX_J:
            for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS UNROLL factor=2
                sum += attn_weights[i][j] * V[j][d];
            }
            context[i][d] = sum;
        }
    }

    // Binarize context for output projection
    float ctx_bin[SEQ_LEN][HEAD_SIZE];
    
BIN_ATTN_OUTPUT:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < HEAD_SIZE; ++d) {
#pragma HLS PIPELINE II=1
            ctx_bin[i][d] = (context[i][d] >= 0.0f) ? 1.0f : 0.0f;
        }
    }

    // Binary output projection
    xnor_dense_128_to_30(&ctx_bin[0][0], w.Wo_bits, w.alpha_o, w.b_o, &attn_out[0][0], SEQ_LEN, EMBED_DIM);
}

// Binary Encoder Block
static void binary_encoder_block(
    float x[][EMBED_DIM],
    const EncoderBlockWeights& w
) {
#pragma HLS INLINE off
    
    // Save input for residual
    float x_input[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=x_input cyclic factor=2 dim=2
    
ENC_SAVE1:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x_input[i][d] = x[i][d];
        }
    }

    // LN1 (FP32)
    layer_norm(x, w.ln1.gamma, w.ln1.beta, SEQ_LEN, EMBED_DIM);

    // MHA (Binary)
    float attn_out[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=attn_out cyclic factor=2 dim=2
    
    binary_multi_head_attention_soft(x, w.mha, attn_out);

    // Residual 1 (FP32)
ENC_RES1:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x[i][d] = attn_out[i][d] + x_input[i][d];
        }
    }

    // Save for residual 2
    float x_residual[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=x_residual cyclic factor=2 dim=2
    
ENC_SAVE2:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x_residual[i][d] = x[i][d];
        }
    }

    // LN2 (FP32)
    layer_norm(x, w.ln2.gamma, w.ln2.beta, SEQ_LEN, EMBED_DIM);

    // FFN (Binary)
    float ffn_hidden[SEQ_LEN][FF_DIM];
#pragma HLS ARRAY_PARTITION variable=ffn_hidden cyclic factor=2 dim=2
    
    // First FFN projection (30→128)
    float x_bin[SEQ_LEN][EMBED_DIM];
    
BIN_FFN_INPUT:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x_bin[i][d] = (x[i][d] >= 0.0f) ? 1.0f : 0.0f;
        }
    }
    
    xnor_dense_30_to_128(&x_bin[0][0], w.ffn.W1_bits, w.ffn.alpha1, w.ffn.b1, &ffn_hidden[0][0], SEQ_LEN, FF_DIM);

    // ReLU (FP32)
ENC_RELU:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < FF_DIM; ++d) {
#pragma HLS PIPELINE II=1
            ffn_hidden[i][d] = relu(ffn_hidden[i][d]);
        }
    }

    // Second FFN projection (128→30)
    float ffn_out[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=ffn_out cyclic factor=2 dim=2
    
    // Binarize ReLU output
    float ffn_bin[SEQ_LEN][FF_DIM];
    
BIN_FFN_OUTPUT:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < FF_DIM; ++d) {
#pragma HLS PIPELINE II=1
            ffn_bin[i][d] = (ffn_hidden[i][d] >= 0.0f) ? 1.0f : 0.0f;
        }
    }
    
    xnor_dense_128_to_30(&ffn_bin[0][0], w.ffn.W2_bits, w.ffn.alpha2, w.ffn.b2, &ffn_out[0][0], SEQ_LEN, EMBED_DIM);

    // Residual 2 (FP32)
ENC_RES2:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x[i][d] = x_residual[i][d] + ffn_out[i][d];
        }
    }
}

// MLP forward (FP32) - IDENTICAL to FP32 version
static void mlp_forward(const float* pooled, const MLPWeights& w, float* out) {
#pragma HLS INLINE off
    
    float hidden[MLP_UNITS];
#pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=4

    dense_vec(pooled, &w.W1[0][0], w.b1, hidden, EMBED_DIM, MLP_UNITS);
    
MLP_RELU:
    for (int i = 0; i < MLP_UNITS; ++i) {
#pragma HLS PIPELINE II=1
        hidden[i] = relu(hidden[i]);
    }
    
    dense_vec(hidden, &w.W2[0][0], w.b2, out, MLP_UNITS, FINAL_NEURONS);
}

// Weight initialization from trans_weights_bin.h
void init_xnor_model_weights(XNORTransformerModel &model) {
#pragma HLS INLINE off

    // Positional encoding
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.pos_emb.position_embedding[i][d] = positional_encoding_EMBED[i][d];
        }
    }

    // ================= Block 0 =================

    // LayerNorm weights
    for (int d = 0; d < EMBED_DIM; ++d) {
        model.encoders[0].ln1.gamma[d] = blk0_ln1_GAMMA[d];
        model.encoders[0].ln1.beta[d]  = blk0_ln1_BETA[d];
        model.encoders[0].ln2.gamma[d] = blk0_ln2_GAMMA[d];
        model.encoders[0].ln2.beta[d]  = blk0_ln2_BETA[d];
    }

    // MHA Q/K/V (30->128) packed in [o][0]
    for (int o = 0; o < HEAD_SIZE; ++o) {
        model.encoders[0].mha.Wq_bits[o] = blk0_bmha_q_W_BITS[o][0];
        model.encoders[0].mha.Wk_bits[o] = blk0_bmha_k_W_BITS[o][0];
        model.encoders[0].mha.Wv_bits[o] = blk0_bmha_v_W_BITS[o][0];

        model.encoders[0].mha.alpha_q[o] = blk0_bmha_q_ALPHA[o];
        model.encoders[0].mha.alpha_k[o] = blk0_bmha_k_ALPHA[o];
        model.encoders[0].mha.alpha_v[o] = blk0_bmha_v_ALPHA[o];

        model.encoders[0].mha.b_q[o]     = blk0_bmha_q_BIAS[o];
        model.encoders[0].mha.b_k[o]     = blk0_bmha_k_BIAS[o];
        model.encoders[0].mha.b_v[o]     = blk0_bmha_v_BIAS[o];
    }

    // MHA output projection (128->30): packed [o][4] words
    for (int o = 0; o < EMBED_DIM; ++o) {
        // copy 4x32-bit words (128 bits)
        for (int w = 0; w < 4; ++w) {
            model.encoders[0].mha.Wo_bits[o][w] = blk0_bmha_o_W_BITS[o][w];
        }
        model.encoders[0].mha.alpha_o[o] = blk0_bmha_o_ALPHA[o];
        model.encoders[0].mha.b_o[o]     = blk0_bmha_o_BIAS[o];
    }

    // FFN first projection (30->128): packed in [o][0]
    for (int o = 0; o < FF_DIM; ++o) {
        model.encoders[0].ffn.W1_bits[o] = blk0_ff1_W_BITS[o][0];
        model.encoders[0].ffn.alpha1[o]  = blk0_ff1_ALPHA[o];
        model.encoders[0].ffn.b1[o]      = blk0_ff1_BIAS[o];
    }

    // FFN second projection (128->30): packed [o][4] words
    for (int o = 0; o < EMBED_DIM; ++o) {
        for (int w = 0; w < 4; ++w) {
            model.encoders[0].ffn.W2_bits[o][w] = blk0_ff2_W_BITS[o][w];
        }
        model.encoders[0].ffn.alpha2[o] = blk0_ff2_ALPHA[o];
        model.encoders[0].ffn.b2[o]     = blk0_ff2_BIAS[o];
    }

    // ================= Block 1 =================

    // LayerNorm weights
    for (int d = 0; d < EMBED_DIM; ++d) {
        model.encoders[1].ln1.gamma[d] = blk1_ln1_GAMMA[d];
        model.encoders[1].ln1.beta[d]  = blk1_ln1_BETA[d];
        model.encoders[1].ln2.gamma[d] = blk1_ln2_GAMMA[d];
        model.encoders[1].ln2.beta[d]  = blk1_ln2_BETA[d];
    }

    // MHA Q/K/V (30->128)
    for (int o = 0; o < HEAD_SIZE; ++o) {
        model.encoders[1].mha.Wq_bits[o] = blk1_bmha_q_W_BITS[o][0];
        model.encoders[1].mha.Wk_bits[o] = blk1_bmha_k_W_BITS[o][0];
        model.encoders[1].mha.Wv_bits[o] = blk1_bmha_v_W_BITS[o][0];

        model.encoders[1].mha.alpha_q[o] = blk1_bmha_q_ALPHA[o];
        model.encoders[1].mha.alpha_k[o] = blk1_bmha_k_ALPHA[o];
        model.encoders[1].mha.alpha_v[o] = blk1_bmha_v_ALPHA[o];

        model.encoders[1].mha.b_q[o]     = blk1_bmha_q_BIAS[o];
        model.encoders[1].mha.b_k[o]     = blk1_bmha_k_BIAS[o];
        model.encoders[1].mha.b_v[o]     = blk1_bmha_v_BIAS[o];
    }

    // MHA output projection (128->30): packed [o][4]
    for (int o = 0; o < EMBED_DIM; ++o) {
        for (int w = 0; w < 4; ++w) {
            model.encoders[1].mha.Wo_bits[o][w] = blk1_bmha_o_W_BITS[o][w];
        }
        model.encoders[1].mha.alpha_o[o] = blk1_bmha_o_ALPHA[o];
        model.encoders[1].mha.b_o[o]     = blk1_bmha_o_BIAS[o];
    }

    // FFN first projection (30->128)
    for (int o = 0; o < FF_DIM; ++o) {
        model.encoders[1].ffn.W1_bits[o] = blk1_ff1_W_BITS[o][0];
        model.encoders[1].ffn.alpha1[o]  = blk1_ff1_ALPHA[o];
        model.encoders[1].ffn.b1[o]      = blk1_ff1_BIAS[o];
    }

    // FFN second projection (128->30): packed [o][4]
    for (int o = 0; o < EMBED_DIM; ++o) {
        for (int w = 0; w < 4; ++w) {
            model.encoders[1].ffn.W2_bits[o][w] = blk1_ff2_W_BITS[o][w];
        }
        model.encoders[1].ffn.alpha2[o] = blk1_ff2_ALPHA[o];
        model.encoders[1].ffn.b2[o]     = blk1_ff2_BIAS[o];
    }

    // ================= MLP Head (FP32) =================

    for (int i = 0; i < EMBED_DIM; ++i) {
        for (int o = 0; o < MLP_UNITS; ++o) {
            model.mlp.W1[i][o] = head_fc_W[i][o];
        }
    }

    for (int o = 0; o < MLP_UNITS; ++o) {
        model.mlp.b1[o] = head_fc_B[o];
    }

    for (int i = 0; i < MLP_UNITS; ++i) {
        model.mlp.W2[i][0] = out_W[i][0];
    }

    model.mlp.b2[0] = out_B[0];
}

// Top-level HLS function
extern "C" void trans_bin(
    const float input[SEQ_LEN][EMBED_DIM],
    float output[FINAL_NEURONS]
) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=IN depth=1500
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=OUT depth=1

    static XNORTransformerModel model;
    static bool initialized = false;

    if (!initialized) {
        init_xnor_model_weights(model);
        initialized = true;
    }
    
    float x[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=2 dim=2
    
// Copy input
INPUT_COPY:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x[i][d] = input[i][d];
        }
    }

    // Positional encoding (FP32)
    apply_positional_embedding(x, model.pos_emb);

    // First binarization (after positional encoding)
    binary_activation(x);

    // Binary encoder blocks
ENCODER_BLOCKS:
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        binary_encoder_block(x, model.encoders[b]);
    }

    // Global pooling (FP32)
    float pooled[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=pooled cyclic factor=4
    
    global_average_pooling_1d(x, pooled);

    // MLP head (FP32)
    mlp_forward(pooled, model.mlp, output);
}