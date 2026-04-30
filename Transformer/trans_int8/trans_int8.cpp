#include "trans_int8.h"

#include <cstdio>
#include <hls_math.h>

#include "trans_weights_fp32.h"   // original float weights (LN, biases)
#include "trans_weights_int8.h"   // quantized INT8 weights + scales

// --------------------------------------------------------------------
// Global numeric configuration
// --------------------------------------------------------------------

// Activation scale
static const float ACT_SCALE = 4.6f / 128.0f;

// Small epsilon for LayerNorm
static const float LN_EPS = 1e-5f;

// --------------------------------------------------------------------
// Quantization helpers
// --------------------------------------------------------------------

static inline qint8_t quantize_act(float x) {
#pragma HLS INLINE
    float q_f = x / ACT_SCALE;
    int q_i   = (int)hls::roundf(q_f);
    if (q_i > 127)  q_i = 127;
    if (q_i < -128) q_i = -128;
    return (qint8_t)q_i;
}

static inline float dequantize_act(qint8_t q) {
#pragma HLS INLINE
    return (float)q * ACT_SCALE;
}

// ReLU on INT8 activations
static inline qint8_t relu_q(qint8_t x) {
#pragma HLS INLINE
    return (x < 0) ? (qint8_t)0 : x;
}

// Dense (matrix) INT8 core. We convert y_real back into INT8 in the same ACT_SCALE domain.
static void dense_int8_mat(
    const qint8_t* x_q,
    const qint8_t* W_q,
    const float*   b,
    float          w_scale,
    qint8_t*       y_q,
    int len,
    int in_dim,
    int out_dim
) {

#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=y_q cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=W_q block factor=4

DENSE_ROWS:
    for (int i = 0; i < len; ++i) {
DENSE_COLS:
        for (int o = 0; o < out_dim; ++o) {
#pragma HLS PIPELINE II=2
            qint32_t acc = 0;

DENSE_DOT:
            for (int d = 0; d < in_dim; ++d) {
#pragma HLS PIPELINE II=1
                qint8_t xv = x_q[i * in_dim + d];
                qint8_t wv = W_q[d * out_dim + o];
                acc += (qint16_t)xv * (qint16_t)wv;
            }

            // Dequant
            float y_real = ACT_SCALE * ((float)acc * w_scale);
            if (b) y_real += b[o];

            y_q[i * out_dim + o] = quantize_act(y_real);
        }
    }
}

// Vector version for the final MLP head
static void dense_int8_vec(
    const qint8_t* x_q,
    const qint8_t* W_q,
    const float*   b,
    float          w_scale,
    qint8_t*       y_q,
    int in_dim,
    int out_dim
) {
#pragma HLS INLINE off

DENSE_VEC_OUT:
    for (int o = 0; o < out_dim; ++o) {
        float acc0 = (b != nullptr) ? b[o] : 0.0f;
        float acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

DENSE_VEC_IN:
        for (int d = 0; d < in_dim; ++d) {
#pragma HLS PIPELINE II=3
            qint8_t xv = x_q[d];
            qint8_t wv = W_q[d * out_dim + o];

            float prod = ACT_SCALE * ((float)((qint16_t)xv * (qint16_t)wv) * w_scale);

            switch (d & 3) {
                case 0: acc0 += prod; break;
                case 1: acc1 += prod; break;
                case 2: acc2 += prod; break;
                default: acc3 += prod; break;
            }
        }

        float y_real = (acc0 + acc1) + (acc2 + acc3);
        y_q[o] = quantize_act(y_real);
    }
}

// Softmax along a 1D vector of length len (in float).
static void softmax_float(const float* x, float* out, int len) {
#pragma HLS INLINE off
    float max_val = x[0];
SF_MAX:
    for (int i = 1; i < len; ++i) {
#pragma HLS PIPELINE II=1
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
SF_EXP:
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=3
#pragma HLS DEPENDENCE variable=sum inter false
        float e = hls::expf(x[i] - max_val);
        out[i] = e;
        sum   += e;
    }

    float inv_sum = 1.0f / (sum + 1e-8f);
SF_NORM:
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        out[i] *= inv_sum;
    }
}

// LayerNorm over the last dimension (dim), with float compute:
static void layernorm_int8(
    const qint8_t* x_q,
    const float* gamma,
    const float* beta,
    qint8_t* out_q,
    int len,
    int dim
) {
#pragma HLS INLINE off

LN_ROWS:
    for (int i = 0; i < len; ++i) {
        // dequantize
        float tmp[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=tmp complete

        float mean = 0.0f;
    LN_LOAD:
        for (int j = 0; j < dim; ++j) {
    #pragma HLS PIPELINE II=3
    #pragma HLS DEPENDENCE variable=mean inter false
            float v = dequantize_act(x_q[i * dim + j]);
            tmp[j] = v;
            mean  += v;
        }
        mean /= (float)dim;

        float var = 0.0f;
    LN_VAR:
        for (int j = 0; j < dim; ++j) {
    #pragma HLS PIPELINE II=3
    #pragma HLS DEPENDENCE variable=var inter false
            float diff = tmp[j] - mean;
            var += diff * diff;
        }
        var /= (float)dim;

        float inv_std = 1.0f / hls::sqrtf(var + LN_EPS);

    LN_WRITE:
        for (int j = 0; j < dim; ++j) {
#pragma HLS PIPELINE II=1
            float norm = (tmp[j] - mean) * inv_std;
            float y    = norm * gamma[j] + beta[j];
            out_q[i * dim + j] = quantize_act(y);
        }
    }
}

static void global_average_pooling_1d(
    const qint8_t x_q[SEQ_LEN][EMBED_DIM],
    float pooled[EMBED_DIM]
) {
#pragma HLS INLINE off

GAP_DIM:
    for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE off
        float sum = 0.0f;

GAP_SEQ:
        for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=3
#pragma HLS DEPENDENCE variable=sum inter false
            sum += dequantize_act(x_q[i][d]);
        }

        pooled[d] = sum / (float)SEQ_LEN;
    }
}

// Multi-head self-attention (single head)
static void multi_head_attention_block_int8(
    const qint8_t x_q[SEQ_LEN][EMBED_DIM],
    const EncoderBlockWeightsInt8 &w,
    qint8_t out_q[SEQ_LEN][EMBED_DIM]
) {
#pragma HLS INLINE off

    qint8_t x_ln_q[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=x_ln_q cyclic factor=2 dim=2

    qint8_t Q_q[SEQ_LEN][HEAD_SIZE];
    qint8_t K_q[SEQ_LEN][HEAD_SIZE];
    qint8_t V_q[SEQ_LEN][HEAD_SIZE];
#pragma HLS ARRAY_PARTITION variable=Q_q cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=K_q cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=V_q cyclic factor=2 dim=2

    // LN1 (float) -> quantized
    layernorm_int8(&x_q[0][0], w.ln1_gamma, w.ln1_beta, &x_ln_q[0][0], SEQ_LEN, EMBED_DIM);

    // Q/K/V projections
    dense_int8_mat(&x_ln_q[0][0], &w.W_q[0][0], w.b_q, w.W_q_scale, &Q_q[0][0], SEQ_LEN, EMBED_DIM, HEAD_SIZE);
    dense_int8_mat(&x_ln_q[0][0], &w.W_k[0][0], w.b_k, w.W_k_scale, &K_q[0][0], SEQ_LEN, EMBED_DIM, HEAD_SIZE);
    dense_int8_mat(&x_ln_q[0][0], &w.W_v[0][0], w.b_v, w.W_v_scale, &V_q[0][0], SEQ_LEN, EMBED_DIM, HEAD_SIZE);

    // Scores
    float scores[SEQ_LEN][SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=scores cyclic factor=4 dim=2

    const float inv_sqrt_dk = 1.0f / hls::sqrtf((float)HEAD_SIZE);
    const float qk_scale = ACT_SCALE * ACT_SCALE; // because Q and K are quantized in ACT_SCALE domain

ATTN_SCORE_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
ATTN_SCORE_J:
        for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=16

            // Stage 1: partial sums
            float psum[HEAD_SIZE/4];
#pragma HLS ARRAY_PARTITION variable=psum complete dim=1

ATTN_SCORE_PS:
            for (int b = 0; b < HEAD_SIZE/4; ++b) {
                int d0 = 4*b + 0;
                int d1 = 4*b + 1;
                int d2 = 4*b + 2;
                int d3 = 4*b + 3;

                // INT8 multiply, converted to float with qk_scale
                float t0 = qk_scale * (float)((qint16_t)Q_q[i][d0] * (qint16_t)K_q[j][d0]);
                float t1 = qk_scale * (float)((qint16_t)Q_q[i][d1] * (qint16_t)K_q[j][d1]);
                float t2 = qk_scale * (float)((qint16_t)Q_q[i][d2] * (qint16_t)K_q[j][d2]);
                float t3 = qk_scale * (float)((qint16_t)Q_q[i][d3] * (qint16_t)K_q[j][d3]);

                psum[b] = (t0 + t1) + (t2 + t3);
            }

            // Stage 2: pipelined reduction
            float dot = 0.0f;
ATTN_SCORE_REDUCE:
            for (int b = 0; b < HEAD_SIZE/4; ++b) {
#pragma HLS PIPELINE II=1
                dot += psum[b];
            }

            scores[i][j] = dot * inv_sqrt_dk;
        }
    }

    // Softmax
    float attn_weights[SEQ_LEN][SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=attn_weights cyclic factor=4 dim=2

ATTN_SOFTMAX:
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE off
        softmax_float(scores[i], attn_weights[i], SEQ_LEN);
    }

    // Context
    qint8_t context_q[SEQ_LEN][HEAD_SIZE];
#pragma HLS ARRAY_PARTITION variable=context_q cyclic factor=2 dim=2

ATTN_CTX_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
ATTN_CTX_D:
        for (int d = 0; d < HEAD_SIZE; ++d) {
#pragma HLS PIPELINE II=1
            float sum = 0.0f;

ATTN_CTX_J:
            for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS UNROLL factor=2
                float p = attn_weights[i][j];
                float v = dequantize_act(V_q[j][d]);
                sum += p * v;
            }

            context_q[i][d] = quantize_act(sum);
        }
    }

    // Output projection
    qint8_t attn_out_q[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=attn_out_q cyclic factor=2 dim=2

    dense_int8_mat(&context_q[0][0], &w.W_o[0][0], w.b_o, w.W_o_scale, &attn_out_q[0][0], SEQ_LEN, HEAD_SIZE, EMBED_DIM);

    // Residual add
RESIDUAL_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
RESIDUAL_J:
        for (int j = 0; j < EMBED_DIM; ++j) {
#pragma HLS PIPELINE II=1
            float xr   = dequantize_act(x_q[i][j]);
            float ar   = dequantize_act(attn_out_q[i][j]);
            out_q[i][j] = quantize_act(xr + ar);
        }
    }
}



// Feed-forward block (FFN)
static void feedforward_block_int8(
    const qint8_t x_q[SEQ_LEN][EMBED_DIM],
    const EncoderBlockWeightsInt8 &w,
    qint8_t out_q[SEQ_LEN][EMBED_DIM]
) {
#pragma HLS INLINE off

    qint8_t x_ln_q[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=x_ln_q cyclic factor=2 dim=2

    layernorm_int8(&x_q[0][0], w.ln2_gamma, w.ln2_beta, &x_ln_q[0][0], SEQ_LEN, EMBED_DIM);

    qint8_t ffn_hidden[SEQ_LEN][FF_DIM];
#pragma HLS ARRAY_PARTITION variable=ffn_hidden cyclic factor=2 dim=2

    dense_int8_mat(&x_ln_q[0][0], &w.W_ff1[0][0], w.b_ff1, w.W_ff1_scale, &ffn_hidden[0][0], SEQ_LEN, EMBED_DIM, FF_DIM);

FFN_RELU:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < FF_DIM; ++d) {
#pragma HLS PIPELINE II=1
            ffn_hidden[i][d] = relu_q(ffn_hidden[i][d]);
        }
    }

    qint8_t ffn_out_q[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=ffn_out_q cyclic factor=2 dim=2

    dense_int8_mat(&ffn_hidden[0][0], &w.W_ff2[0][0], w.b_ff2, w.W_ff2_scale, &ffn_out_q[0][0], SEQ_LEN, FF_DIM, EMBED_DIM);

RES2_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            float xr  = dequantize_act(x_q[i][d]);
            float fr  = dequantize_act(ffn_out_q[i][d]);
            out_q[i][d] = quantize_act(xr + fr);
        }
    }
}

// One encoder block
static void encoder_block_forward_int8(
    qint8_t x_q[SEQ_LEN][EMBED_DIM],
    const EncoderBlockWeightsInt8 &w
) {
#pragma HLS INLINE off

    qint8_t attn_out_q[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=attn_out_q cyclic factor=2 dim=2

    multi_head_attention_block_int8(x_q, w, attn_out_q);
    feedforward_block_int8(attn_out_q, w, x_q);
}

// Positional embedding addition
static void apply_positional_embedding_int8(
    qint8_t x_q[SEQ_LEN][EMBED_DIM],
    const PositionalEmbeddingWeightsInt8 &w
) {
#pragma HLS INLINE off

POS_I:
    for (int i = 0; i < SEQ_LEN; ++i) {
    POS_J:
        for (int j = 0; j < EMBED_DIM; ++j) {
#pragma HLS PIPELINE II=1
            float x_r   = dequantize_act(x_q[i][j]);
            float pe_r  = (float)w.table[i][j] * w.scale;
            float sum   = x_r + pe_r;
            x_q[i][j] = quantize_act(sum);
        }
    }
}

// Final MLP head
static void mlp_head_int8(
    const float pooled[EMBED_DIM],
    const MLPWeightsInt8 &w,
    float out[FINAL_NEURONS]
) {
#pragma HLS INLINE off

    // Hidden activations
    qint8_t pooled_q[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=pooled_q complete

    for (int j = 0; j < EMBED_DIM; ++j) {
#pragma HLS PIPELINE II=1
        pooled_q[j] = quantize_act(pooled[j]);
    }

    qint8_t hidden_q[MLP_UNITS];
#pragma HLS ARRAY_PARTITION variable=hidden_q complete

    // Dense1
    dense_int8_vec(
        &pooled_q[0],
        &w.W1[0][0],
        w.b1,
        w.W1_scale,
        &hidden_q[0],
        EMBED_DIM,
        MLP_UNITS
    );

    // ReLU
MLP_RELU_INT8:
    for (int i = 0; i < MLP_UNITS; ++i) {
#pragma HLS PIPELINE II=1
        hidden_q[i] = relu_q(hidden_q[i]);
    }

    // Dense2
    qint8_t out_q[FINAL_NEURONS];
#pragma HLS ARRAY_PARTITION variable=out_q complete

    dense_int8_vec(
        &hidden_q[0],
        &w.W2[0][0],
        w.b2,
        w.W2_scale,
        &out_q[0],
        MLP_UNITS,
        FINAL_NEURONS
    );

    // Convert final output to float
FINAL_DEQ:
    for (int o = 0; o < FINAL_NEURONS; ++o) {
#pragma HLS PIPELINE II=1
        out[o] = dequantize_act(out_q[o]);
    }
}

// --------------------------------------------------------------------
// Weight initialization
// --------------------------------------------------------------------

void init_model_weights_int8(TransformerModelInt8 &model) {
    // --------------------------------------------------
    // Positional encoding (INT8)
    // --------------------------------------------------
    {
        model.pos_emb.scale =
            w_0017_model_positional_encoding_embedding_embedding_lookup_scale;

        const int8_t *src =
            w_0017_model_positional_encoding_embedding_embedding_lookup_q;
        for (int i = 0; i < SEQ_LEN; ++i) {
            for (int d = 0; d < EMBED_DIM; ++d) {
                model.pos_emb.table[i][d] = (qint8_t)(*src++);
            }
        }
    }

    // --------------------------------------------------
    // LayerNorm weights (kept in float, same as float model)
    // --------------------------------------------------
    {
        // Encoder 0 - LN1
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.encoders[0].ln1_gamma[d] =
                w_0038_model_layer_normalization_batchnorm_mul_ReadVariableOp[d];
            model.encoders[0].ln1_beta[d]  =
                w_0039_model_layer_normalization_batchnorm_ReadVariableOp[d];
        }

        // Encoder 0 - LN2
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.encoders[0].ln2_gamma[d] =
                w_0032_model_layer_normalization_1_batchnorm_mul_ReadVariableOp[d];
            model.encoders[0].ln2_beta[d]  =
                w_0033_model_layer_normalization_1_batchnorm_ReadVariableOp[d];
        }

        // Encoder 1 - LN1
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.encoders[1].ln1_gamma[d] =
                w_0034_model_layer_normalization_2_batchnorm_mul_ReadVariableOp[d];
            model.encoders[1].ln1_beta[d]  =
                w_0035_model_layer_normalization_2_batchnorm_ReadVariableOp[d];
        }

        // Encoder 1 - LN2
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.encoders[1].ln2_gamma[d] =
                w_0036_model_layer_normalization_3_batchnorm_mul_ReadVariableOp[d];
            model.encoders[1].ln2_beta[d]  =
                w_0037_model_layer_normalization_3_batchnorm_ReadVariableOp[d];
        }
    }

    // --------------------------------------------------
    // MHA block 0 (encoder 0) - INT8
    // --------------------------------------------------
    {
        // Q, K, V, O scales
        model.encoders[0].W_q_scale =
            w_0007_model_multi_head_attention_query_einsum_Einsum_scale;
        model.encoders[0].W_k_scale =
            w_0006_model_multi_head_attention_key_einsum_Einsum1_scale;
        model.encoders[0].W_v_scale =
            w_0011_model_multi_head_attention_value_einsum_Einsum_scale;
        model.encoders[0].W_o_scale =
            w_0012_model_multi_head_attention_attention_output_einsum_Einsum2_scale;

        // Q: [EMBED_DIM][HEAD_SIZE], src shape (30,128)
        {
            const int8_t *src =
                w_0007_model_multi_head_attention_query_einsum_Einsum_q;
            for (int r = 0; r < EMBED_DIM; ++r) {
                for (int c = 0; c < HEAD_SIZE; ++c) {
                    model.encoders[0].W_q[r][c] = (qint8_t)(*src++);
                }
            }
        }
        // K
        {
            const int8_t *src =
                w_0006_model_multi_head_attention_key_einsum_Einsum1_q;
            for (int r = 0; r < EMBED_DIM; ++r) {
                for (int c = 0; c < HEAD_SIZE; ++c) {
                    model.encoders[0].W_k[r][c] = (qint8_t)(*src++);
                }
            }
        }
        // V
        {
            const int8_t *src =
                w_0011_model_multi_head_attention_value_einsum_Einsum_q;
            for (int r = 0; r < EMBED_DIM; ++r) {
                for (int c = 0; c < HEAD_SIZE; ++c) {
                    model.encoders[0].W_v[r][c] = (qint8_t)(*src++);
                }
            }
        }
        // O: [HEAD_SIZE][EMBED_DIM], src shape (128,30)
        {
            const int8_t *src =
                w_0012_model_multi_head_attention_attention_output_einsum_Einsum2_q;
            for (int r = 0; r < HEAD_SIZE; ++r) {
                for (int c = 0; c < EMBED_DIM; ++c) {
                    model.encoders[0].W_o[r][c] = (qint8_t)(*src++);
                }
            }
        }

        // Biases for MHA 0 - Float
        for (int i = 0; i < HEAD_SIZE; ++i) {
            model.encoders[0].b_q[i] =
                w_0046_model_multi_head_attention_query_add_ReadVariableOp[i];
            model.encoders[0].b_k[i] =
                w_0045_model_multi_head_attention_key_add_ReadVariableOp[i];
            model.encoders[0].b_v[i] =
                w_0047_model_multi_head_attention_value_add_ReadVariableOp[i];
        }
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.encoders[0].b_o[d] =
                w_0044_model_multi_head_attention_attention_output_add_ReadVariable[d];
        }
    }

    // --------------------------------------------------
    // MHA block 1 (encoder 1) - INT8
    // --------------------------------------------------
    {
        model.encoders[1].W_q_scale =
            w_0014_model_multi_head_attention_1_query_einsum_Einsum_scale;
        model.encoders[1].W_k_scale =
            w_0013_model_multi_head_attention_1_key_einsum_Einsum_scale;
        model.encoders[1].W_v_scale =
            w_0015_model_multi_head_attention_1_value_einsum_Einsum_scale;
        model.encoders[1].W_o_scale =
            w_0016_model_multi_head_attention_1_attention_output_einsum_Einsum_scale;

        // Q1
        {
            const int8_t *src =
                w_0014_model_multi_head_attention_1_query_einsum_Einsum_q;
            for (int r = 0; r < EMBED_DIM; ++r) {
                for (int c = 0; c < HEAD_SIZE; ++c) {
                    model.encoders[1].W_q[r][c] = (qint8_t)(*src++);
                }
            }
        }
        // K1
        {
            const int8_t *src =
                w_0013_model_multi_head_attention_1_key_einsum_Einsum_q;
            for (int r = 0; r < EMBED_DIM; ++r) {
                for (int c = 0; c < HEAD_SIZE; ++c) {
                    model.encoders[1].W_k[r][c] = (qint8_t)(*src++);
                }
            }
        }
        // V1
        {
            const int8_t *src =
                w_0015_model_multi_head_attention_1_value_einsum_Einsum_q;
            for (int r = 0; r < EMBED_DIM; ++r) {
                for (int c = 0; c < HEAD_SIZE; ++c) {
                    model.encoders[1].W_v[r][c] = (qint8_t)(*src++);
                }
            }
        }
        // O1
        {
            const int8_t *src =
                w_0016_model_multi_head_attention_1_attention_output_einsum_Einsum_q;
            for (int r = 0; r < HEAD_SIZE; ++r) {
                for (int c = 0; c < EMBED_DIM; ++c) {
                    model.encoders[1].W_o[r][c] = (qint8_t)(*src++);
                }
            }
        }

        // Biases for MHA 1 - Float
        for (int i = 0; i < HEAD_SIZE; ++i) {
            model.encoders[1].b_q[i] =
                w_0042_model_multi_head_attention_1_query_add_ReadVariableOp[i];
            model.encoders[1].b_k[i] =
                w_0041_model_multi_head_attention_1_key_add_ReadVariableOp[i];
            model.encoders[1].b_v[i] =
                w_0043_model_multi_head_attention_1_value_add_ReadVariableOp[i];
        }
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.encoders[1].b_o[d] =
                w_0040_model_multi_head_attention_1_attention_output_add_ReadVariab[d];
        }
    }

    // --------------------------------------------------
    // FFN block 0 (encoder 0) - INT8
    // --------------------------------------------------
    {
        // W1: src shape (FF_DIM, EMBED_DIM) = (128, 30)
        model.encoders[0].W_ff1_scale =
            w_0048_model_dense_Tensordot_MatMul1_scale;
        for (int in = 0; in < EMBED_DIM; ++in) {
            for (int out = 0; out < FF_DIM; ++out) {
                int idx = out * EMBED_DIM + in;
                model.encoders[0].W_ff1[in][out] =
                    (qint8_t)w_0048_model_dense_Tensordot_MatMul1_q[idx];
            }
        }
        // b1
        for (int i = 0; i < FF_DIM; ++i) {
            model.encoders[0].b_ff1[i] =
                w_0031_model_dense_BiasAdd_ReadVariableOp[i];
        }

        // W2: src shape (EMBED_DIM, FF_DIM) = (30, 128)
        model.encoders[0].W_ff2_scale =
            w_0049_model_dense_1_Tensordot_MatMul_scale;
        for (int in = 0; in < FF_DIM; ++in) {
            for (int out = 0; out < EMBED_DIM; ++out) {
                int idx = out * FF_DIM + in;
                model.encoders[0].W_ff2[in][out] =
                    (qint8_t)w_0049_model_dense_1_Tensordot_MatMul_q[idx];
            }
        }
        // b2
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.encoders[0].b_ff2[d] =
                w_0026_model_dense_1_BiasAdd_ReadVariableOp[d];
        }
    }

    // --------------------------------------------------
    // FFN block 1 (encoder 1) - INT8
    // --------------------------------------------------
    {
        // W1: src shape (FF_DIM, EMBED_DIM) = (128, 30)
        model.encoders[1].W_ff1_scale =
            w_0050_model_dense_2_Tensordot_MatMul_scale;
        for (int in = 0; in < EMBED_DIM; ++in) {
            for (int out = 0; out < FF_DIM; ++out) {
                int idx = out * EMBED_DIM + in;
                model.encoders[1].W_ff1[in][out] =
                    (qint8_t)w_0050_model_dense_2_Tensordot_MatMul_q[idx];
            }
        }
        // b1
        for (int i = 0; i < FF_DIM; ++i) {
            model.encoders[1].b_ff1[i] =
                w_0027_model_dense_2_BiasAdd_ReadVariableOp[i];
        }

        // W2: src shape (EMBED_DIM, FF_DIM) = (30, 128)
        model.encoders[1].W_ff2_scale =
            w_0051_model_dense_3_Tensordot_MatMul_scale;
        for (int in = 0; in < FF_DIM; ++in) {
            for (int out = 0; out < EMBED_DIM; ++out) {
                int idx = out * FF_DIM + in;
                model.encoders[1].W_ff2[in][out] =
                    (qint8_t)w_0051_model_dense_3_Tensordot_MatMul_q[idx];
            }
        }
        // b2
        for (int d = 0; d < EMBED_DIM; ++d) {
            model.encoders[1].b_ff2[d] =
                w_0028_model_dense_3_BiasAdd_ReadVariableOp[d];
        }
    }

    // --------------------------------------------------
    // Final MLP head - INT8
    // --------------------------------------------------
    {
        // W1: src shape (MLP_UNITS, EMBED_DIM) = (160, 30)
        model.mlp.W1_scale = w_0052_model_dense_4_MatMul_scale;
        for (int in = 0; in < EMBED_DIM; ++in) {
            for (int out = 0; out < MLP_UNITS; ++out) {
                int idx = out * EMBED_DIM + in;
                model.mlp.W1[in][out] =
                    (qint8_t)w_0052_model_dense_4_MatMul_q[idx];
            }
        }
        // b1
        for (int i = 0; i < MLP_UNITS; ++i) {
            model.mlp.b1[i] =
                w_0029_model_dense_4_BiasAdd_ReadVariableOp[i];
        }

        // W2: src shape (MLP_UNITS, FINAL_NEURONS) = (160, 1)
        model.mlp.W2_scale = w_0053_model_dense_5_kernel_scale;
        for (int i = 0; i < MLP_UNITS; ++i) {
            model.mlp.W2[i][0] =
                (qint8_t)w_0053_model_dense_5_kernel_q[i];
        }
        // b2
        model.mlp.b2[0] = w_0030_model_dense_5_BiasAdd_ReadVariableOp[0];
    }
}


// Top-level HLS function
extern "C" void trans_int8(
    const float input[SEQ_LEN][EMBED_DIM],
    float output[FINAL_NEURONS]
) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=IN  depth=1500
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=OUT depth=1

    static TransformerModelInt8 model;

    static bool model_initialized = false;
    if (!model_initialized) {
        init_model_weights_int8(model);
        model_initialized = true;
    }

    // 1) Quantize input to INT8
    qint8_t x_q[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=x_q cyclic factor=2 dim=2

Q_INPUT:
    for (int i = 0; i < SEQ_LEN; ++i) {
    Q_INPUT_J:
        for (int j = 0; j < EMBED_DIM; ++j) {
#pragma HLS PIPELINE II=1
            x_q[i][j] = quantize_act(input[i][j]);
        }
    }

    // 2) Add positional embedding
    apply_positional_embedding_int8(x_q, model.pos_emb);

ENC_BLOCKS:
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        encoder_block_forward_int8(x_q, model.encoders[b]);
    }

    float pooled[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=pooled cyclic factor=4

    global_average_pooling_1d(x_q, pooled);

    mlp_head_int8(pooled, model.mlp, output);
}
