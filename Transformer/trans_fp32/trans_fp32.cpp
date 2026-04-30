#include "trans_fp32.h"
#include "trans_weights_fp32.h"
#include <hls_math.h>

// --------------------------------------------------
// Helper functions
// --------------------------------------------------

// RelU
static inline float relu(float x) { 
#pragma HLS INLINE
    return (x > 0.0f) ? x : 0.0f; 
}

// LayerNorm
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
        // Mean - break dependency with pragma
        float mean = 0.0f;
        LN_MEAN:
        for (int j = 0; j < dim; ++j) {
#pragma HLS PIPELINE II=3
#pragma HLS DEPENDENCE variable=mean inter false
            mean += x[i][j];
        }
        mean /= (float)dim;

        // Variance - break dependency with pragma
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

// Softmax
static void softmax(const float* x, float* out, int len) {
#pragma HLS INLINE off
    
    // Find max
    float max_val = x[0];
    SOFTMAX_MAX:
    for (int i = 1; i < len; ++i) {
#pragma HLS PIPELINE II=1
        if (x[i] > max_val) max_val = x[i];
    }

    // Exp and sum - use pragma to allow variable read-after-write
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

// Dense with flattened loops and resource allocation
static void dense(
    const float* x,
    const float* W,
    const float* b,
    float* y,
    int len,
    int in_dim,
    int out_dim
) {
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable=y cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=W block factor=4

DENSE_ROWS:
    for (int i = 0; i < len; ++i) {
DENSE_COLS:
        for (int o = 0; o < out_dim; ++o) {
#pragma HLS PIPELINE II=2 // Adjust II based on resource usage and latency requirements
            float sum = (b != nullptr) ? b[o] : 0.0f;

DENSE_DOT:
            for (int d = 0; d < in_dim; ++d) {
#pragma HLS PIPELINE II=1
                sum += x[i * in_dim + d] * W[d * out_dim + o];
            }
            y[i * out_dim + o] = sum;
        }
    }
}

// Dense vector with explicit unrolling
static void dense_vec(
    const float* x,
    const float* W,
    const float* b,
    float* y,
    int in_dim,
    int out_dim
) {
#pragma HLS INLINE off

DENSE_VEC_OUT:
    for (int o = 0; o < out_dim; ++o) {
        // 4 independent accumulators break the long FP adder chain
        float acc0 = (b != nullptr) ? b[o] : 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

DENSE_VEC_IN:
        for (int d = 0; d < in_dim; ++d) {
#pragma HLS PIPELINE II=3
            float prod = x[d] * W[d * out_dim + o];

            // round-robin accumulate
            switch (d & 3) {
                case 0: acc0 += prod; break;
                case 1: acc1 += prod; break;
                case 2: acc2 += prod; break;
                default: acc3 += prod; break;
            }
        }

        // final small reduction (bounded depth)
        y[o] = (acc0 + acc1) + (acc2 + acc3);
    }
}


// Global pooling
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

// Multi-Head Attention
static void multi_head_attention_forward(
    float x_norm[][EMBED_DIM],
    const MHAWeights& w,
    float attn_out[][EMBED_DIM]
) {
#pragma HLS INLINE off
    
    // Projections with reduced arrays
    float Q[SEQ_LEN][HEAD_SIZE];
    float K[SEQ_LEN][HEAD_SIZE];
    float V[SEQ_LEN][HEAD_SIZE];
    
#pragma HLS ARRAY_PARTITION variable=Q cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=K cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=V cyclic factor=2 dim=2

    dense(&x_norm[0][0], &w.W_q[0][0], w.b_q, &Q[0][0], SEQ_LEN, EMBED_DIM, HEAD_SIZE);
    dense(&x_norm[0][0], &w.W_k[0][0], w.b_k, &K[0][0], SEQ_LEN, EMBED_DIM, HEAD_SIZE);
    dense(&x_norm[0][0], &w.W_v[0][0], w.b_v, &V[0][0], SEQ_LEN, EMBED_DIM, HEAD_SIZE);

    // Attention scores - REGISTERED partial reduction
    float scores[SEQ_LEN][SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=scores cyclic factor=4 dim=2

        const float inv_sqrt_dk = 1.0f / hls::sqrtf((float)HEAD_SIZE);

    ATTN_SCORE_I:
        for (int i = 0; i < SEQ_LEN; ++i) {
    ATTN_SCORE_J:
            for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=16

                // ---- Stage 1: partial sums buffer (32 entries, each sums 4 MACs) ----
                float psum[HEAD_SIZE/4]; // 128/4 = 32
#pragma HLS ARRAY_PARTITION variable=psum complete dim=1

    ATTN_SCORE_PS:
                for (int b = 0; b < HEAD_SIZE/4; ++b) {
                    // unroll 4 multiplies, but keep *addition local*
                    int d0 = 4*b + 0;
                    int d1 = 4*b + 1;
                    int d2 = 4*b + 2;
                    int d3 = 4*b + 3;

                    float t0 = Q[i][d0] * K[j][d0];
                    float t1 = Q[i][d1] * K[j][d1];
                    float t2 = Q[i][d2] * K[j][d2];
                    float t3 = Q[i][d3] * K[j][d3];

                    // small local adder tree (bounded)
                    psum[b] = (t0 + t1) + (t2 + t3);
                }

                // ---- Stage 2: reduce partial sums in a pipelined loop (registered) ----
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

    // Output projection
    dense(&context[0][0], &w.W_o[0][0], w.b_o, &attn_out[0][0], SEQ_LEN, HEAD_SIZE, EMBED_DIM);
}

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

static void encoder_block_forward(float x[][EMBED_DIM], const EncoderBlockWeights& w) {
#pragma HLS INLINE off
    
    // Residual save
    float x_input[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=x_input cyclic factor=2 dim=2
    
    ENC_SAVE1:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x_input[i][d] = x[i][d];
        }
    }

    // LN1
    layer_norm(x, w.ln1.gamma, w.ln1.beta, SEQ_LEN, EMBED_DIM);

    // MHA
    float attn_out[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=attn_out cyclic factor=2 dim=2
    
    multi_head_attention_forward(x, w.mha, attn_out);

    // Residual 1
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

    // LN2
    layer_norm(x, w.ln2.gamma, w.ln2.beta, SEQ_LEN, EMBED_DIM);

    // FFN
    float ffn_hidden[SEQ_LEN][FF_DIM];
#pragma HLS ARRAY_PARTITION variable=ffn_hidden cyclic factor=2 dim=2
    
    dense(&x[0][0], &w.ffn.W1[0][0], w.ffn.b1, &ffn_hidden[0][0], SEQ_LEN, EMBED_DIM, FF_DIM);

    ENC_RELU:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < FF_DIM; ++d) {
#pragma HLS PIPELINE II=1
            ffn_hidden[i][d] = relu(ffn_hidden[i][d]);
        }
    }

    float ffn_out[SEQ_LEN][EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=ffn_out cyclic factor=2 dim=2
    
    dense(&ffn_hidden[0][0], &w.ffn.W2[0][0], w.ffn.b2, &ffn_out[0][0], SEQ_LEN, FF_DIM, EMBED_DIM);

    // Residual 2
    ENC_RES2:
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
#pragma HLS PIPELINE II=1
            x[i][d] = x_residual[i][d] + ffn_out[i][d];
        }
    }
}

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


// Weight-copy
static inline void copy_matrix_2d(float dst[][HEAD_SIZE], const float* src, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[r][c] = src[r * cols + c];
}

static inline void copy_matrix_2d_head_out(float dst[][EMBED_DIM], const float* src, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[r][c] = src[r * cols + c];
}

static inline void copy_vector_1d(float* dst, const float* src, int len) {
    for (int i = 0; i < len; ++i) dst[i] = src[i];
}

static inline void copy_vector_1d_from_2d(float* dst, const float* src, int len) {
    for (int i = 0; i < len; ++i) dst[i] = src[i];
}


// Weight initialization from trans_weights_fp32.h
void init_model_weights(TransformerModel& model) {
    // Positional encoding
    {
        const float* src = w_0017_model_positional_encoding_embedding_embedding_lookup;
        for (int i = 0; i < SEQ_LEN; ++i)
            for (int d = 0; d < EMBED_DIM; ++d)
                model.pos_emb.position_embedding[i][d] = *src++;
    }

    // LayerNorm weights
    {
        copy_vector_1d(model.encoders[0].ln1.gamma,
                       w_0038_model_layer_normalization_batchnorm_mul_ReadVariableOp, EMBED_DIM);
        copy_vector_1d(model.encoders[0].ln1.beta,
                       w_0039_model_layer_normalization_batchnorm_ReadVariableOp, EMBED_DIM);

        copy_vector_1d(model.encoders[0].ln2.gamma,
                       w_0032_model_layer_normalization_1_batchnorm_mul_ReadVariableOp, EMBED_DIM);
        copy_vector_1d(model.encoders[0].ln2.beta,
                       w_0033_model_layer_normalization_1_batchnorm_ReadVariableOp, EMBED_DIM);

        copy_vector_1d(model.encoders[1].ln1.gamma,
                       w_0034_model_layer_normalization_2_batchnorm_mul_ReadVariableOp, EMBED_DIM);
        copy_vector_1d(model.encoders[1].ln1.beta,
                       w_0035_model_layer_normalization_2_batchnorm_ReadVariableOp, EMBED_DIM);

        copy_vector_1d(model.encoders[1].ln2.gamma,
                       w_0036_model_layer_normalization_3_batchnorm_mul_ReadVariableOp, EMBED_DIM);
        copy_vector_1d(model.encoders[1].ln2.beta,
                       w_0037_model_layer_normalization_3_batchnorm_ReadVariableOp, EMBED_DIM);
    }

    // MHA block 0
    {
        copy_matrix_2d(model.encoders[0].mha.W_q,
                       w_0007_model_multi_head_attention_query_einsum_Einsum, EMBED_DIM, HEAD_SIZE);
        copy_matrix_2d(model.encoders[0].mha.W_k,
                       w_0006_model_multi_head_attention_key_einsum_Einsum1, EMBED_DIM, HEAD_SIZE);
        copy_matrix_2d(model.encoders[0].mha.W_v,
                       w_0011_model_multi_head_attention_value_einsum_Einsum, EMBED_DIM, HEAD_SIZE);
        copy_matrix_2d_head_out(model.encoders[0].mha.W_o,
                                w_0012_model_multi_head_attention_attention_output_einsum_Einsum2,
                                HEAD_SIZE, EMBED_DIM);

        copy_vector_1d_from_2d(model.encoders[0].mha.b_q,
                               w_0046_model_multi_head_attention_query_add_ReadVariableOp, HEAD_SIZE);
        copy_vector_1d_from_2d(model.encoders[0].mha.b_k,
                               w_0045_model_multi_head_attention_key_add_ReadVariableOp, HEAD_SIZE);
        copy_vector_1d_from_2d(model.encoders[0].mha.b_v,
                               w_0047_model_multi_head_attention_value_add_ReadVariableOp, HEAD_SIZE);

        copy_vector_1d(model.encoders[0].mha.b_o,
                       w_0044_model_multi_head_attention_attention_output_add_ReadVariable, EMBED_DIM);
    }

    // MHA block 1
    {
        copy_matrix_2d(model.encoders[1].mha.W_q,
                       w_0014_model_multi_head_attention_1_query_einsum_Einsum, EMBED_DIM, HEAD_SIZE);
        copy_matrix_2d(model.encoders[1].mha.W_k,
                       w_0013_model_multi_head_attention_1_key_einsum_Einsum, EMBED_DIM, HEAD_SIZE);
        copy_matrix_2d(model.encoders[1].mha.W_v,
                       w_0015_model_multi_head_attention_1_value_einsum_Einsum, EMBED_DIM, HEAD_SIZE);
        copy_matrix_2d_head_out(model.encoders[1].mha.W_o,
                                w_0016_model_multi_head_attention_1_attention_output_einsum_Einsum,
                                HEAD_SIZE, EMBED_DIM);

        copy_vector_1d_from_2d(model.encoders[1].mha.b_q,
                               w_0042_model_multi_head_attention_1_query_add_ReadVariableOp, HEAD_SIZE);
        copy_vector_1d_from_2d(model.encoders[1].mha.b_k,
                               w_0041_model_multi_head_attention_1_key_add_ReadVariableOp, HEAD_SIZE);
        copy_vector_1d_from_2d(model.encoders[1].mha.b_v,
                               w_0043_model_multi_head_attention_1_value_add_ReadVariableOp, HEAD_SIZE);

        copy_vector_1d(model.encoders[1].mha.b_o,
                       w_0040_model_multi_head_attention_1_attention_output_add_ReadVariab, EMBED_DIM);
    }

    // FFN block 0
    {
        for (int in = 0; in < EMBED_DIM; ++in) {
            for (int out = 0; out < FF_DIM; ++out) {
                model.encoders[0].ffn.W1[in][out] =
                    w_0048_model_dense_Tensordot_MatMul1[out * EMBED_DIM + in];
            }
        }
        copy_vector_1d(model.encoders[0].ffn.b1,
                       w_0031_model_dense_BiasAdd_ReadVariableOp, FF_DIM);

        for (int in = 0; in < FF_DIM; ++in) {
            for (int out = 0; out < EMBED_DIM; ++out) {
                model.encoders[0].ffn.W2[in][out] =
                    w_0049_model_dense_1_Tensordot_MatMul[out * FF_DIM + in];
            }
        }
        copy_vector_1d(model.encoders[0].ffn.b2,
                       w_0026_model_dense_1_BiasAdd_ReadVariableOp, EMBED_DIM);
    }

    // FFN block 1
    {
        for (int in = 0; in < EMBED_DIM; ++in) {
            for (int out = 0; out < FF_DIM; ++out) {
                model.encoders[1].ffn.W1[in][out] =
                    w_0050_model_dense_2_Tensordot_MatMul[out * EMBED_DIM + in];
            }
        }
        copy_vector_1d(model.encoders[1].ffn.b1,
                       w_0027_model_dense_2_BiasAdd_ReadVariableOp, FF_DIM);

        for (int in = 0; in < FF_DIM; ++in) {
            for (int out = 0; out < EMBED_DIM; ++out) {
                model.encoders[1].ffn.W2[in][out] =
                    w_0051_model_dense_3_Tensordot_MatMul[out * FF_DIM + in];
            }
        }
        copy_vector_1d(model.encoders[1].ffn.b2,
                       w_0028_model_dense_3_BiasAdd_ReadVariableOp, EMBED_DIM);
    }

    // Final MLP
    {
        for (int in = 0; in < EMBED_DIM; ++in) {
            for (int out = 0; out < MLP_UNITS; ++out) {
                model.mlp.W1[in][out] = w_0052_model_dense_4_MatMul[out * EMBED_DIM + in];
            }
        }
        copy_vector_1d(model.mlp.b1, w_0029_model_dense_4_BiasAdd_ReadVariableOp, MLP_UNITS);

        for (int i = 0; i < MLP_UNITS; ++i) {
            model.mlp.W2[i][0] = w_0053_model_dense_5_kernel[i];
        }
        model.mlp.b2[0] = w_0030_model_dense_5_BiasAdd_ReadVariableOp[0];
    }
}

// Top-level HLS function
extern "C" void transformer_hls(
    const float input[SEQ_LEN][EMBED_DIM],
    float output[FINAL_NEURONS]
) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=IN depth=1500
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=OUT depth=1

    static TransformerModel model;
    static bool initialized = false;

    if (!initialized) {
        init_model_weights(model);
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

    // Positional encoding
    apply_positional_embedding(x, model.pos_emb);

ENCODER_BLOCKS:
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        encoder_block_forward(x, model.encoders[b]);
    }

    // Global pooling
    float pooled[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=pooled cyclic factor=4
    
    global_average_pooling_1d(x, pooled);

    // MLP head
    mlp_forward(pooled, model.mlp, output);
}