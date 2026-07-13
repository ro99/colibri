#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
int coli_cuda_init(const int *devices, int count);
void coli_cuda_shutdown(void);
int coli_cuda_device_count(void);
int coli_cuda_device_at(int index);
int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* device < 0 returns aggregate statistics for all configured devices. */
void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms,
                           uint64_t *h2d_bytes, uint64_t *d2h_bytes);
void coli_cuda_cache_stats(uint64_t *fills, uint64_t *h2d_bytes, double *h2d_ms);

/* Upload without executing, so capacity failures happen during model startup. */
int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2.
 * The first successful call uploads W and its row scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Overwrite an existing resident expert triple in place. The destination
 * tensors keep their device allocations; only the weight bytes and per-row
 * scales are refreshed on that device's stream. */
int coli_cuda_expert_cache_fill(ColiCudaTensor *gate, ColiCudaTensor *up,
                                ColiCudaTensor *down,
                                const void *gate_weights, const void *up_weights,
                                const void *down_weights,
                                const float *gate_scales,
                                const float *up_scales,
                                const float *down_scales);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);

/* Compact grouped-expert path. x/out are [S,D]. row_ids and row_weights contain
 * sum(rows) entries in call order, mapping each routed expert row back to the
 * source/output row. Outputs are accumulated on device before one D2H copy. */
int coli_cuda_expert_group_accum(ColiCudaTensor *const *gates,
                                 ColiCudaTensor *const *ups,
                                 ColiCudaTensor *const *downs,
                                 const int *rows, int count,
                                 float *out, const float *x,
                                 int S, const int *row_ids,
                                 const float *row_weights);

/* Transient counterpart for RAM-resident experts. The packed weights and
 * scales are staged into reusable device scratch, used by one grouped MLP,
 * and are not retained in the resident tensor set. This lets RAM act as the
 * expert backing store while CUDA remains the compute tier. */
int coli_cuda_expert_group_host_accum(const void *const *gates,
                                      const void *const *ups,
                                      const void *const *downs,
                                      const float *const *gate_scales,
                                      const float *const *up_scales,
                                      const float *const *down_scales,
                                      const int *gate_fmts,
                                      const int *up_fmts,
                                      const int *down_fmts,
                                      const int *rows, int count,
                                      float *out, const float *x,
                                      int S, const int *row_ids,
                                      const float *row_weights,
                                      int D, int I, int device);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

void coli_cuda_tensor_free(ColiCudaTensor *tensor);
size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
