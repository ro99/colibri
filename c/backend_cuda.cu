#include "backend_cuda.h"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

struct PendingTiming {
    cudaEvent_t start;
    cudaEvent_t end;
    uint64_t bytes;
};

typedef struct {
    int device;
    float *x, *y, *gate, *up, *x_src, *out;
    size_t x_cap, y_cap, gate_cap, up_cap, x_src_cap, out_cap;
    void *stage; size_t stage_cap;
    void *host_stage; size_t host_stage_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    int *row_ids; float *row_weights;
    size_t row_ids_cap, row_weights_cap;
    float *host_x,*host_y; size_t host_x_cap,host_y_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    cudaStream_t stream;
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
    uint64_t cache_uploads, cache_h2d_bytes;
    double cache_h2d_ms;
    std::vector<PendingTiming> cache_pending;
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static uint64_t g_group_h2d_bytes,g_group_d2h_bytes;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static std::mutex g_group_stats_mu;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

static int select_ctx(DeviceContext *ctx) {
    return ctx && cuda_ok(cudaSetDevice(ctx->device), "select device");
}

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

__global__ static void quant_matmul_w4(float *y,const float *x,const uint8_t *weights,
                                        const float *scales,int S,int I,int O){
    int o=blockIdx.x,s=blockIdx.y;if(s>=S)return;
    const uint8_t *row=weights+(size_t)o*((I+1)/2);const float *xs=x+(size_t)s*I;
    float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){
        uint8_t v=row[b];int a=v&15,z=v>>4;a=a&8?a-16:a;z=z&8?z-16:z;
        int i=b*2;sum+=xs[i]*(float)a;if(i+1<I)sum+=xs[i+1]*(float)z;
    }
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)s*O+o]=p[0]*scales[o];
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

__global__ static void fill_zero(float *dst, size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) dst[i]=0.0f;
}

__global__ static void gather_rows(float *dst,const float *src,const int *row_ids,int rows,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, n=(size_t)rows*D;
    if(i<n){ int row=(int)(i/(size_t)D), d=(int)(i%(size_t)D); dst[i]=src[(size_t)row_ids[row]*D+d]; }
}

__global__ static void weighted_scatter_add(float *dst,const float *src,const int *row_ids,
                                            const float *row_weights,int rows,int D){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, n=(size_t)rows*D;
    if(i<n){
        int row=(int)(i/(size_t)D), d=(int)(i%(size_t)D);
        atomicAdd(&dst[(size_t)row_ids[row]*D+d], src[i]*row_weights[row]);
    }
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;gate[z]=gp[0]*d.gs[o];up[z]=upv[0]*d.us[o];}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*(fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);ctx[(size_t)h*V+v]=a*(fmt?wscale[row]:1.f);}
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

static int reserve_pinned_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned weight staging allocation"))return 0;*cap=bytes;return 1;
}

static void drain_cache_pending(DeviceContext *ctx, int wait_all){
    if(!ctx||ctx->cache_pending.empty()) return;
    size_t keep=0;
    for(size_t i=0;i<ctx->cache_pending.size();i++){
        PendingTiming timing=ctx->cache_pending[i];
        cudaError_t st=wait_all?cudaEventSynchronize(timing.end):cudaEventQuery(timing.end);
        if(st==cudaSuccess){
            float ms=0;
            if(cuda_ok(cudaEventElapsedTime(&ms,timing.start,timing.end),"cache fill timing"))
                ctx->cache_h2d_ms+=ms;
            ctx->cache_h2d_bytes+=timing.bytes;
            cudaEventDestroy(timing.start);
            cudaEventDestroy(timing.end);
            continue;
        }
        if(!wait_all&&st==cudaErrorNotReady){
            ctx->cache_pending[keep++]=timing;
            continue;
        }
        cuda_ok(st,"cache fill completion");
        cudaEventDestroy(timing.start);
        cudaEventDestroy(timing.end);
    }
    ctx->cache_pending.resize(keep);
}

static int copy_tensor_async(DeviceContext *ctx, ColiCudaTensor *tensor,
                             const void *weights, const float *scales,
                             uint64_t *bytes){
    if(!ctx||!tensor||!weights||!bytes||!select_ctx(ctx)) return 0;
    size_t scale_bytes=tensor->fmt?(size_t)tensor->O*sizeof(float):0;
    if(!cuda_ok(cudaMemcpyAsync(tensor->weights,weights,tensor->weight_bytes,cudaMemcpyHostToDevice,ctx->stream),
        "cache fill weight upload")) return 0;
    if(tensor->fmt==2){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256,0,ctx->stream>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"cache fill int4 conversion")) return 0;
    }
    if(scale_bytes){
        if(!scales) return 0;
        if(!cuda_ok(cudaMemcpyAsync(tensor->scales,scales,scale_bytes,cudaMemcpyHostToDevice,ctx->stream),
            "cache fill scale upload")) return 0;
    }
    *bytes+=(uint64_t)(tensor->weight_bytes+scale_bytes);
    return 1;
}

static int build_group_desc(ColiCudaTensor *const *gates,
                            ColiCudaTensor *const *ups,
                            ColiCudaTensor *const *downs,
                            const int *rows, int count,
                            GroupDesc *host, int *device,
                            int *D, int *I, int *total,
                            int *max_rows, int *all_s4){
    if (!gates || !ups || !downs || !rows || !host || !device ||
        !D || !I || !total || !max_rows || !all_s4 || count < 1 || count > 64) return 0;
    ColiCudaTensor *first=gates[0];
    if(!first) return 0;
    *device=first->device; *D=first->I; *I=first->O; *total=0; *max_rows=0; *all_s4=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=*device||u->device!=*device||d->device!=*device||
           g->I!=*D||u->I!=*D||g->O!=*I||u->O!=*I||d->I!=*I||d->O!=*D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],*total};
        *all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        *total+=rows[c];
        if(rows[c]>*max_rows) *max_rows=rows[c];
    }
    return 1;
}

static int launch_group_kernels(DeviceContext *ctx, GroupDesc *dev,const GroupDesc *host,
                                int total, int count, int D, int I,
                                int max_rows, int all_s4){
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    tc=tc&&all_s4&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=host[c].rows>=tc_min;
    if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        }
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else{
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    g_group_calls=g_group_experts=g_group_rows=0;
    g_group_h2d_bytes=g_group_d2h_bytes=0;
    g_group_h2d_ms=g_group_kernel_ms=g_group_d2h_ms=0.0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
        ctx->cache_uploads=0;
        ctx->cache_h2d_bytes=0;
        ctx->cache_h2d_ms=0.0;
        ctx->cache_pending.clear();
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        drain_cache_pending(ctx,1);
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->x_src) cudaFree(ctx->x_src);
        if (ctx->out) cudaFree(ctx->out);
        if (ctx->stage) cudaFree(ctx->stage);
        if (ctx->host_stage) cudaFreeHost(ctx->host_stage);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if (ctx->row_ids) cudaFree(ctx->row_ids);
        if (ctx->row_weights) cudaFree(ctx->row_weights);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
        ctx->cache_pending.clear();
        ctx->x = ctx->y = ctx->gate = ctx->up = ctx->x_src = ctx->out = nullptr;
        ctx->stage = ctx->host_stage = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->row_ids=nullptr; ctx->row_weights=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->host_x=ctx->host_y=nullptr;ctx->stream=nullptr;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = ctx->x_src_cap = ctx->out_cap = 0;
        ctx->stage_cap = ctx->host_stage_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->row_ids_cap=ctx->row_weights_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
        ctx->cache_uploads=ctx->cache_h2d_bytes=0;
        ctx->cache_h2d_ms=0.0;
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms,
                                        uint64_t *h2d_bytes, uint64_t *d2h_bytes) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
    if(h2d_bytes) *h2d_bytes=g_group_h2d_bytes;
    if(d2h_bytes) *d2h_bytes=g_group_d2h_bytes;
}

extern "C" void coli_cuda_cache_stats(uint64_t *fills, uint64_t *h2d_bytes, double *h2d_ms) {
    uint64_t upload_count=0, upload_bytes=0;
    double upload_ms=0.0;
    for(int i=0;i<g_nctx;i++){
        DeviceContext *ctx=&g_ctx[i];
        if(!select_ctx(ctx)) continue;
        drain_cache_pending(ctx,1);
        upload_count+=ctx->cache_uploads;
        upload_bytes+=ctx->cache_h2d_bytes;
        upload_ms+=ctx->cache_h2d_ms;
    }
    if(fills) *fills=upload_count;
    if(h2d_bytes) *h2d_bytes=upload_bytes;
    if(h2d_ms) *h2d_ms=upload_ms;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2){offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    if(fmt==2)quant_matmul_w4<<<grid,256>>>(ctx->y,ctx->x,(const uint8_t*)t->weights,t->scales,S,I,O);
    else quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    dim3 hidden_grid((unsigned)I,(unsigned)S), output_grid((unsigned)D,(unsigned)S);
    if(gate->fmt==2)quant_matmul_w4<<<hidden_grid,256>>>(ctx->gate,ctx->x,(const uint8_t*)gate->weights,gate->scales,S,D,I);
    else quant_matmul<<<hidden_grid,256>>>(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D));
    if(up->fmt==2)quant_matmul_w4<<<hidden_grid,256>>>(ctx->up,ctx->x,(const uint8_t*)up->weights,up->scales,S,D,I);
    else quant_matmul<<<hidden_grid,256>>>(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D));
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    if(down->fmt==2)quant_matmul_w4<<<output_grid,256>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    else quant_matmul<<<output_grid,256>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I));
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_cache_fill(ColiCudaTensor *gate, ColiCudaTensor *up,
                                             ColiCudaTensor *down,
                                             const void *gate_weights, const void *up_weights,
                                             const void *down_weights,
                                             const float *gate_scales,
                                             const float *up_scales,
                                             const float *down_scales) {
    if(!gate||!up||!down||!gate_weights||!up_weights||!down_weights||
       gate->device!=up->device||gate->device!=down->device||
       gate->I!=up->I||gate->O!=up->O||
       down->I!=gate->O||down->O!=gate->I) return 0;
    DeviceContext *ctx=find_ctx(gate->device);
    if(!select_ctx(ctx)) return 0;
    uint64_t bytes=0;
    PendingTiming timing{};
    int track=0;
    if(cudaEventCreate(&timing.start)==cudaSuccess){
        if(cudaEventCreate(&timing.end)==cudaSuccess) track=1;
        else cudaEventDestroy(timing.start);
    }
    if(track) cudaEventRecord(timing.start,ctx->stream);
    if(!copy_tensor_async(ctx,gate,gate_weights,gate_scales,&bytes)||
       !copy_tensor_async(ctx,up,up_weights,up_scales,&bytes)||
       !copy_tensor_async(ctx,down,down_weights,down_scales,&bytes)){
        if(track){ cudaEventDestroy(timing.start); cudaEventDestroy(timing.end); }
        return 0;
    }
    ctx->cache_uploads++;
    if(track){
        timing.bytes=bytes;
        if(!cuda_ok(cudaEventRecord(timing.end,ctx->stream),"cache fill end event")){
            cudaEventDestroy(timing.start);
            cudaEventDestroy(timing.end);
            return 0;
        }
        ctx->cache_pending.push_back(timing);
    }else{
        ctx->cache_h2d_bytes+=bytes;
    }
    return cuda_ok(cudaGetLastError(),"cache fill launch");
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    int device=0,D=0,I=0,total=0,max_rows=0,all_s4=0;
    GroupDesc host[64];
    if(!build_group_desc(gates,ups,downs,rows,count,host,&device,&D,&I,&total,&max_rows,&all_s4)) return 0;
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")) profile=0;
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    if(!launch_group_kernels(ctx,dev,host,total,count,D,I,max_rows,all_s4)) return 0;
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_group_h2d_bytes+=(uint64_t)((size_t)count*sizeof(GroupDesc)+xb);
      g_group_d2h_bytes+=(uint64_t)xb; }
    return 1;
}

extern "C" int coli_cuda_expert_group_accum(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              float *out, const float *x,
                                              int S, const int *row_ids,
                                              const float *row_weights) {
    if (!gates || !ups || !downs || !rows || !out || !x || !row_ids || !row_weights || S < 1 || count < 1) return 0;
    int device=0,D=0,I=0,total=0,max_rows=0,all_s4=0;
    GroupDesc host[64];
    if(!build_group_desc(gates,ups,downs,rows,count,host,&device,&D,&I,&total,&max_rows,&all_s4)) return 0;
    for(int i=0;i<total;i++) if(row_ids[i]<0||row_ids[i]>=S) return 0;
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    size_t inb=(size_t)S*D*sizeof(float), outb=(size_t)S*D*sizeof(float);
    size_t rowb=(size_t)total*sizeof(int), wtb=(size_t)total*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->x_src,&ctx->x_src_cap,inb)||!reserve(&ctx->out,&ctx->out_cap,outb)||
       !reserve_bytes((void**)&ctx->row_ids,&ctx->row_ids_cap,rowb)||
       !reserve((float**)&ctx->row_weights,&ctx->row_weights_cap,wtb)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,inb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,outb)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")) profile=0;
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async) std::memcpy(ctx->host_x,x,inb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x_src,ctx->host_x,inb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x_src,x,inb,cudaMemcpyHostToDevice);
    cudaError_t copy_rows=async?cudaMemcpyAsync(ctx->row_ids,row_ids,rowb,cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->row_ids,row_ids,rowb,cudaMemcpyHostToDevice);
    cudaError_t copy_weights=async?cudaMemcpyAsync(ctx->row_weights,row_weights,wtb,cudaMemcpyHostToDevice,ctx->stream)
                                  :cudaMemcpy(ctx->row_weights,row_weights,wtb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group compact input upload")||
       !cuda_ok(copy_rows,"expert group row-id upload")||
       !cuda_ok(copy_weights,"expert group row-weight upload")) return 0;
    gather_rows<<<(unsigned)((((size_t)total*D)+255)/256),256,0,ctx->stream>>>(ctx->x,ctx->x_src,ctx->row_ids,total,D);
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    if(!launch_group_kernels(ctx,dev,host,total,count,D,I,max_rows,all_s4)) return 0;
    fill_zero<<<(unsigned)(((outb/sizeof(float))+255)/256),256,0,ctx->stream>>>(ctx->out,outb/sizeof(float));
    weighted_scatter_add<<<(unsigned)((((size_t)total*D)+255)/256),256,0,ctx->stream>>>(ctx->out,ctx->y,ctx->row_ids,ctx->row_weights,total,D);
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group compact synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->out,outb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(out,ctx->out,outb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group compact launch")||
       !cuda_ok(copy_y,"expert group compact output download"))return 0;
    if(async){ if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group compact synchronize"))return 0;
        std::memcpy(out,ctx->host_y,outb); }
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_group_h2d_bytes+=(uint64_t)((size_t)count*sizeof(GroupDesc)+inb+rowb+wtb);
      g_group_d2h_bytes+=(uint64_t)outb; }
    return 1;
}

extern "C" int coli_cuda_expert_group_host_accum(const void *const *gates,
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
                                                     int D, int I, int device) {
    if(!gates||!ups||!downs||!gate_scales||!up_scales||!down_scales||
       !gate_fmts||!up_fmts||!down_fmts||!rows||!out||!x||!row_ids||
       !row_weights||count<1||count>64||S<1||D<1||I<1) return 0;
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    int total=0,max_rows=0,all_s4=1;
    size_t weight_bytes=0;
    size_t goff[64],uoff[64],doff[64],gsoff[64],usoff[64],dsoff[64];
    for(int c=0;c<count;c++){
        int gf=gate_fmts[c],uf=up_fmts[c],df=down_fmts[c];
        size_t gb=row_bytes(gf,D)*(size_t)I,ub=row_bytes(uf,D)*(size_t)I;
        size_t db=row_bytes(df,I)*(size_t)D;
        if(!gates[c]||!ups[c]||!downs[c]||!gb||!ub||!db||rows[c]<1||
           ((gf||uf||df)&&(!gate_scales[c]||!up_scales[c]||!down_scales[c]))) return 0;
        goff[c]=weight_bytes; weight_bytes+=gb;
        uoff[c]=weight_bytes; weight_bytes+=ub;
        doff[c]=weight_bytes; weight_bytes+=db;
        all_s4&=gf==2&&uf==2&&df==2;
        total+=rows[c]; if(rows[c]>max_rows)max_rows=rows[c];
    }
    size_t scale_base=(weight_bytes+255u)&~(size_t)255u, stage_bytes=scale_base;
    for(int c=0;c<count;c++){
        gsoff[c]=stage_bytes; stage_bytes+=(size_t)I*sizeof(float);
        usoff[c]=stage_bytes; stage_bytes+=(size_t)I*sizeof(float);
        dsoff[c]=stage_bytes; stage_bytes+=(size_t)D*sizeof(float);
    }
    if(total<1) return 0;
    if(!reserve_pinned_bytes(&ctx->host_stage,&ctx->host_stage_cap,stage_bytes))return 0;
    if(!reserve_bytes(&ctx->stage,&ctx->stage_cap,stage_bytes)) return 0;
    uint8_t *hs=(uint8_t*)ctx->host_stage,*ds=(uint8_t*)ctx->stage;
    GroupDesc host[64]; int offset=0;
    for(int c=0;c<count;c++){
        size_t gb=row_bytes(gate_fmts[c],D)*(size_t)I;
        size_t ub=row_bytes(up_fmts[c],D)*(size_t)I;
        size_t db=row_bytes(down_fmts[c],I)*(size_t)D;
        std::memcpy(hs+goff[c],gates[c],gb); std::memcpy(hs+uoff[c],ups[c],ub);
        std::memcpy(hs+doff[c],downs[c],db);
        std::memcpy(hs+gsoff[c],gate_scales[c],(size_t)I*sizeof(float));
        std::memcpy(hs+usoff[c],up_scales[c],(size_t)I*sizeof(float));
        std::memcpy(hs+dsoff[c],down_scales[c],(size_t)D*sizeof(float));
        host[c]={ds+goff[c],ds+uoff[c],ds+doff[c],
                 (float*)(ds+gsoff[c]),(float*)(ds+usoff[c]),(float*)(ds+dsoff[c]),
                 gate_fmts[c],up_fmts[c],down_fmts[c],rows[c],offset};
        offset+=rows[c];
    }
    size_t xb=(size_t)total*D*sizeof(float),ib=(size_t)total*I*sizeof(float);
    size_t inb=(size_t)S*D*sizeof(float),outb=inb;
    size_t rowb=(size_t)total*sizeof(int),wtb=(size_t)total*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->x_src,&ctx->x_src_cap,inb)||!reserve(&ctx->out,&ctx->out_cap,outb)||
       !reserve_bytes((void**)&ctx->row_ids,&ctx->row_ids_cap,rowb)||
       !reserve((float**)&ctx->row_weights,&ctx->row_weights_cap,wtb)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,inb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,outb)) return 0;
    for(int r=0;r<total;r++) if(row_ids[r]<0||row_ids[r]>=S)return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile)for(int i=0;i<4;i++)if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event"))profile=0;
    if(profile)cudaEventRecord(ev[0],ctx->stream);
    std::memcpy(ctx->host_x,x,inb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->stage,ctx->host_stage,stage_bytes,cudaMemcpyHostToDevice,ctx->stream),"streamed expert upload")||
       !cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice,ctx->stream),"streamed expert descriptors")||
       !cuda_ok(cudaMemcpyAsync(ctx->x_src,ctx->host_x,inb,cudaMemcpyHostToDevice,ctx->stream),"streamed expert input")||
       !cuda_ok(cudaMemcpyAsync(ctx->row_ids,row_ids,rowb,cudaMemcpyHostToDevice,ctx->stream),"streamed expert row ids")||
       !cuda_ok(cudaMemcpyAsync(ctx->row_weights,row_weights,wtb,cudaMemcpyHostToDevice,ctx->stream),"streamed expert row weights")) return 0;
    if(all_s4)offset_to_signed_s4<<<(unsigned)((weight_bytes+255)/256),256,0,ctx->stream>>>((uint8_t*)ctx->stage,weight_bytes);
    gather_rows<<<(unsigned)((((size_t)total*D)+255)/256),256,0,ctx->stream>>>(ctx->x,ctx->x_src,ctx->row_ids,total,D);
    if(profile)cudaEventRecord(ev[1],ctx->stream);
    if(!launch_group_kernels(ctx,(GroupDesc*)ctx->group_desc,host,total,count,D,I,max_rows,all_s4))return 0;
    fill_zero<<<(unsigned)(((outb/sizeof(float))+255)/256),256,0,ctx->stream>>>(ctx->out,outb/sizeof(float));
    weighted_scatter_add<<<(unsigned)((((size_t)total*D)+255)/256),256,0,ctx->stream>>>(ctx->out,ctx->y,ctx->row_ids,ctx->row_weights,total,D);
    if(profile)cudaEventRecord(ev[2],ctx->stream);
    if(!cuda_ok(cudaGetLastError(),"streamed expert group launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->out,outb,cudaMemcpyDeviceToHost,ctx->stream),"streamed expert output")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"streamed expert synchronize"))return 0;
    std::memcpy(out,ctx->host_y,outb);
    if(profile){
        cudaEventRecord(ev[3],ctx->stream);cudaEventSynchronize(ev[3]);float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]);cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        {std::lock_guard<std::mutex>lock(g_group_stats_mu);g_group_h2d_ms+=a;g_group_kernel_ms+=b;g_group_d2h_ms+=c;}
        for(int i=0;i<4;i++)cudaEventDestroy(ev[i]);
    }
    {std::lock_guard<std::mutex>lock(g_group_stats_mu);
     g_group_calls++;g_group_experts+=(uint64_t)count;g_group_rows+=(uint64_t)total;
     g_group_h2d_bytes+=(uint64_t)(stage_bytes+(size_t)count*sizeof(GroupDesc)+inb+rowb+wtb);
     g_group_d2h_bytes+=(uint64_t)outb;}
    return 1;
}

extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if(!w||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}
