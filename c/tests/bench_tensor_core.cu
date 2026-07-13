#include "../backend_cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double run(ColiCudaTensor *g,ColiCudaTensor *u,ColiCudaTensor *d,
                  const float *x,float *y,int rows,int iterations,int mode){
    ColiCudaTensor *gs[1]={g},*us[1]={u},*ds[1]={d}; int rs[1]={rows};
    if(mode==2){setenv("COLI_CUDA_TC_INT4","1",1);setenv("COLI_CUDA_TC_MIN_ROWS","1",1);}
    else unsetenv("COLI_CUDA_TC_INT4");
    setenv("COLI_CUDA_W4_PACKED",mode==0?"0":"1",1);
    if(!coli_cuda_expert_group(gs,us,ds,rs,1,y,x))std::exit(2);
    auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<iterations;i++)if(!coli_cuda_expert_group(gs,us,ds,rs,1,y,x))std::exit(2);
    auto end=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::milli>(end-begin).count()/iterations;
}

static double run_host(const void *g,const void *u,const void *d,
                       const float *gs,const float *us,const float *ds,
                       const float *x,float *y,int rows,int iterations,int D,int I,int device){
    const void *ga[1]={g},*ua[1]={u},*da[1]={d};
    const float *gsa[1]={gs},*usa[1]={us},*dsa[1]={ds};
    int rs[1]={rows},fmt[1]={2},ids[8];float weights[8];
    for(int r=0;r<rows;r++){ids[r]=r;weights[r]=1.f;}
    unsetenv("COLI_CUDA_TC_INT4");
    if(!coli_cuda_expert_group_host_accum(ga,ua,da,gsa,usa,dsa,fmt,fmt,fmt,
        rs,1,y,x,rows,ids,weights,D,I,device))std::exit(2);
    auto begin=std::chrono::steady_clock::now();
    for(int n=0;n<iterations;n++)if(!coli_cuda_expert_group_host_accum(
        ga,ua,da,gsa,usa,dsa,fmt,fmt,fmt,rs,1,y,x,rows,ids,weights,D,I,device))std::exit(2);
    auto end=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::milli>(end-begin).count()/iterations;
}

int main(){
    constexpr int D=6144,I=2048,O=8;
    int device=0;if(!coli_cuda_init(&device,1))return 77;
    std::vector<unsigned char> hidden((size_t)I*D/2),down((size_t)D*I/2);
    std::vector<float> hs(I),ds(D),x((size_t)O*D),a((size_t)O*D),b((size_t)O*D),c((size_t)O*D);
    for(size_t i=0;i<hidden.size();i++)hidden[i]=(unsigned char)((i*17+29)&255);
    for(size_t i=0;i<down.size();i++)down[i]=(unsigned char)((i*13+41)&255);
    for(int i=0;i<I;i++)hs[i]=0.006f+(i%11)*0.0002f;
    for(int i=0;i<D;i++)ds[i]=0.006f+(i%7)*0.0002f;
    for(size_t i=0;i<x.size();i++)x[i]=std::sin((float)(i+1)*0.013f)*2.f;
    ColiCudaTensor *g=nullptr,*u=nullptr,*d=nullptr;
    if(!coli_cuda_tensor_upload(&g,hidden.data(),hs.data(),2,D,I,device)||
       !coli_cuda_tensor_upload(&u,hidden.data(),hs.data(),2,D,I,device)||
       !coli_cuda_tensor_upload(&d,down.data(),ds.data(),2,I,D,device))return 2;
    for(int rows: {1,2,4,8}){
        double scalar=run(g,u,d,x.data(),a.data(),rows,3,0);
        double packed=run(g,u,d,x.data(),b.data(),rows,3,1);
        double tc=run(g,u,d,x.data(),c.data(),rows,3,2);
        double streamed=run_host(hidden.data(),hidden.data(),down.data(),hs.data(),hs.data(),ds.data(),
                                 x.data(),c.data(),rows,10,D,I,device);
        double pe=0,te=0,ref=0;for(int i=0;i<rows*D;i++){double p=b[i]-a[i],t=c[i]-a[i];pe+=p*p;te+=t*t;ref+=(double)a[i]*a[i];}
        std::printf("rows=%d scalar_ms=%.3f packed_ms=%.3f packed_speedup=%.3fx packed_rms=%.7f tensor_ms=%.3f tensor_speedup=%.3fx tensor_rms=%.5f streamed_ms=%.3f\n",
                    rows,scalar,packed,scalar/packed,std::sqrt(pe/(ref+1e-20)),tc,scalar/tc,std::sqrt(te/(ref+1e-20)),streamed);
    }
    coli_cuda_tensor_free(g);coli_cuda_tensor_free(u);coli_cuda_tensor_free(d);coli_cuda_shutdown();
}
