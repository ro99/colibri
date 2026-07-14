/* Streamed-expert cache ownership is bidirectional: an ESlot names exactly one
 * preallocated CUDA slot and that slot names the same ESlot.  Exercise the real
 * glm.c helpers, especially the ws[] -> per-layer-LRU struct swap that used to
 * leave stale metadata able to steal ownership by matching tensor handles. */
#define main coli_glm_main_unused
#include "../glm.c"
#undef main

#define CHECK(x) do{ if(!(x)){ \
    fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; \
} }while(0)

static ColiCudaTensor *fake_tensor(uintptr_t n){ return (ColiCudaTensor*)n; }

int main(void){
    Model m; CudaCacheSlot slots[2]; ESlot ws,dst,other,stale;
    memset(&m,0,sizeof(m)); memset(slots,0,sizeof(slots));
    memset(&ws,0,sizeof(ws)); memset(&dst,0,sizeof(dst));
    memset(&other,0,sizeof(other)); memset(&stale,0,sizeof(stale));

    g_cuda_ndev=2; g_cuda_devices[0]=3; g_cuda_devices[1]=7;
    slots[0].g=fake_tensor(0x101); slots[0].u=fake_tensor(0x102); slots[0].d=fake_tensor(0x103);
    slots[0].device=3; slots[0].bytes=100;
    slots[1].g=fake_tensor(0x201); slots[1].u=fake_tensor(0x202); slots[1].d=fake_tensor(0x203);
    slots[1].device=7; slots[1].bytes=200;
    m.cuda_cache_arena[0]=(CudaCacheArena){&slots[0],1,3};
    m.cuda_cache_arena[1]=(CudaCacheArena){&slots[1],1,7};
    m.cuda_cache_slots=2; m.cuda_cache_limit=300;

    CHECK(cuda_cache_bind(&m,&ws,&slots[0]));
    CHECK(cuda_cache_validate(&m));
    CHECK(m.cuda_cache_bytes==100 && m.cuda_cache_by_device[0]==100);

    /* This is the exact promotion sequence in moe(): after the value swap the
     * destination carries the back-pointer, but the slot still owns ws. */
    { ESlot tmp=dst; dst=ws; ws=tmp; }
    CHECK(cuda_cache_rebind(&m,&ws,&dst));
    CHECK(cuda_cache_slot_for_entry(&m,&dst)==&slots[0]);
    CHECK(cuda_cache_validate(&m));

    /* A stale struct copy has identical tensor handles and a copied back-pointer.
     * It must not hijack or debit the slot owned by dst. */
    stale=dst;
    CHECK(cuda_cache_slot_for_entry(&m,&stale)==NULL);
    CHECK(!cuda_cache_drop(&m,&stale));
    CHECK(!stale.cuda_cache && stale.cuda_cache_slot==NULL);
    CHECK(slots[0].owner==&dst && m.cuda_cache_bytes==100);
    CHECK(cuda_cache_validate(&m));

    CHECK(cuda_cache_bind(&m,&other,&slots[1]));
    CHECK(!cuda_cache_bind(&m,&ws,&slots[1]));
    dst.cuda_cache_used=2; other.cuda_cache_used=1;
    CHECK(cuda_cache_lru(&m)==&slots[1]);
    CHECK(cuda_cache_drop(&m,&other));
    CHECK(m.cuda_cache_bytes==100 && m.cuda_cache_by_device[1]==0);
    CHECK(cuda_cache_drop(&m,&dst));
    CHECK(m.cuda_cache_bytes==0 && m.cuda_cache_by_device[0]==0);
    CHECK(cuda_cache_validate(&m));

    puts("cuda cache ownership: ok");
    return 0;
}
