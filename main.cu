#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include "matrix.h"
#include <time.h>

#define CUDA(call){ \
    cudaError_t err=(call); \
    if(err!=0){ \
        fprintf(stderr,"%d -> CUDA(%s) error(%s) in function %s in file %s \n", \
            __LINE__,#call,cudaGetErrorString(err),__func__,__FILE__); \
        exit(1); \
    } \
}

#pragma pack(2)
struct bmp_header{
    uint16_t magic;
    uint32_t file_size;
    uint32_t app;
    uint32_t offset;
    uint32_t info_size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bits_per_pix;
    uint32_t comp;
    uint32_t comp_size;
    uint32_t xres;
    uint32_t yres;
    uint32_t cols_used;
    uint32_t imp_cols;
}head;

const int w=512;
const int h=w;


__device__ vec2 cxPow(vec2 a,vec2 b){
    float len=length(a);
    float theta=b[0]*atan(a[1]/a[0]);
    float phi=b[0]*log(len);
    return vec2(cos(theta)*cos(phi)-sin(theta)*sin(phi),
                cos(theta)*sin(phi)+sin(theta)*cos(phi))
                *pow(len,b[0])*exp(-b[1]*atan(a[1]/a[0]));
}

__device__ vec3 powertower(vec2 c){
    vec2 z=c;
    for (int i=0;i<100;i++){
        z=cxPow(c,z);
    }
    z=abs<float,2>(normalize(z));
    return vec3(z[0],0,z[1]);
}

__device__ vec3 pixel_main(vec2 uv){
    uv*=4.0f;
    return powertower(uv);
}


__global__ void kernel_main(char *bmp_buf){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int i=idx/w,j=idx%w;
    vec2 uv(j/(float)w,i/(float)h);
    uv*=2.0f;
    uv-=1.0f;

    vec3 col=clamp(pixel_main(uv),0.1f,1.0f);
    bmp_buf[4*idx+0]=col[0]*(1<<8-1);
    bmp_buf[4*idx+1]=col[1]*(1<<8-1);
    bmp_buf[4*idx+2]=col[2]*(1<<8-1);
    bmp_buf[4*idx+3]=0;
}

int main(){
    puts("cuda test");

    time_t start=time(0);

    memset(&head,0,sizeof(head));
    head.magic=0x4d42;
    head.app=0;
    head.offset=sizeof(bmp_header);
    head.info_size=40;
    head.width=w;
    head.height=h;
    head.planes=1;
    head.bits_per_pix=32;
    head.comp_size=w*h*head.bits_per_pix/8;
    printf("buf size: %d\n",head.comp_size);
    head.file_size=sizeof(bmp_header)+head.comp_size;

    
    FILE* bmp=fopen("img.bmp", "wb");
    assert(offsetof(bmp_header,file_size)==2);
    assert(sizeof(head)==54);



    int m=fwrite(&head,1,sizeof(head),bmp);
    assert(m==sizeof(head));

    char *bmp_buf=NULL;
    assert(head.comp_size!=0);
    CUDA(cudaMallocManaged(&bmp_buf,head.comp_size));
    assert(bmp_buf);

    int block_size=256;
    int num_blocks=(w*h+block_size-1)/block_size;
    kernel_main<<<num_blocks,block_size>>>(bmp_buf);
    CUDA(cudaDeviceSynchronize());
    assert(head.comp_size!=0);
    assert(fwrite(bmp_buf,1,head.comp_size,bmp)==head.comp_size);
    CUDA(cudaFree(bmp_buf));

    printf("took %.20f seconds\n",(double)(time(0)-start));
}