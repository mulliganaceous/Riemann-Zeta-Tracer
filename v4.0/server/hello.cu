#include <stdio.h>

#define N (4*4*4*2*3*5)
#define M (N*sizeof(long long))

__global__ void devicecode(long long *d_idata, long long *d_odata) { 
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  int idy = blockDim.y*blockIdx.y + threadIdx.y;
  int idz = blockDim.z*blockIdx.z + threadIdx.z;

  int height = gridDim.x*blockDim.x;
  int width = gridDim.y*blockDim.y;
  int depth = gridDim.z*blockDim.z;

  int id = width*depth*idx + depth*idy + idz;
  d_idata[id] = 10000*(idx + 1) + 100*(idy + 1) + (idz + 1);
  __syncthreads();
  d_odata[id] = 100*threadIdx.x + 10*threadIdx.y + threadIdx.z;
  __syncthreads();
}
int main()
{
    long long *h_idata = (long long *)malloc(M);
    long long *h_odata  = (long long *)malloc(M);
    /* {HOSTSIDE INITIALIZATION */
    long long *d_idata = NULL;
    long long *d_odata = NULL;
    cudaMalloc((void **)&d_idata, M);
    cudaMalloc((void **)&d_odata, M);
    cudaMemcpy(d_idata, h_idata, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_odata, h_odata, M, cudaMemcpyHostToDevice);
    devicecode<<<dim3(2,3,5), dim3(4,4,4)>>>(d_idata, d_odata);
    cudaMemcpy(h_idata, d_idata, M, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_odata, d_odata, M, cudaMemcpyDeviceToHost); 
    cudaFree(d_idata);
    cudaFree(d_odata);
    for (int k = 0; k < N; k++) {
      printf("[%d] block: %06lld, thread: %03lld\n", k, h_idata[k], h_odata[k]);
    }

    free(h_idata);
    free(h_odata);
    return 0;
}