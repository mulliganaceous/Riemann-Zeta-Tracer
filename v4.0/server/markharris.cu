/* 
 * Based on Mark Harris's Optimizing Parallel Reduction in CUDA.
 * However, we are now aiming to sum a 3-tensor into a matrix, rather than a vector into a scalar.
 */
/* Libraries to import */
#include <iostream>
#include <iomanip>

/* Preprocessing to change type and space considerations */
#define NUMTYPE long long
typedef NUMTYPE num;
#define BLOCKBITS 7
#define BLOCKSIZE (1 << BLOCKBITS)
#define WARPSIZE 32

#define _INTERVAL 1
#define _INITIAL 0
#define _FINAL (n >> 7) + 5

/*
 * Initialize the kernel to the overall indexing number
 */
__global__ void reset(num *g_array) {
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    g_array[idx] = idx;
    __syncthreads();
    /// printf("%d\t%d-%d\n", idx, blockIdx.x, threadIdx.x);
}

__global__ void interleaved(num *g_idata, num *g_odata) {
    extern __shared__ num sdata[];

    // Load one element from global to shared
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[idx];
    __syncthreads();

    // Reduction done in shared memory
    for (unsigned s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (s << 1) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result from shared to global
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void strided(num *g_idata, num *g_odata) {
    extern __shared__ num sdata[];

    // Load one element from global to shared
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[idx];
    __syncthreads();

    // Reduction done in shared memory
    for (unsigned s = 1; s < blockDim.x; s <<= 1) {
        unsigned idx2 = (s*tid << 1);
        if (idx2 < blockDim.x) {
            sdata[idx2] += sdata[idx2 + s];
        }
        __syncthreads();
    }

    // Write result from shared to global
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void sequential(num *g_idata, num *g_odata) {
    extern __shared__ num sdata[];

    // Load one element from global to shared
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[idx];
    __syncthreads();

    // Reduction done in shared memory
    for (unsigned s = (blockDim.x >> 1); s; s >>= 1) {
        if (tid < s) { // On the left half
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result from shared to global
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/* 
 * Ensures all threads are used, need to half the number of blocks
 * in the first call.
 */
__global__ void dosequential(num *g_idata, num *g_odata) {
    extern __shared__ num sdata[];

    // Load one element from global to shared
    unsigned tid = threadIdx.x;
    unsigned idx = (blockIdx.x*blockDim.x << 1) + threadIdx.x;
    sdata[tid] = g_idata[idx] + g_idata[idx + blockDim.x];
    __syncthreads();

    // Reduction done in shared memory
    for (unsigned s = (blockDim.x >> 1); s; s >>= 1) {
        if (tid < s) { // On the left half
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result from shared to global
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/* 
 * Perform a max-reduce on a single warp
 */
template<unsigned blockSize>
__device__ void warpReduce(volatile num *sdata, unsigned tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

/*
 * Unroll the threads
 */
__global__ void dounrolledsequential(num *g_idata, num *g_odata) {
    extern __shared__ num sdata[];

    // Load one element from global to shared
    unsigned tid = threadIdx.x;
    unsigned idx = (blockIdx.x*blockDim.x << 1) + threadIdx.x;
    sdata[tid] = g_idata[idx] + g_idata[idx + blockDim.x];
    __syncthreads();

    // Reduction done in shared memory, with last 32 with syncthreads obviated
    // for (unsigned s = (blockDim.x >> 1); s > 32; s >>= 1) {
    //     if (tid < s) { // On the left half
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }
    if (blockDim.x >= 512 && tid < 256){
        sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256 && tid < 128){
        sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128 && tid < 64){
        sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if (tid < 32) {
        warpReduce<BLOCKSIZE>(sdata, tid);
    }

    // Write result from shared to global
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main() {
    // List CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int deviceId;
    for (deviceId = 0; deviceId < deviceCount; deviceId++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);
        printf("Device %d: %s (v%d.%d)\n", deviceId, deviceProp.name, deviceProp.major, deviceProp.minor);
        printf("\tL2 cache size    : %d\n", deviceProp.l2CacheSize);
        printf("\tThread dimensions: %d,%d,%d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("\tMemory bus width : %d\n", deviceProp.memoryBusWidth);
        printf("\tMultiprocessor   : %d\n", deviceProp.multiProcessorCount);
        printf("\tRegisters        : %d / %d\n", deviceProp.regsPerBlock, deviceProp.regsPerMultiprocessor);
        printf("\tShared memory    : %d / %d\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerMultiprocessor);
        printf("\tConstant memory  : %d\n", deviceProp.totalConstMem);
        printf("\tGlobal memory    : %lld\n", deviceProp.totalGlobalMem);
        printf("\tWarp size        : %d\n", deviceProp.warpSize);
        //  printf("\tCluster support  : %d\n", deviceProp.clusterLaunch);
    }

    // Auxiliary variables and hyperparameters
    cudaError_t errorstatus;

    // Perform pagelocked host memory
    const unsigned n = 1<<25;
    const size_t size = n*sizeof(num);

    // Allocate host memory (ideally a power of two)
    num *h_terms = (num *)malloc(size);

    // Allocate device memory
    num *d_input;
    num *d_output;
    errorstatus = cudaMalloc(&d_input, size);
    errorstatus = cudaMalloc(&d_output, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_input, h_terms, size, cudaMemcpyHostToDevice);

    // Run kernels
    reset<<<n >> BLOCKBITS, BLOCKSIZE>>>(d_input);
    errorstatus = cudaDeviceSynchronize();
    dounrolledsequential<<<n >> (BLOCKBITS + 1), BLOCKSIZE>>>(d_input, d_output);
    errorstatus = cudaDeviceSynchronize();
    dounrolledsequential<<<n >> (BLOCKBITS + 1), BLOCKSIZE>>>(d_input, d_output);
    errorstatus = cudaDeviceSynchronize();
    dounrolledsequential<<<n >> (1 + BLOCKBITS << 1), BLOCKSIZE>>>(d_output, d_output + (n >> (BLOCKBITS)));
    errorstatus = cudaDeviceSynchronize();
    dounrolledsequential<<<n >> (1 + BLOCKBITS << 1), BLOCKSIZE>>>(d_output + (n >> (BLOCKBITS)), d_output + 2*(n >> (BLOCKBITS)));
    errorstatus = cudaDeviceSynchronize();
    dounrolledsequential<<<n >> (1 + BLOCKBITS << 1), BLOCKSIZE>>>(d_output + 2*(n >> (BLOCKBITS)), d_output + 3*(n >> (BLOCKBITS)));
    errorstatus = cudaDeviceSynchronize();
    dounrolledsequential<<<n >> (1 + BLOCKBITS << 1), BLOCKSIZE>>>(d_output + 3*(n >> (BLOCKBITS)), d_output + 4*(n >> (BLOCKBITS)));
    errorstatus = cudaDeviceSynchronize();
    if (errorstatus) {
        // Check error status
        std::cout << cudaGetErrorString(errorstatus) << std::endl;
        return errorstatus;
    }

    // Copy vectors back to host memory
    cudaMemcpy(h_terms, d_output, size, cudaMemcpyDeviceToHost);

    // Output
    for (unsigned k = _INITIAL; k < _FINAL; k += _INTERVAL) {
        std::cout << "x(" << std::setw(5) << k << ") = \t" << h_terms[k];
        std::cout << "\t" << h_terms[k + (n >> (BLOCKBITS))];
        std::cout << "\t" << h_terms[k + 2*(n >> (BLOCKBITS))];
        std::cout << "\t" << h_terms[k + 3*(n >> (BLOCKBITS))];
        std::cout << "\t" << h_terms[k + 4*(n >> (BLOCKBITS))];
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_terms);

    return errorstatus;
}