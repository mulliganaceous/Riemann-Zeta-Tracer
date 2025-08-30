// C++ standard
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define __STDC_FORMAT_MACROS

// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#define __CUDA_RUNTIME_H__

// Complex domain
#include <complex>
#include <cuComplex.h>
#define __COMPLEX

// Unix networking
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#define PORT "8081"
#define BACKLOG 16

// Helpers
#include "helper.cu"

// Dimensions
#define WIDTH 256
#define HEIGHT 240

// Kernel definition
/*
 * Compute the Riemann zeta function using the Dirichlet eta function
 * without using row-reduction.
 */
__global__ void zeta(cuDoubleComplex *d_plot) {
    // Obtain pixel subcoordinates
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    // Temporary variables
    cuDoubleComplex z = make_cuDoubleComplex(idx/32.f, 14 + idy/32.f);
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex temp;
    double smagnitude, angle;

    // Straightforward summation
    for (int n = 1; n <= 1000; n++) {
        // Must code the exponentiation manually
        smagnitude = (-1 + ((n & 1) << 1))/exp(cuCreal(z)*log((double)n));
        angle = cuCimag(z)*log((double)n);
        temp = make_cuDoubleComplex(smagnitude*cos(angle), -smagnitude*sin(angle));
        sum = cuCadd(sum, temp);
    }
    // Must code the coefficient manually
    temp = make_cuDoubleComplex(1 - cuCreal(z), -cuCimag(z)); // temp is now the complement of z
    smagnitude = exp(cuCreal(temp)*log(2.0));
    angle = cuCimag(temp)*log(2.0);
    temp = make_cuDoubleComplex(1 - smagnitude*cos(angle), -smagnitude*sin(angle));
    sum = cuCdiv(sum, temp);

    // Store the result
    d_plot[idy*width + idx] = sum;
}

cuDoubleComplex *cudaZeta() {
    // Determine size
    unsigned int width = WIDTH;
    unsigned int height = HEIGHT;
    unsigned int memsize = sizeof(cuDoubleComplex) * width * height;
    // Allocate host memory for the plot
    cuDoubleComplex *h_plot;
    cudaError_t status = cudaMallocHost(&h_plot, memsize);
    getStatus(status, "Failed to allocate cudaMallocHost! ");
    // Allocate device memory for the plot
    cuDoubleComplex *d_plot;
    cudaHostGetDevicePointer(&d_plot, h_plot, 0);
    getStatus(status, "Failed to allocate cudaMemcpy! ");
    // Perform the zeta computation
    zeta<<<dim3(WIDTH/16, HEIGHT/16), dim3(16, 16)>>>(d_plot);
    cudaDeviceSynchronize();
    // Free memory
    cudaFree(d_plot);
    return h_plot;
}

void sigchld_handler(int s) {
    int saved_errno = errno;
    while(waitpid(-1, NULL, WNOHANG) > 0);
    errno = saved_errno;
}

void *get_in_addr(struct sockaddr *sa) {
    switch (sa->sa_family) {
        case AF_INET:
            return &(((struct sockaddr_in*)sa)->sin_addr);
        case AF_INET6:
            return &(((struct sockaddr_in6*)sa)->sin6_addr);
        default:
            throw std::runtime_error("Unknown address family");
    }
}

int main()
{
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
        printf("\tCluster support  : %d\n", deviceProp.clusterLaunch);
    }

    // // Construct server
    int sockfd, new_fd;
    struct addrinfo hints, *servinfo, *p;
    struct sockaddr_storage their_addr;
    socklen_t sin_size;
    struct sigaction sa;
    int yes = 1;
    char s[INET6_ADDRSTRLEN];
    int rv;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    if ((rv = getaddrinfo(NULL, PORT, &hints, &servinfo)) != 0)
    {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        return EXIT_FAILURE;
    }

    for (p = servinfo; p != NULL; p = p->ai_next)
    {
        if ((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1)
        {
            perror("server: socket");
            continue;
        }
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1)
        {
            perror("setsockopt");
            return EXIT_FAILURE;
        }
        if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1)
        {
            close(sockfd);
            perror("server: bind");
            continue;
        }
        break;
    }

    freeaddrinfo(servinfo);
    if (p == NULL)
    {
        fprintf(stderr, "server: failed to bind\n");
        return EXIT_FAILURE;
    }
    if (listen(sockfd, BACKLOG) == -1)
    {
        perror("listen");
        return EXIT_FAILURE;
    }
    sa.sa_handler = sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGCHLD, &sa, NULL) == -1)
    {
        perror("sigaction");
        return EXIT_FAILURE;
    }

    printf("server: waiting for connections...\n");
    while (1)
    {
        sin_size = sizeof their_addr;
        new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
        if (new_fd == -1)
        {

            perror("accept");
            continue;
        }
        printf("server: the new fd is %d\n", new_fd);
        inet_ntop(their_addr.ss_family, get_in_addr((struct sockaddr *)&their_addr), s, sizeof s);
        printf("server: got connection from %s\n", s);
        // Compute the Riemann zeta function
        double2 *plot = cudaZeta();
        // Fork a child process to handle the request
        if (!fork())
        {
            close(sockfd);
            if (send(new_fd, plot, sizeof(plot) - 1, 0) == -1)
            {
                perror("send");
            }
            close(new_fd);
            exit(0);
        }
        close(new_fd);
    }

    return EXIT_SUCCESS;
}