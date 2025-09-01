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

// Frame visualization
#include <opencv2/opencv.hpp>

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
#include <opencv2/core/mat.hpp>

// Dimensions
#define WIDTH 256
#define HEIGHT 16384
#define ENTRIES (WIDTH*HEIGHT)
#define DEPTH 1024
#define CASCADE 1024
#define TERMS (DEPTH*CASCADE)
#define BATCHES 1000
#define MEMSIZE (sizeof(cuDoubleComplex) * WIDTH * HEIGHT * DEPTH)

// CUDA Code

// Kernel definition
/*
 * Compute the Riemann zeta function using the Dirichlet eta function
 * without using row-reduction.
 */
__global__ void zeta(cuDoubleComplex *d_plot, double x_ini, double y_ini, double x_res, double y_res)
{
    // Obtain pixel subcoordinates
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.y * blockDim.y;

    // Temporary variables
    cuDoubleComplex z = make_cuDoubleComplex(x_ini + idx/x_res, y_ini + idy/y_res);
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex temp;
    double smagnitude, angle;

    // Straightforward summation
    double terms = (y_ini + idy/y_res)*4;
    if (terms < 1024) {
        terms = 1024;
    }
    for (int n = 1; n <= terms; n++)
    {
        // Must code the exponentiation manually
        smagnitude = (-1 + ((n & 1) << 1)) / exp(cuCreal(z) * log((double)n));
        angle = cuCimag(z) * log((double)n);
        temp = make_cuDoubleComplex(smagnitude * cos(angle), -smagnitude * sin(angle));
        sum = cuCadd(sum, temp);
    }
    // Must code the coefficient manually
    temp = make_cuDoubleComplex(1 - cuCreal(z), -cuCimag(z)); // temp is now the complement of z
    smagnitude = exp(cuCreal(temp) * log(2.0));
    angle = cuCimag(temp) * log(2.0);
    temp = make_cuDoubleComplex(1 - smagnitude * cos(angle), -smagnitude * sin(angle));
    sum = cuCdiv(sum, temp);

    // Store the result
    d_plot[idx*width + idy] = sum;
}

/* TODO
 * Compute the Riemann Zeta function using the Dirichlet eta function, keeping terms separate.
 */
__global__ void zetaterms(cuDoubleComplex *d_plot, double x_ini, double y_ini, double x_res, double y_res)
{
    // Obtain pixel subcoordinates
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int width = gridDim.x * blockDim.x;
    int height = gridDim.y * blockDim.y;
    int depth = gridDim.z * blockDim.z;

    // Temporary variables
    cuDoubleComplex z = make_cuDoubleComplex(x_ini + idx * x_res, y_ini + idy * y_res);
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex temp;
    double smagnitude, angle;

    // Straightforward summation
    for (int k = 1; k <= CASCADE; k++)
    {
        // Determine n
        int n = k + idz*CASCADE;
        // Must code the exponentiation manually
        smagnitude = (-1 + ((n & 1) << 1)) / exp(cuCreal(z) * log((double)n));
        angle = cuCimag(z) * log((double)n);
        temp = make_cuDoubleComplex(smagnitude * cos(angle), -smagnitude * sin(angle));
        sum = cuCadd(sum, temp);
    }
    // Must code the coefficient manually
    temp = make_cuDoubleComplex(1 - cuCreal(z), -cuCimag(z)); // temp is now the complement of z
    smagnitude = exp(cuCreal(temp) * log(2.0));
    angle = cuCimag(temp) * log(2.0);
    temp = make_cuDoubleComplex(1 - smagnitude * cos(angle), -smagnitude * sin(angle));
    sum = cuCdiv(sum, temp);

    // Store the result, with individual terms represented as planes, and vertical lines as rows.
    d_plot[idz*height*width + idy + idx*height] = sum;
}

/* TODO
 * Perform warp reduction.
 */
__device__ cuDoubleComplex warpReduceSum(cuDoubleComplex val) {
    return val;
}

/* TODO
 * Perform parallel reduction to merge the depth terms. They never apply to contiguous ranges of memory,
 * as the z-coordinate (term number) is the most major.
 */
__global__ void depthmerge(cuDoubleComplex *d_plot) {
    // Obtain pixel subcoordinates
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int width = gridDim.x * blockDim.x;
    int height = gridDim.y * blockDim.y;
    int depth = gridDim.z * blockDim.z;

    // Temporary variables
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    // Straightforward summation
    for (int k = 0; k < depth; k++)
    {
        sum = cuCadd(sum, d_plot[idx*height*depth + idy*depth + k]);
    }

    // Store the result
    d_plot[idy * width + idx] = sum;
}

void cudaZeta(cuDoubleComplex *h_plot, double x_ini, double y_ini, double x_res, double y_res)
{
    // Determine size
    unsigned int width = WIDTH;
    unsigned int height = HEIGHT;
    unsigned int memsize = sizeof(cuDoubleComplex) * width * height;
    // Allocate device memory for the plot
    cuDoubleComplex *d_plot;
    cudaError_t status = cudaHostGetDevicePointer(&d_plot, h_plot, 0);
    getStatus(status, "Failed to allocate cudaMemcpy! ");
    // Perform the zeta computation
    zeta<<<dim3(WIDTH / 32, HEIGHT / 32), dim3(32, 32)>>>(d_plot, x_ini, y_ini, x_res, y_res);
    cudaDeviceSynchronize();
    // Free memory
    cudaFree(d_plot);
}

/*
 * Compute the Riemann zeta function using the Dirichlet eta function, keeping terms separate.
 * Then merge the terms by depth.
 * The result is an array which applicable values are spaced DEPTH entries apart.
 */
void cudaZetaDepth(cuDoubleComplex *h_plot, double x_ini, double y_ini, double x_res, double y_res)
{
    // Host and device-side memory allocation
    cuDoubleComplex *d_plot;
    cudaError_t status = cudaHostGetDevicePointer(&d_plot, h_plot, 0);
    getStatus(status, "Failed to allocate cudaMemcpy! ");
    // Perform the term by term computation
    zetaterms<<<dim3(WIDTH, HEIGHT, DEPTH), CASCADE>>>(d_plot, x_ini, y_ini, x_res, y_res);
    cudaDeviceSynchronize();
    // Merge the terms by depth (contiguous range of 1024 blocks)
    depthmerge<<<dim3(WIDTH, HEIGHT, DEPTH), 256>>>(d_plot);
    // Free memory
    cudaFree(d_plot);
}

// Server code adapted from Beej's Guide to Network Programming

void sigchld_handler(int s)
{
    int saved_errno = errno;
    while (waitpid(-1, NULL, WNOHANG) > 0)
        ;
    errno = saved_errno;
}

void *get_in_addr(struct sockaddr *sa)
{
    switch (sa->sa_family)
    {
    case AF_INET:
        return &(((struct sockaddr_in *)sa)->sin_addr);
    case AF_INET6:
        return &(((struct sockaddr_in6 *)sa)->sin6_addr);
    default:
        throw std::runtime_error("Unknown address family");
    }
}

int server() {
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
        while (1)
        {
            close(sockfd);
            cuDoubleComplex *h_plot;
            // Allocate host memory for the plot
            cudaError_t status = cudaMallocHost(&h_plot, MEMSIZE/DEPTH);
            getStatus(status, "Failed to allocate cudaMallocHost! ");
            cudaZeta(h_plot, 0, 0, 16, 16);
            for (int k = 0; k < BATCHES; k++)
            {
                long long bytes_to_send = sizeof(cuDoubleComplex) * WIDTH * HEIGHT;
                long long interval = bytes_to_send / BATCHES;
                printf("Preparing to send %lld bytes\n", bytes_to_send);
                ssize_t sent_bytes = send(new_fd, h_plot + (k * interval)/sizeof(cuDoubleComplex), interval, 0);
                sleep(0.1);
                if (sent_bytes == -1)
                {
                    printf("neg one\n");
                }
                else
                {
                    printf("Sent %ld bytes\n", sent_bytes);
                }
                sleep(1);
            }
        }
        close(new_fd);
    }
    return EXIT_SUCCESS;
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
        //  printf("\tCluster support  : %d\n", deviceProp.clusterLaunch);
    }

    for (int ini = 0; ini < 65536; ini += 256) {
        clock_t t0 = clock();
        cuDoubleComplex *h_plot;
        // Allocate host memory for the plot
        cudaError_t status = cudaMallocHost(&h_plot, MEMSIZE/DEPTH);
        getStatus(status, "Failed to allocate cudaMallocHost! ");
        cudaZeta(h_plot, -1.5, -1 + ini, 64, 64);
        cv::Mat3b image = cv::Mat3b::zeros(WIDTH, HEIGHT);
        for (int x = 0; x < WIDTH; x++)
        {
            for (int y = 0; y < HEIGHT; y++)
            {
                double2 zeta = h_plot[x*HEIGHT + y];
                double magnitude = cuCabs(zeta);
                double angle = atan2(cuCimag(zeta), cuCreal(zeta));
                double saturation = 1;
                if (abs(1 - magnitude) < 0.0625) {
                    saturation /= 3.;
                    magnitude /= 2;
                }
                if (abs(cuCreal(zeta)) < 0.0625) {
                    saturation /= 1.5;
                }
                if (abs(cuCimag(zeta)) < 0.0625) {
                    saturation /= 1.5;
                }
                if (x % 64 == 32 || y % 64 == 0) {
                    magnitude /= 16;
                    saturation = 0;
                }
                cv::Mat3f hls(cv::Vec3f(angle*180/M_PI, 
                            min(0.96, 1.0/(1 + 1.0/sqrt(magnitude))), 
                            saturation));
                cv::Mat3f bgr;
                cvtColor(hls, bgr, cv::COLOR_HLS2BGR); 
                // printf("(%3.2f %+3.2f)", cuCreal(zeta), cuCimag(zeta));
                image.at<cv::Vec3b>(x, y) = cv::Vec3b((int)(bgr.at<cv::Vec3f>(0,0)[0]*255), (int)(bgr.at<cv::Vec3f>(0,0)[1]*255), (int)(bgr.at<cv::Vec3f>(0,0)[2]*255));
            }
            // std::cout << std::endl;
        }
        std::stringstream ss;
        ss << std::setbase(10) << std::setw(4) << ini;
        std::string hexstr = ss.str();
        std::replace(hexstr.begin(), hexstr.end(), ' ', '0');
        cv::imwrite("Test" + hexstr + ".png", image);
        std::cout << "Generated plot starting from height " << hexstr << " in time " << (float)(clock() - t0)/CLOCKS_PER_SEC << "s." << std::endl;
    }
    // Construct server
    server();

    return EXIT_SUCCESS;
}
