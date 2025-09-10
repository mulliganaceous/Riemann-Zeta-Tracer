// Main
#define __MAIN__

// C++ standard
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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
#include <opencv2/freetype.hpp>

// Helpers
#include "helper.cu"
#include <opencv2/core/mat.hpp>

// Dimensions
#define WIDTH 1920
#define HEIGHT 1024
#define ENTRIES (WIDTH*HEIGHT)
#define DEPTH 16
#define CASCADE 1024
#define TERMS (DEPTH*CASCADE)
#define BATCHES 1000
#define MEMSIZE (sizeof(cuDoubleComplex) * WIDTH * HEIGHT * DEPTH)
#define IMGMEMSIZE (sizeof(unsigned char) * WIDTH * HEIGHT)

// CUDA Code

// Kernel definition
/*
 * Identity function
 */
__global__ void id(cuDoubleComplex *d_plot, double x_ini, double y_ini, double x_res, double y_res)
{
    // Obtain pixel subcoordinates
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.y * blockDim.y;

    // Temporary variables
    cuDoubleComplex z = make_cuDoubleComplex(x_ini + idx/x_res, y_ini + idy/y_res);
    d_plot[idx*width + idy] = z;
}

// Kernel definition
/*
 * Compute the Riemann zeta function using the Dirichlet eta function
 * without using row-reduction.
 */
__global__ void zeta(cuDoubleComplex *d_plot, cuDoubleComplex *d_input)
{
    // Obtain pixel subcoordinates
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.y * blockDim.y;

    // Temporary variables
    cuDoubleComplex z = d_input[idx*width + idy];
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex temp;
    double smagnitude, angle;

    // Straightforward summation
    int terms = cuCimag(d_input[0])*1.25 + sqrt(cuCimag(d_input[0])) + log(1 + cuCimag(d_input[0]));
    if (terms < 256) {
        terms = 256;
    }
    int n;
    for (n = 1; n <= terms; n++) {
        // Must code the exponentiation manually
        smagnitude = (-1 + ((n & 1) << 1)) / exp(cuCreal(z) * log((double)n));
        angle = cuCimag(z) * log((double)n);
        temp = make_cuDoubleComplex(smagnitude * cos(angle), -smagnitude * sin(angle));
        sum = cuCadd(sum, temp);
    }

    // Average out
    smagnitude = (-1 + ((n & 1) << 1)) / exp(cuCreal(z) * log((double)n));
    angle = cuCimag(z) * log((double)n);
    temp = make_cuDoubleComplex(smagnitude * cos(angle) / 2, -smagnitude * sin(angle) / 2);
    sum = cuCadd(sum, temp);

    // Must code the coefficient manually
    temp = make_cuDoubleComplex(1 - cuCreal(z), -cuCimag(z)); // temp is now the complement of z
    smagnitude = exp(cuCreal(temp) * log(2.0));
    angle = cuCimag(temp) * log(2.0);
    temp = make_cuDoubleComplex(1 - 1 * smagnitude * cos(angle), -1 * smagnitude * sin(angle));
    sum = cuCdiv(sum, temp);

    // Store the result
    d_plot[idx*width + idy] = sum;
}

/* TODO
 * Compute the Riemann Zeta function using the Dirichlet eta function, keeping terms separate.
 */
__global__ void zetaterms(cuDoubleComplex *d_plot, cuDoubleComplex *d_input)
{
    // Obtain pixel subcoordinates
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    int width = gridDim.x * blockDim.x;
    int height = gridDim.y * blockDim.y;
    int depth = gridDim.z * blockDim.z;

    // Temporary variables
    cuDoubleComplex z = d_input[idx*width + idy];
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex temp;
    double smagnitude, angle;

    // Straightforward summation
    for (int k = 1; k <= CASCADE; k++) {
        // Determine n
        int n = k + idz*CASCADE;
        // Must code the exponentiation manually
        // smagnitude = (-1 + ((n & 1) << 1)) / exp(cuCreal(z) * log((double)n));
        // angle = cuCimag(z) * log((double)n);
        // temp = make_cuDoubleComplex(smagnitude * cos(angle), -smagnitude * sin(angle));
        // sum = cuCadd(sum, temp);
        sum = make_cuDoubleComplex(n, -n);
    }

    // Average out, only applicable for the last block and thread by depth
    if (blockIdx.z == gridDim.z - 1 && threadIdx.z == blockDim.z - 1) {
        int n = 1 + depth*CASCADE;
        // smagnitude = (-1 + ((n & 1) << 1)) / exp(cuCreal(z) * log((double)n));
        // angle = cuCimag(z) * log((double)n);
        // temp = make_cuDoubleComplex(smagnitude * cos(angle) / 2, -smagnitude * sin(angle) / 2);
        // sum = cuCadd(sum, temp);
        sum = make_cuDoubleComplex(n, -n);
    }

    // Must code the coefficient manually
    // temp = make_cuDoubleComplex(1 - cuCreal(z), -cuCimag(z)); // temp is now the complement of z
    // smagnitude = exp(cuCreal(temp) * log(2.0));
    // angle = cuCimag(temp) * log(2.0);
    // temp = make_cuDoubleComplex(1 - smagnitude * cos(angle), -smagnitude * sin(angle));
    // sum = cuCdiv(sum, temp);

    // Store the result, with individual terms represented as planes, and vertical lines as rows.
    d_plot[idx*height*depth + idy*depth + idz] = sum;
}

/* TODO
 */
__device__ void warpReduce(volatile double *sdata, unsigned tid) {
    sdata[tid] = sdata[tid] + sdata[tid + 32]; 
    sdata[tid] = sdata[tid] + sdata[tid + 16];
    sdata[tid] = sdata[tid] + sdata[tid + 8];
    sdata[tid] = sdata[tid] + sdata[tid + 4];
    sdata[tid] = sdata[tid] + sdata[tid + 2];
}

/* TODO
 * Perform warp reduction.
 */
__device__ cuDoubleComplex warpReduceSum(cuDoubleComplex *g_idata, cuDoubleComplex *g_odata) {
    // Shared data is componentwise
    extern __shared__ double sdata[];

    // Load one element from global to shared. The sdata contains even-odd pairs.
    unsigned tid = threadIdx.x;
    unsigned idx = (blockIdx.x*blockDim.x << 1) + threadIdx.x;
    sdata[tid] = g_idata[idx].x + g_idata[idx + blockDim.x].x;
    sdata[tid + 1] = g_idata[idx].x + g_idata[idx + blockDim.x].x;
    __syncthreads();

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
        warpReduce(sdata, tid);
    }

    // Write result from shared to global
    if (tid == 0) {
        g_odata[blockIdx.x].x = sdata[0];
        g_odata[blockIdx.x].y = sdata[0];
    }
}

void cudaZeta(cuDoubleComplex *h_plot, double x_ini, double y_ini, double x_res, double y_res, cuDoubleComplex *h_input)
{
    clock_t t0 = clock();
    std::cout << "Generate plot starting from height " << y_ini << std::endl;

    // Allocate device memory for the plot
    cuDoubleComplex *d_plot, *d_input;
    cudaError_t status;
    status = cudaHostGetDevicePointer(&d_plot, h_plot, 0);
    getStatus(status, "Failed to allocate cudaMemcpy! ");
    status = cudaHostGetDevicePointer(&d_input, h_input, 0);
    getStatus(status, "Failed to allocate cudaMemcpy! ");
    // Perform the zeta computation
    cudaDeviceSynchronize();
    id<<<dim3(WIDTH, 1), dim3(1, HEIGHT)>>>(d_input, x_ini, y_ini, x_res, y_res);
    cudaDeviceSynchronize();
    zeta<<<dim3(WIDTH, 1), dim3(1, HEIGHT)>>>(d_plot, d_input);
    cudaDeviceSynchronize();
    // Free memory
    cudaFree(d_plot);
    cudaFree(d_input);

    std::cout << "Generated plot starting from height " << y_ini << " in time " << (float)(clock() - t0)/CLOCKS_PER_SEC << "s." << std::endl;
}

/*
 * Compute the Riemann zeta function using the Dirichlet eta function, keeping terms separate.
 * Then merge the terms by depth.
 * The result is an array which applicable values are spaced DEPTH entries apart.
 */
void cudaZetaDepth(cuDoubleComplex *h_plot, double x_ini, double y_ini, double x_res, double y_res, cuDoubleComplex *h_input)
{
    clock_t t0;

    // Host and device-side memory allocation
    cuDoubleComplex *d_plot, *d_input;
    getStatus(cudaHostGetDevicePointer(&d_plot, h_plot, 0), "(plot cube) Failed to allocate cudaMemcpy! ");
    getStatus(cudaHostGetDevicePointer(&d_input, h_input, 0), "(input cube) Failed to allocate cudaMemcpy! ");
    // Perform the term by term computation
    t0 = clock();
    std::cout << "Generate plot starting from height " << y_ini << std::endl;
    cudaDeviceSynchronize();
    id<<<dim3(WIDTH, 1), dim3(1, HEIGHT)>>>(d_input, x_ini, y_ini, x_res, y_res);
    cudaDeviceSynchronize();
    zetaterms<<<dim3(WIDTH, HEIGHT, 1), DEPTH>>>(d_plot, d_input);
    cudaDeviceSynchronize();
    // Merge the terms by depth (contiguous range of 1024 blocks)

    std::cout << "Generated plot starting from height " << y_ini << " in time " << (float)(clock() - t0)/CLOCKS_PER_SEC << "s." << std::endl;
    // Free memory
    cudaFree(d_plot);
    cudaFree(d_input);
}

__global__ void generate_phase_plot(unsigned char *d_image, cuDoubleComplex *d_plot, cuDoubleComplex *d_input, int unitsquare) {
    // Obtain pixel subcoordinates
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.y * blockDim.y;
    
    // Input and output
    double2 z = d_input[idx*HEIGHT + idy];
    double2 zeta = d_plot[idx*HEIGHT + idy];
    double magnitude = cuCabs(zeta);
    double angle = atan2(cuCimag(zeta), cuCreal(zeta));
    double lightness = magnitude;
    double csaturation = 0.36;

    // Cross and conversion
    if (angle < 0) {
        angle += 2*M_PI;
    }
    double slope = cuCreal(zeta)/cuCimag(zeta);
    double islope = cuCimag(zeta)/cuCreal(zeta);
    unsigned char cross = ((abs(cuCreal(zeta)) < 0.0625) || (abs(slope) < 0.015625))
                        + ((abs(cuCimag(zeta)) < 0.0625) || (abs(islope) < 0.015625));

    // Output gridlines
    switch (cross) {
        case 1:
            csaturation = 0.09;
            break;
        case 2:
            csaturation = 0;
            break;
    }
    // Checkerboarding
    if (magnitude < 32 && (((((int)(floor(zeta.x))) & 1) + (((int)(floor(zeta.y))) & 1)) & 1)) {
        csaturation = cross ? 0.25 : csaturation * 64./36.;
    }
    // Magnitude
    if (lightness > 65536) {
        lightness -= 24576; // Triangulous2
    }                
    else if (lightness > 32768) {
        lightness -= 12288; // Triangulous
    }                
    else if (lightness > 16384) {
        lightness -= 6144; // Volleo
    }                
    else if (lightness > 8192) {
        lightness -= 3072; // Whalend
    }                
    else if (lightness > 4096) {
        lightness -= 1536; // Terrence
    }
    else if (lightness > 2048) {
        lightness -= 768; // Triferatu
    }
    else if (lightness > 512) {
        lightness -= 384; // Triad
    }
    else if (lightness > 256) {
        lightness -= 192; // Threejay
    }
    else if (lightness > 128) {
        lightness -= 96; // 32 to 160
    }
    else if (lightness > 64) {
        lightness -= 48; // 16 to 80
    }
    else if (lightness > 32) {
        lightness -= 28; // 4 to 36
    }
    else if (lightness > 24) {
        lightness -= 21; // 3 to 11
    }
    else if (lightness > 16) {
        lightness -= 14; // 2 to to 10
    }
    else if (lightness > 8) {
        lightness -= 7.5 ; // 8/16 to 8.5
    }
    else if (lightness > 7) {
        lightness -= 6.5625; // 7/16
    }
    else if (lightness > 6) {
        lightness -= 5.625; // 6/16
    }
    else if (lightness > 5) {
        lightness -= 4.6875; // 5/16
    }
    else if (lightness > 4) {
        lightness -= 3.75; // 4/16
    }
    else if (lightness > 3) {
        lightness -= 2.8125; // 3/16
    }
    else if (lightness > 2) {
        lightness -= 1.875; // 2/16
    }
    else if (lightness > 1) {
        lightness -= 0.9375; // 1/16
    }

    // Output
    lightness = 0.96/(1 + 1.0/sqrtf(lightness)); // sigmoid
    // Unit circle
    if (magnitude < 1) {
        int sector = ((int)(angle*6/M_PI)) % 3;
        switch (sector) {
            case 2:
                csaturation = 0.16;
                break;
            case 1:
                csaturation = 0.09;
                break;
            case 0:
                csaturation = 0;
                break;
            default:
                csaturation = 1;
                break;
        }
    } // Antidiagonal
    else if (cross) {
        lightness /= 4;
        lightness += 0.375;
    }
    else {
        if (slope < 0) {
            csaturation = 1 - (1 - csaturation) / 2;
        }
        else {
            
        }
    }

    // Input
    if ((int)(cuCreal(z)) - cuCreal(z) == 0 || (int)(cuCimag(z)) - cuCimag(z) == 0) {
        lightness = lightness*lightness*lightness/sqrtf(2);
        csaturation = 1;
    }

    // Color (from NVDA)
    idx = 3*(idx*HEIGHT + idy);
    float nNormalizedH = angle/2/M_PI;
    float nNormalizedL = lightness;
    float nNormalizedS = 1 - csaturation;
    float nM1, nM2, nR, nG, nB;
    float nh = 0.0f;
    if (nNormalizedL <= 0.5F)
        nM2 = nNormalizedL * (1.0F + nNormalizedS);
    else
        nM2 = nNormalizedL + nNormalizedS - nNormalizedL * nNormalizedS;
    nM1 = 2.0F * nNormalizedL - nM2;
    if (nNormalizedS == 0.0F)
        nR = nG = nB = nNormalizedL;
    else
    {
        nh = nNormalizedH + 0.3333F;
        if (nh > 1.0F)
            nh -= 1.0F;
    }
    float nMDiff = nM2 - nM1;
    if (0.6667F < nh)
        nR = nM1;
    else
    {    
        if (nh < 0.1667F)
            nR = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
        else if (nh < 0.5F)
            nR = nM2;
        else
            nR = nM1 + nMDiff * ( 0.6667F - nh ) * 6.0F; // / 0.1667F
    }
    // Green   
    nh = nNormalizedH;
    if (0.6667F < nh)
        nG = nM1;
    else
    {
        if (nh < 0.1667F)
            nG = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
        else if (nh < 0.5F)
            nG = nM2;
        else
            nG = nM1 + nMDiff * (0.6667F - nh ) * 6.0F; // / 0.1667F
    }
    // Blue    
    nh = nNormalizedH - 0.3333F;
    if (nh < 0.0F)
        nh += 1.0F;
    if (0.6667F < nh)
        nB = nM1;
    else
    {
        if (nh < 0.1667F)
            nB = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
        else if (nh < 0.5F)
            nB = nM2;
        else
            nB = nM1 + nMDiff * (0.6667F - nh ) * 6.0F; // / 0.1667F
    }        
    d_image[idx    ] = 255*(nB);
    d_image[idx + 1] = 255*(nG);
    d_image[idx + 2] = 255*(nR);
}

void generateplot(int initial = 0, int interval = 256, int unitsquare = 256, int increment = 4) {
    interval += initial;
    std::cout << "Generating sequences of images starting at height " << initial << ", resolution " << unitsquare << std::endl;
    for (int ini = initial; ini <= interval; ini += increment) {
        // Allocate host memory for the plot
        cuDoubleComplex *h_plot;
        getStatus(cudaMallocHost(&h_plot, ENTRIES), "Failed to allocate cudaMallocHost! ");
        cuDoubleComplex *h_input;
        getStatus(cudaMallocHost(&h_input, ENTRIES), "Failed to allocate cudaMallocHost! ");

        // Plot
        double x_ini = -3.25;
        double y_ini = -2 + ini;
        cudaZeta(h_plot, x_ini, y_ini, unitsquare, unitsquare, h_input);

        // Generate image
        unsigned char *h_image, *d_image;
        getStatus(cudaMallocHost(&h_image, 3*IMGMEMSIZE), "Failed to allocate cudaMallocHost! ");
        getStatus(cudaHostGetDevicePointer(&d_image, h_image, 0), "Failed to perform host to device for image");
        cudaDeviceSynchronize();
        generate_phase_plot<<<dim3(WIDTH >> 5, HEIGHT >> 5), dim3(32, 32)>>>(d_image, h_plot, h_input, 32);
        cudaDeviceSynchronize();
        cudaFree(d_image);

        // Save image
        cv::Mat3f hls = cv::Mat(WIDTH, HEIGHT, CV_8UC3, h_image);
        std::stringstream ss;
        ss << std::setbase(10) << std::setw(4) << ini;
        std::string hexstr = ss.str();
        std::replace(hexstr.begin(), hexstr.end(), ' ', '0');
        cv::imwrite("test/plot/Plot" + hexstr + ".png", hls);

        // Draw spiral frames
        const int FINE = unitsquare / 8;
        const int YFINE = FINE / 8;
        for (int y = 0; y < HEIGHT; y += FINE/2) {
            // Generate header and spiral plot
            cv::Ptr<cv::freetype::FreeType2> ft2;
            ft2 = cv::freetype::createFreeType2();
            ft2->loadFontData("/usr/share/fonts/opentype/unifont/unifont.otf", 0 );
            double2 zeta = h_plot[WIDTH*HEIGHT/2 + y];
            double2 input = h_input[WIDTH*HEIGHT/2 + y];
            double z_y = cuCreal(input);
            
            // Header
            cv::Mat3b header = cv::Mat3b::zeros((1080 - HEIGHT)/2, WIDTH);
            std::stringstream headertext;
            std::stringstream decimaltext;
            decimaltext << std::setprecision(3) << std::setw(3) << z_y - (int)z_y;
            std::string hexstr = decimaltext.str();
            headertext << "zeta(0.5 + i" << (int)z_y << std::setprecision(3) << "." << hexstr.substr(2) << ") = " << (cuCreal(zeta) < 0 ? '-' : ' ') << (cuCreal(zeta) < 0 ? -cuCreal(zeta) : cuCreal(zeta)) << ' ' << (cuCimag(zeta) < 0 ? '-' : '+') << " i" << (cuCimag(zeta) < 0 ? -cuCimag(zeta) : cuCimag(zeta));
            std::string headerstr = headertext.str();
            ft2->putText(header, headerstr, cv::Point(0,16), 16, cv::Scalar(255, 255, 255), -1, cv::LINE_8, true);
            std::stringstream abstext;
            abstext << "magnitude = " << cuCabs(zeta);
            headerstr = abstext.str();
            ft2->putText(header, headerstr, cv::Point(960,16), 16, cv::Scalar(255, 255, 255), -1, cv::LINE_8, true);
            std::stringstream angletext;
            angletext << "phase = " << atan2(cuCimag(zeta), cuCreal(zeta))*180/M_PI << " deg";
            headerstr = angletext.str();
            ft2->putText(header, headerstr, cv::Point(1440,16), 16, cv::Scalar(255, 255, 255), -1, cv::LINE_8, true);
            std::stringstream ss;
            ss << std::setbase(10) << std::setw(4) << ini << ".x" << std::setbase(16) << std::setw(4) << y;
            hexstr = ss.str();
            std::replace(hexstr.begin(), hexstr.end(), ' ', '0');
            cv::imwrite("test/header/Header" + hexstr + ".gif", header);
            std::cout << " H\t" << hexstr << std::endl;
            
            // Spiral graph
            cv::Mat3b spiralimage = cv::Mat3b::zeros(WIDTH, HEIGHT);
            const int tracegrid = 64;
            // Vertical
            for (int t_x = WIDTH/2 - unitsquare / 2; t_x <= WIDTH/2 + unitsquare / 2; t_x += FINE) {
                for (int t_y = 0; t_y < y - 1; t_y += YFINE) {
                    double2 zeta = h_plot[t_x*HEIGHT + t_y];
                    double2 dzeta = h_plot[t_x*HEIGHT + t_y + YFINE];
                    cv::Point2d tracezeta(HEIGHT/2 + tracegrid*cuCimag(zeta), WIDTH/2 + tracegrid*cuCreal(zeta));
                    cv::Point2d tracedzeta(HEIGHT/2 + tracegrid*cuCimag(dzeta), WIDTH/2 + tracegrid*cuCreal(dzeta));
                    cv::Vec3f righthalf(0, 0, 240*(1 - (((float)t_x) - WIDTH/2)/unitsquare) + 10);
                    cv::Vec3f lefthalf(240*(((float)t_x) - WIDTH/2 + unitsquare)/unitsquare + 10, 0, 0);
                    if (t_x < WIDTH / 2)
                        cv::line(spiralimage, tracezeta, tracedzeta, lefthalf, 1, cv::LINE_AA);
                    else if (t_x > WIDTH / 2)
                        cv::line(spiralimage, tracezeta, tracedzeta, righthalf, 1, cv::LINE_AA);
                        
                }
            }
            // Horizontal
            for (int t_x = WIDTH/2 - unitsquare / 2; t_x < WIDTH/2 + unitsquare / 2; t_x += FINE) {
                for (int t_y = 0; t_y <= y; t_y += FINE) {
                    double2 zeta = h_plot[t_x*HEIGHT + t_y];
                    double2 lzeta = h_plot[(t_x + FINE)*HEIGHT + t_y];
                    cv::Point2d tracezeta(HEIGHT/2 + tracegrid*cuCimag(zeta), WIDTH/2 + tracegrid*cuCreal(zeta));
                    cv::Point2d tracelzeta(HEIGHT/2 + tracegrid*cuCimag(lzeta), WIDTH/2 + tracegrid*cuCreal(lzeta));
                    cv::Vec3f righthalf(0, 0, 240*(1 - (((float)t_x) - WIDTH/2)/unitsquare) + 10);
                    cv::Vec3f lefthalf(240*(((float)t_x) - WIDTH/2 + unitsquare)/unitsquare + 10, 0, 0);
                    if (t_x < WIDTH / 2)
                        cv::line(spiralimage, tracezeta, tracelzeta, lefthalf, 1, cv::LINE_AA);
                    else if (t_x >= WIDTH / 2)
                        cv::line(spiralimage, tracezeta, tracelzeta, righthalf, 1, cv::LINE_AA);
                }
            }
            // Front end
            cv::Vec3f front(0,240,0);
            for (int t_x = WIDTH/2 - unitsquare/2; t_x < WIDTH/2 + unitsquare/2 + unitsquare; t_x += YFINE) {
                int t_y = y;
                double2 zeta = h_plot[t_x*HEIGHT + t_y];
                double2 lzeta = h_plot[(t_x + YFINE)*HEIGHT + t_y];
                cv::Point2d tracezeta(HEIGHT/2 + tracegrid*cuCimag(zeta), WIDTH/2 + tracegrid*cuCreal(zeta));
                cv::Point2d tracelzeta(HEIGHT/2 + tracegrid*cuCimag(lzeta), WIDTH/2 + tracegrid*cuCreal(lzeta));
                
                if (t_x < WIDTH / 2)
                    cv::line(spiralimage, tracezeta, tracelzeta, front, 2, cv::LINE_AA);
                else if (t_x >= WIDTH / 2)
                    cv::line(spiralimage, tracezeta, tracelzeta, front, 2, cv::LINE_AA);
            }
            // Basel line
            cv::Vec3f unity(0,250,0);
            int t_x = WIDTH/2 + unitsquare/2 + unitsquare;
            for (int t_y = 0; t_y < y - 1; t_y += 1) {
                double2 zeta = h_plot[t_x*HEIGHT + t_y];
                double2 dzeta = h_plot[t_x*HEIGHT + t_y + 1];
                cv::Point2d tracezeta(HEIGHT/2 + tracegrid*cuCimag(zeta), WIDTH/2 + tracegrid*cuCreal(zeta));
                cv::Point2d tracedzeta(HEIGHT/2 + tracegrid*cuCimag(dzeta), WIDTH/2 + tracegrid*cuCreal(dzeta));
                cv::line(spiralimage, tracezeta, tracedzeta, unity, 1, cv::LINE_AA);
            }
            // Critical line
            t_x = WIDTH / 2;
            cv::Vec3f criticalline(240,240,240);
            for (int t_y = 0; t_y < y - 1; t_y += 1) {
                double2 zeta = h_plot[t_x*HEIGHT + t_y];
                double2 dzeta = h_plot[t_x*HEIGHT + t_y + 1];
                cv::Point2d tracezeta(HEIGHT/2 + tracegrid*cuCimag(zeta), WIDTH/2 + tracegrid*cuCreal(zeta));
                cv::Point2d tracedzeta(HEIGHT/2 + tracegrid*cuCimag(dzeta), WIDTH/2 + tracegrid*cuCreal(dzeta));
                cv::Vec3f righthalf(480*(1-((float)t_x)/WIDTH) + 1./8, 0, 0);
                cv::Vec3f criticalline(240,240,240);
                cv::Vec3f lefthalf(0,0,240*((float)t_x)/WIDTH - 1./8);
                cv::line(spiralimage, tracezeta, tracedzeta, criticalline, 2, cv::LINE_AA);
            }
            cv::Vec3f grid(127,127,127);
            cv::Point2d realbegin(HEIGHT / 2, 0);
            cv::Point2d realend(HEIGHT/2, WIDTH);
            cv::Point2d imagbegin(0, WIDTH/2);
            cv::Point2d imagend(HEIGHT, WIDTH/2);
            cv::Point2d center(HEIGHT / 2, WIDTH / 2);
            cv::line(spiralimage, realbegin, realend, grid, 1);
            cv::line(spiralimage, imagbegin, imagend, grid, 1);
            cv::circle(spiralimage, center, tracegrid, grid, 1);

            ss = std::stringstream();
            ss << std::setbase(10) << std::setw(4) << ini << ".x" << std::setbase(16) << std::setw(4) << y;
            hexstr = ss.str();
            std::replace(hexstr.begin(), hexstr.end(), ' ', '0');

            cv :: imwrite("test/spiral/Spiral" + hexstr + ".gif", spiralimage);
            std::cout << " S\t" << hexstr << std::endl;
            
        }
        // Free memory
        cudaFreeHost(h_plot);
        cudaFreeHost(h_input);
    }
}

void generatedepthplot(int initial = 0, int interval = 256, int unitsquare = 256, int increment = 4) {
    interval += initial;
    std::cout << "Generating sequences of images starting at height " << initial << ", resolution " << unitsquare << std::endl;
    for (int ini = initial; ini <= interval; ini += increment) {
        // Allocate host memory for the plot
        cuDoubleComplex *h_plot;
        getStatus(cudaMallocHost(&h_plot, MEMSIZE), "(h_plot) Failed to allocate cudaMallocHost! ");
        cuDoubleComplex *h_input;
        getStatus(cudaMallocHost(&h_input, ENTRIES), "(h_input) Failed to allocate cudaMallocHost! ");

        // Plot
        double x_ini = -3.25;
        double y_ini = -2 + ini;
        // cudaZetaDepth(h_plot, x_ini, y_ini, unitsquare, unitsquare, h_input);

        // Free memory
        cudaFreeHost(h_plot);
        cudaFreeHost(h_input);

        sleep(5);
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
        //  printf("\tCluster support  : %d\n", deviceProp.clusterLaunch);
    }

    // Generate plot
    generatedepthplot(0, 1024, 256, 16);

    return EXIT_SUCCESS;
}
