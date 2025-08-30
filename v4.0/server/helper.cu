#ifndef __COMPLEX
#include <cuComplex.h>
#endif

#ifndef __STDC_FORMAT_MACROS
#include <iostream>
#endif

int getStatus(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << msg << cudaGetErrorString(status) << std::endl;
        exit(status);
    }
    return 0;
}

void printNES(cuDoubleComplex *plot, int WIDTH, int HEIGHT) {
    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col += 16)
        {
            int idx = row * WIDTH + col;
            printf("(%5.2f %+5.2f)", ((double2)plot[idx]).x, ((double2)plot[idx]).y);
        }
        printf("\n");
    }
}