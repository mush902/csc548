// Single Author info:
// mshaikh2 Mushtaq Ahmed Shaikh

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define XI  -M_PI/4.0
#define XF  M_PI/4.0

double fn(double x)
{
    return atan(x);
}

__global__ void compute_area(double* yc, double* inf, int *NGRID, double *h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("In compute area value of i is %d ", i);
    double area = 0.0;
    if (i >= 2 && i <= *NGRID)
    {
        area = (yc[i] + yc[i - 1]) / 2 * *h;
	//printf("In compute area value of i  %d yc[i] %g yc[i-1] %g inf[i] %g \n", i, yc[i], yc[i-1], inf[i]);
    	inf[i]=area;
	//printf("inf[i] %g \n", inf[i]);
    }
}

void print_function_data(int np, double* x, double* y, double* dydx);

int main(int argc, char* argv[])
{
    int NGRID;

    if (argc > 1)
        NGRID = atoi(argv[1]);
    else
    {
        printf("Please specify the number of grid points.\n");
        exit(0);
    }

    int i;
    double h;

    double* inf = (double*)malloc(sizeof(double) * (NGRID + 1));
    double* xc = (double*)malloc(sizeof(double) * (NGRID + 1));
    double* yc = (double*)malloc(sizeof(double) * (NGRID + 1));

    for (i = 1; i <= NGRID; i++)
    {
        xc[i] = XI + (XF - XI) * (double)(i - 1) / (double)(NGRID - 1);
    }

    int imin, imax;

    imin = 1;
    imax = NGRID;

    for (i = imin; i <= imax; i++)
    {
        yc[i] = fn(xc[i]);
    }

    inf[1] = 0.0;
    h = (XF - XI) / (NGRID - 1);

    //memory for gpu 
    double* gpu_yc;
    double* gpu_inf;
    double* gpu_h;
    int* gpu_NGRID;
    //gpu memory allocation
    cudaMalloc((void**)&gpu_yc, sizeof(double) * (NGRID + 1));
    cudaMalloc((void**)&gpu_inf, sizeof(double) * (NGRID + 1));
    cudaMalloc( (void **) &gpu_NGRID, sizeof(int) * 1 );
    cudaMalloc( (void **) &gpu_h, sizeof(double) * 1 );
    
    // copy data from host xc and inf to gpu variables gpu_yc and gpu_inf
    cudaMemcpy(gpu_yc, yc, sizeof(double) * (NGRID + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_inf, inf, sizeof(double) * (NGRID + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_h, &h, sizeof(double) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_NGRID, &NGRID, sizeof(int) * 1, cudaMemcpyHostToDevice);
    
    // Defining block size using 32 threads per block
    int block_size = NGRID/32;
    if(NGRID%32!=0) {
        block_size+=1;
    }
    //printf("Block size %d \n", block_size);

    compute_area<<<block_size, 32>>>(gpu_yc, gpu_inf, gpu_NGRID, gpu_h);

    // Copy the result back
    cudaMemcpy(inf, gpu_inf, sizeof(double) * (NGRID + 1), cudaMemcpyDeviceToHost);
    for(int i=0;i<=NGRID;i++) {
        inf[i]+=inf[i-1];
    }
    // Free GPU memory
    cudaFree(gpu_yc);
    cudaFree(gpu_inf);
    cudaFree(gpu_h);
    cudaFree(gpu_NGRID);

    print_function_data(NGRID, &xc[1], &yc[1], &inf[1]);
    //Free the host variables on heap
    free(xc);
    free(yc);
    free(inf);

    return 0;
}

void print_function_data(int np, double* x, double* y, double* dydx)
{
    int i;

    char filename[1024];
    sprintf(filename, "fn-%d.dat", np);

    FILE* fp = fopen(filename, "w");

    for (i = 0; i < np; i++)
    {
        fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
    }

    fclose(fp);
}

