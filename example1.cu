#include <stdio.h>

__global__ void calculate(int *a, int *b, int *c){
    c[threadIdx.x] = ((a[threadIdx.x]+2)+b[threadIdx.x])*3;   
}

#define N 512
int main(void){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    //Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    //Setup input values
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    //Copy input from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    //Launch one add() kernel block on GPU with N threads
    calculate<<<1,N>>>(d_a, d_b, d_c);

    //Copy result back to host
    cudaMemcpy(d_c, c, size, cudaMemcpyDeviceToHost);

    //Display the result
    printf("The result is %d",*c);

    //Cleanup
    free(a);free(b);free(c);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);

    //Exit program
    return 0;
}