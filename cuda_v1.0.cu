#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "cub/cub.cuh"
#define N 10
__global__ void five_point_model_calc(double* U_d, double* U_d_n, int n)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n - 1 && i > 0)
	{
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (j < n - 1 && j > 0)
		{
			float left = U_d[i*n + j - 1];
			float right = U_d[i*n + j + 1];
			float up = U_d[(i-1)*n + j];
			float down = U_d[(i+1)*n + j];

			U_d_n[i*n + j] = 0.25 * (left + right + up + down);
		}
	}
}

int main(void)
{

double* U = (double*)malloc(N*N*sizeof(double));
double* U_n =(double*)malloc(N*N*sizeof(double));

double* U_d;
double* U_d_n;

cudaMalloc(&U_d, sizeof(double)*N*N);
cudaMalloc(&U_d_n, sizeof(double)*N*N);

double delta = 10.0 / (N - 1);

for (int i = 0; i < N; i++)
{
	U[i*N] = 10 + delta * i;
	U[i] = 10 + delta * i;
	U[(N-1)*N + i] = 20 + delta * i;
	U[i*N + N - 1] = 20 + delta * i;

	U_n[i*N] = U[i*N];
	U_n[i] = U[i];
	U_n[(N-1)*N + i] = U[(N-1)*N + i];
	U_n[i*N + N - 1] = U[i*N + N - 1];
}
cudaMemcpy(U_d, U, N*N*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(U_d_n, U_n, N*N*sizeof(double), cudaMemcpyHostToDevice);

dim3 GRID_SIZE = dim3(ceil((N + 127)/128.), 1, 1);
dim3 BLOCK_SIZE = dim3(128, 1, 1);

five_point_model_calc<<<GRID_SIZE, BLOCK_SIZE>>>(U_d, U_d_n, N);

cudaMemcpy(U_n, U_d_n, N*N*sizeof(double), cudaMemcpyDeviceToHost);

for(int i = 0; i < N; i++){
	for(int j = 0; j < N; j++)
		printf("%13.10f ", U_n[i][j]);
	printf("\n");
}

free(U);
free(U_n);
cudaFree(U_d);
cudaFree(U_d_n);

return 0;
}
