#include <stdio.h>

#define N 128
__global__ void five_point_model_calc(double* U_d, double* U_d_n, const int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < N - 1 && i > 0)
	{
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (j < N - 1 && j > 0)
		{
			float left = U_d[i*N + j - 1];
			float right = U_d[i*N + j + 1];
			float up = U_d[(i-1)*N + j];
			float down = U_d[(i+1)*N + j];

			U_d_n[i*N + j] = 0.25 * (left + right + up + down);
		}
	}
}

int main(void){

double* U = (double*)malloc(N*N*sizeof(double));
double* U_n = (double*)malloc(N*N*sizeof(double));
double* U_d = (double*)malloc(N*N*sizeof(double));
double* U_d_n = (double*)malloc(N*N*sizeof(double));

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

cudaMemcpy(U_d_n, U_n, N*N*sizeof(double), cudaMemcpyHostToDevice);

dim3 BLOCK_SIZE = dim3(ceil(N/128), 1, 1);
dim3 GRID_SIZE = dim3(128, 1, 1);
five_point_model_calc<<<BLOCK_SIZE, GRID_SIZE>>>(U_d, U_d_n, N);

free(U);
free(U_n);
cudaFree(U_d);
cudaFree(U_d_n);

return 0;
}
