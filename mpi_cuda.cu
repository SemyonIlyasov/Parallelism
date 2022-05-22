#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cub/cub.cuh>
#include <mpi.h>
#define N 256
#define MAX_ITER_NUM 15000
#define STEP 100
__global__ void five_point_model_calc(double* U_d, double* U_d_n, int n)
{
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < n - 1 && j > 0 && i > 0 && i < n - 1)
		{
			double left = U_d[i*n + j - 1];
			double right = U_d[i*n + j + 1];
			double up = U_d[(i-1)*n + j];
			double down = U_d[(i+1)*n + j];

			U_d_n[i*n + j] = 0.25 * (left + right + up + down);
		}
}

__global__ void arr_diff(double* U_d, double* U_d_n, double* U_d_diff, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= 0 && i < n && j >= 0 && j < n)
		U_d_diff[i*n + j] = U_d_n[i*n + j] - U_d[i*n + j];
}




int main(int argc, char* argv[])
{


MPI_Init(&argc, &argv);
int local_rank, size;
int num_device;

MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

MPI_Finalize();
return 0;
}
