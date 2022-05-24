#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
// #define NUM_DEVICES 4

void print_matrix_d(double* dst, int numRows, int n)
{
    double *a = (double*)calloc(sizeof(double), numRows * n);
    cudaMemcpy(a, dst, numRows * n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < n; ++j)
            printf("%lf ", a[i * n + j]);
        printf("\n");
    }
    printf("\n");

    free(a);
}


__global__ void evalEquation(double *newA, const double *A, int width, int y_start, int y_end)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < idx && idx < width - 1) && (y_start < idy && idy < y_end))
    {
        newA[idy * width + idx] = 0.25 * (A[(idy - 1) * width + idx] + A[(idy + 1) * width + idx] +
                                          A[idy * width + (idx - 1)] + A[idy * width + (idx + 1)]);
    }
}

__global__ void five_point_model_calc(double* U_n_d, double *U_d, int n, int y_start, int y_end)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(j < y_end && j > y_start && i > 0 && i < n - 1)
	{
		double left = U_d[j*n + i - 1];
		double right = U_d[j*n + i + 1];
		double up = U_d[(j-1)*n + i];
		double down = U_d[(j+1)*n + i];

		U_n_d[j*n + i] = 0.25 * (left + right + up + down);

	}
}

__global__ void arr_diff(double* U_n_d, double* U_d, double* U_d_diff, int n, int y_start, int y_end)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i >= 0 && i < n && j > y_start && j < y_end)
		U_d_diff[i*n + j] = U_n_d[i*n + j] - U_d[i*n + j];
}

__global__ void vecNeg(const double *newA, const double *A, double* ans, int mx_size, int numElems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= mx_size && idx < numElems + mx_size)
    {
        ans[idx] = newA[idx] - A[idx];
    }
}

int main(int argc, char * argv[])
{

    MPI_Status status;
    int local_rank, proc_amount;

    MPI_Init(&argc, &argv);

    double min_error = 0.000001;
    int N = 128;
    int iter_max = 50000;

    MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_amount);

    double *tmp = NULL;
    double *U_d = NULL;
    double *U_n_d = NULL;
    double *tmp_d = NULL;
    double *max_d = NULL;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaSetDevice(local_rank % proc_amount);
    // setDevice(local_rank);

    int isLastProcFlag = (local_rank / (proc_amount - 1));
    int isFirstProcFlag = (proc_amount - local_rank) / proc_amount;
    // OK

    int start = ((N / proc_amount) * local_rank) * N;
    int end = (N / proc_amount * (local_rank + 1) + (N % proc_amount) * isLastProcFlag) * N; // [start; end)
    // OK

    int numElems = end - start;
    int numRows = numElems / N;
    // OK

    // interpolation for different processes
    tmp = (double*)calloc(numElems, sizeof(double));

    double step = (double)(20.0 - 10.0) / ((double)N - 1.0);
    if (isFirstProcFlag)
    {
        // interpolate upper boarder
        tmp[0] = 10.0;
        tmp[N - 1] = 20.0;

        for (int i = 1; i < N - 1; i++)
            tmp[i] = tmp[i - 1] + step;

    }
    if (isLastProcFlag)
    {
        // interpolate lower boarder
        tmp[(numRows - 1) * N] = 20.0;
        tmp[numRows * N - 1] = 30.0;
	// [numRows - 1][127] == [(numRows - 1)*N + (N - 1)] == [numRows * N - 1]
        for (int i = (numRows - 1) * N + 1; i < numRows * N - 1; i++)
            tmp[i] = tmp[i - 1] + step;
    }

    // interpolate left and right boarders
    for (int i = 0 + isFirstProcFlag; i < numRows - isLastProcFlag; i++) {
        tmp[i * N + 0] = 10.0 + step * ((start/ N) + i);
        tmp[i * N + (N - 1)] = 20.0 + step * ((start/ N) + i);
    }
    // OK
    // copying to GPU
    cudaMalloc((void **)&U_d, (numElems + 2 * N) * sizeof(double));
    cudaMalloc((void **)&U_n_d, (numElems + 2 * N) * sizeof(double));
    double* zeros = (double*)calloc(numElems + 2*N, sizeof(double));
    // fill arrays by zeros to clear first and last buff rows
    cudaMemcpy(U_d, zeros,(numElems + 2*N) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_n_d, zeros, (numElems + 2*N)* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_d + N, tmp, numElems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_n_d + N, tmp, numElems * sizeof(double), cudaMemcpyHostToDevice);

    free(tmp);

    dim3 GS = dim3(16, 16);
    dim3 BS = dim3(ceil(N / (double)GS.x), ceil((numRows + 2) / (double)GS.y));

    cudaMalloc(&tmp_d, sizeof(double) * numElems);
    cudaMalloc(&max_d, sizeof(double));

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, numRows * N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int topProcess = (local_rank + 1) % proc_amount;
    int bottomProcess = (local_rank + proc_amount - 1) % proc_amount;
    // OK , mb swap



    // calculation of values that limit the upper and lower boundaries for processes, depending on the position
    int y_start = isFirstProcFlag;
    int y_end = numRows + 1 - isLastProcFlag;

    int GS_neg = N;
    int BS_neg = ceil(numElems / (double)GS_neg);

    int iter = 0;
    double error = 228;
    double local_error = 0;

    while (error > min_error && iter < iter_max)
    {
        iter += 1;

        MPI_Sendrecv(U_d + N, N, MPI_DOUBLE, bottomProcess, local_rank,
                     U_d + numElems + N, N, MPI_DOUBLE, topProcess, topProcess,
                     MPI_COMM_WORLD, &status);

        MPI_Sendrecv(U_d + numElems, N, MPI_DOUBLE, topProcess, local_rank,
                     U_d, N, MPI_DOUBLE, bottomProcess, bottomProcess,
                     MPI_COMM_WORLD, &status);

        //evalEquation<<<BS, GS>>>(U_n_d, U_d, N, y_start, y_end);
	five_point_model_calc<<<BS, GS>>>(U_n_d, U_d, N, y_start, y_end);
        if (iter % 100 == 0)
        {
            vecNeg<<<BS_neg, GS_neg>>>(U_n_d, U_d, tmp_d, N, numElems);
	    //arr_diff<<<BS, GS>>>(U_n_d, U_d, tmp_d, N, y_start, y_end);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, numRows * N);

            cudaMemcpy(&local_error, max_d, sizeof(double), cudaMemcpyDeviceToHost);
            MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            if (isFirstProcFlag)
                printf("iter: %d error: %e\n", iter, error);
        }

        double *tmp = U_d;
        U_d = U_n_d;
        U_n_d = tmp;
    }
    //if(isLastProcFlag)
    //printf("rank: %d\n", local_rank);
    //print_matrix_d(U_d, numRows, N);
    cudaFree(U_d);
    cudaFree(U_n_d);
    cudaFree(tmp_d);
    cudaFree(max_d);

    MPI_Finalize();

    return 0;
}
