#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include <nccl.h>
// #define NUM_DEVICES 4

void print_matrix_d(double* dst, int num_rows, int n)
{
    double *a = (double*)calloc(sizeof(double), num_rows * n);
    cudaMemcpy(a, dst, num_rows * n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = 0; j < n; ++j)
            printf("%lf ", a[i * n + j]);
        printf("\n");
    }
    printf("\n");

    free(a);
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

	if(i > 0 && i < n - 1 && j > y_start && j < y_end)
		U_d_diff[j*n + i] = U_n_d[j*n + i] - U_d[j*n + i];
}

int main(int argc, char * argv[])
{

    int local_rank, proc_amount;

    MPI_Init(&argc, &argv);

    double min_err = 0.000001;
    int N = 128;
    int maxi = 50000;

    MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_amount);

    ncclUniqueId nccl_id;
    
    double *tmp = NULL;
    double *U_d = NULL;
    double *U_n_d = NULL;
    double *tmp_d = NULL;
    double *max_d = NULL;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

   //  cudaSetDevice(local_rank % proc_amount);

    int isLastProcFlag = (local_rank / (proc_amount - 1));
    int isFirstProcFlag = (proc_amount - local_rank) / proc_amount;

    if(isFirstProcFlag)
	ncclGetUniqueId(&nccl_id);
	 
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaSetDevice(local_rank % proc_amount);

    int start = ((N / proc_amount) * local_rank) * N;
    int end = (N / proc_amount * (local_rank + 1) + (N % proc_amount) * isLastProcFlag) * N; // [start; end)

    int num_elems = end - start;
    int num_rows = num_elems / N;

    // interpolation for different processes
    tmp = (double*)calloc(num_elems, sizeof(double));

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
        tmp[(num_rows - 1) * N] = 20.0;
        tmp[num_rows * N - 1] = 30.0;
	// [num_rows - 1][127] == [(num_rows - 1)*N + (N - 1)] == [num_rows * N - 1]
        for (int i = (num_rows - 1) * N + 1; i < num_rows * N - 1; i++)
            tmp[i] = tmp[i - 1] + step;
    }

    // interpolate left and right boarders
    for (int i = 0 + isFirstProcFlag; i < num_rows - isLastProcFlag; i++) {
        tmp[i * N + 0] = 10.0 + step * ((start/ N) + i);
        tmp[i * N + (N - 1)] = 20.0 + step * ((start/ N) + i);
    }

    // copying to GPU
    cudaMalloc((void **)&U_d, (num_elems + 2 * N) * sizeof(double));
    cudaMalloc((void **)&U_n_d, (num_elems + 2 * N) * sizeof(double));
    double* zeros = (double*)calloc(num_elems + 2*N, sizeof(double));
    // fill arrays by zeros to clear first and last buff rows
    cudaMemcpyAsync(U_d, zeros,(num_elems + 2*N) * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(U_n_d, zeros, (num_elems + 2*N)* sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(U_d + N, tmp, num_elems * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(U_n_d + N, tmp, num_elems * sizeof(double), cudaMemcpyHostToDevice, stream);

    free(tmp);

    dim3 GS = dim3(16, 16);
    dim3 BS = dim3(ceil(N / (double)GS.x), ceil((num_rows + 2) / (double)GS.y));

    cudaMalloc(&tmp_d, sizeof(double) * num_elems);
    cudaMalloc(&max_d, sizeof(double));

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, num_rows * N, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int topProcess = (local_rank + 1) % proc_amount;
    int bottomProcess = (local_rank + proc_amount - 1) % proc_amount;


    // calc of values that boundaries of  process
    int y_start = isFirstProcFlag;
    int y_end = num_rows + 1 - isLastProcFlag;

    int it = 0;
    double error = 228;
    //double local_error = 0;
    double * err_d;
    cudaMalloc(&err_d, sizeof(double));

    ncclComm_t com;
    ncclCommInitRank(&com, proc_amount, nccl_id, local_rank);

    while (error > min_err && it < maxi)
    {
        it += 1;

        ncclGroupStart();
        ncclSend(U_d + N, N, ncclDouble, bottomProcess, com, stream);
        ncclSend(U_d + num_elems, N, ncclDouble, topProcess, com, stream);

        ncclRecv(U_d + num_elems + N, N, ncclDouble, topProcess, com, stream);
        ncclRecv(U_d, N, ncclDouble, bottomProcess, com, stream);
        ncclGroupEnd();

        five_point_model_calc<<<BS, GS,0, stream>>>(U_n_d, U_d, N, y_start, y_end);
        if (it % 100 == 0)
        {
            arr_diff<<<BS, GS, 0, stream>>>(U_n_d, U_d, tmp_d, N, y_start, y_end);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, num_rows * N, stream);

            ncclAllReduce(max_d, err_d, 1, ncclDouble, ncclMax, com, stream);
            cudaMemcpyAsync(&error, err_d, sizeof(double), cudaMemcpyDeviceToHost, stream);
            
            if (isFirstProcFlag)
               printf("iter: %d error: %e\n", it, error);
        }

        double *tmp = U_d;
        U_d = U_n_d;
        U_n_d = tmp;
    }
    //if(isLastProcFlag)
    //printf("rank: %d\n", local_rank);
    //print_matrix_d(U_d, num_rows, N);
    cudaFree(U_d);
    cudaFree(U_n_d);
    cudaFree(tmp_d);
    cudaFree(max_d);

    cudaStreamDestroy(stream);
    ncclCommDestroy(com);
    MPI_Finalize();

    return 0;
}
