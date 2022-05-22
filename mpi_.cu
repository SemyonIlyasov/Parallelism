#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#define NUM_DEVICES 4

void printMatrix(double* a, int height, int width)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            printf("%lf ", a[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printCudaMatrix(double* dst, int height, int width)
{
    double *a = (double*)calloc(sizeof(double), height * width);

    cudaMemcpy(a, dst, height * width * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            printf("%lf ", a[i * width + j]);
        }
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

__global__ void vecNeg(const double *newA, const double *A, double* ans, int mx_size, int numElems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= mx_size && idx < numElems + mx_size)
    {
        ans[idx] = newA[idx] - A[idx];
    }
}

int main(int argc, char ** argv)
{

    MPI_Status status;
    int local_rank, proc_amount;

    MPI_Init(&argc, &argv);

    double min_error = 0.000001;
    int N = 128;
    int iter_max = 50000;

    MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_amount);

    cudaError_t err;

    double *tmp, *U_d, *U_n_d, *tmp_d, *max_d;

    void *d_temp_storage;
    size_t temp_storage_bytes = 0;

    //cudaSetDevice(local_rank % NUM_DEVICES);
    setDevice(local_rank);

    int isLastProcFlag = (local_rank / (proc_amount - 1));
    int isFirstProcFrag = (proc_amount - local_rank) / proc_amount;

    int start = (N / proc_amount * local_rank) * N;
    int end = (N / proc_amount * (local_rank + 1) + (N % proc_amount) * isLastProcFlag) * N; // [start; end)

    int numElems = end - start;
    int numRows = numElems / N;

    // interpolation for different processes
    tmp = (double*)calloc(numElems, sizeof(double));

    double step = (10.0 - 20.0) / ((double)N - 1);
    if (isFirstProcFrag)
    {
        // interpolate upper boarder
        tmp[0] = 10.0;
        tmp[N - 1] = 20.0;

        for (int i = 1; i < N - 1; ++i)
            tmp[i] = tmp[i - 1] + step;

    }
    if (isLastProcFlag)
    {
        // interpolate lower boarder
        tmp[(N - 1) * N] = 20.0;
        tmp[N*N - 1] = 30.0;

        for (int i = (N-1)*N + 1; i < N*N - 1; ++i)
            tmp[i] = tmp[i - 1] + step;
    }
    // interpolate left and right boarders
    for (int i = 0; i < numRows; ++i) {
        tmp[i * N + 0] = 10.0 + step * ((start/ N) + i);
        tmp[i * N + (N - 1)] = 20.0 + step * ((start/ N) + i);
    }
}

    // copying to GPU
    cudaMalloc((void **)&U_d, (numElems + 2 * N) * sizeof(double));
    cudaMalloc((void **)&U_n_d, (numElems + 2 * N) * sizeof(double));
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

        evalEquation<<<BS, GS>>>(U_n_d, U_d, N, y_start, y_end);

        if (iter % 100 == 0)
        {
            vecNeg<<<BS_neg, GS_neg>>>(U_n_d, U_d, tmp_d, N, numElems);

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

    cudaFree(U_d);
    cudaFree(U_n_d);
    cudaFree(tmp_d);
    cudaFree(max_d);

    MPI_Finalize();

    return 0;
}