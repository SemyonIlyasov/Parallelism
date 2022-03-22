#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <cublas_v2.h>

#define N 128

/*
    main difference between nonoptimized and optimized programs is
    that the second program calculates max_error value only 1 time per 100 repeats
*/
int main(void)
{
    double* copy_pointer;
    int iteration = 0;
    double* max_error = (double*)calloc(1, sizeof(double));
    max_error[0] = 10;
    double delta = 10.0 / (N - 1);
    cublasHandle_t handle;
    cublasCreate(&handle);

    double* U = (double*)calloc(N*N, sizeof(double));
    double* U_n = (double*)calloc(N*N, sizeof(double));

#pragma acc enter data create(U[0:N*N], U_n[0:N*N]) copyin(N, delta)
    // enter data - enter the unstructured data section.

    // create - allocation of memory on the GPU

    // copyin(list) - allocate memory for all variables from the list
    // and copy their values to the GPU at
    // the beginning of the section;
    // release memory on the GPU after exiting the section

#pragma acc kernels
    // the code may contain
    // parallelism, and the compiler
    // determines which of this code can be
    // safely parallelized.

    {
#pragma acc loop independent
        for (int i = 0; i < N; i++) {
            U[i*N] = 10 + delta * i;
            U[i] = 10 + delta * i;
            U[(N - 1)*N + i] = 20 + delta * i;
            U[i*N + N - 1] = 20 + delta * i;

            U_n[i*N] = U[i*N];
            U_n[i] = U[i];
            U_n[(N - 1)*N + i] = U[(N - 1)*N + i];
            U_n[i*N + N - 1] = U[i*N + N - 1];
        }
    }


    {
        while (max_error[0] > 1e-6 && iteration < 1e+6) {

            iteration++;

            if (iteration % 100 == 0) {
#pragma acc data present(U[0:N*N], U_n[0:N*N])
    // present(list) - assume that all variables 
    // from the list are already present on the GPU;

#pragma acc kernels async(1)
    // async[(n)] - indicates that the kernel is run asynchronously in
    // queue n, and at the end of the section, do not force synchronization;

                {
#pragma acc loop independent collapse(2)
    // independent - to assure the compiler that
    // there are no dependencies in this loop
    // and all iterations can be executed in parallel;

    // collapse(n) - turn n nested loops into one; it may
    // be advantageous if the loops themselves are small; n - amount of loops
                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++)
                            U_n[i*N + j] = 0.25 * (U[(i+1)*N + j] + U[(i - 1)*N +j] + U[i*N + j - 1] + U[i*N + j + 1]);
		}
		int max_err_idx = 0;
#pragma acc wait
#pragma acc host_data use_device(U, U_n)
{
		const double alpha = -1;
		cublasDaxpy(handle, N * N, &alpha, U_n, 1, U, 1);
		cublasIdamax(handle, N*N, U, 1, &max_err_idx);
}

#pragma acc update self(U[max_err_idx - 1:1])
		max_error[0] = fabs(U[max_err_idx -1]);
#pragma acc host_data use_device(U, U_n)
		cublasDcopy(handle, N*N, U_n, 1, U, 1);
            }
            else {

#pragma acc data present(U[0:N*N], U_n[0:N*N])
#pragma acc kernels async(1)
                {
#pragma acc loop independent collapse(2)
                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++)
                            U_n[i*N + j] = 0.25 * (U[(i + 1)*N + j] + U[(i - 1)* N + j] + U[i*N + j - 1] + U[i*N + j + 1]);
                }
            }

            // swap U_n and U arrays

            copy_pointer = U;
            U = U_n;
            U_n = copy_pointer;

            if (iteration % 100 == 0) {
#pragma acc wait(1)
    // synchronization point
                printf("%d %lf\n", iteration, max_error[0]);
            }
        }
    }

    printf("%d %lf\n", iteration, max_error[0]);


    return 0;
}
