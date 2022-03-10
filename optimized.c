
#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#include <math.h>

#define N 128

/*
    main difference between nonoptimized and optimized programs is
    that the second program calculates max_error value only 1 time per 100 repeats
*/
int main(void) 
{
    double** copy_pointer;
    int iteration = 0;
    double max_error = 1.0;
    double delta = 10.0 / (N - 1);

    double** U = (double**)calloc(N, sizeof(double*));
    double** U_n = (double**)calloc(N, sizeof(double*));

    for (int i = 0; i < N; i++) {
        U[i] = (double*)calloc(N, sizeof(double));
        U_n[i] = (double*)calloc(N, sizeof(double));
    }

#pragma acc enter data create(U[0:N][0:N], U_n[0:N][0:N]) copyin(N, delta)
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
        for (int i = 0; i < N; i++) {
            U[i][0] = 10 + delta * i;
            U[0][i] = 10 + delta * i;
            U[N - 1][i] = 20 + delta * i;
            U[i][N - 1] = 20 + delta * i;

            U_n[i][0] = U[i][0];
            U_n[0][i] = U[0][i];
            U_n[N - 1][i] = U[N - 1][i];
            U_n[i][N - 1] = U[i][N - 1];
        }
    }


#pragma acc data create(max_error)
    {
        while (max_error > 1e-6 && iteration < 1e+6) {

            iteration++;

            if (iteration % 100 == 0) {
#pragma acc kernels async(1)
                max_error = 0.0;

#pragma acc data present(U, U_n)
    // present(list) - assume that all variables 
    // from the list are already present on the GPU;

#pragma acc kernels async(1)
    // async[(n)] - indicates that the kernel is run asynchronously in
    // queue n, and at the end of the section, do not force synchronization;

                {
#pragma acc loop independent collapse(2) reduction(max:max_error)
    // independent - to assure the compiler that
    // there are no dependencies in this loop
    // and all iterations can be executed in parallel;

    // collapse(n) - turn n nested loops into one; it may
    // be advantageous if the loops themselves are small; n - amount of loops

    // reduction(operation:list of variables) - perform
    // reduction on variables from the list

                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++) {
                            U_n[i][j] = 0.25 * (U[i + 1][j] + U[i - 1][j] + U[i][j - 1] + U[i][j + 1]);
                            max_error = fmax(max_error, U_n[i][j] - U[i][j]);
                        }
                }

            }
            else {

#pragma acc data present(U, U_n)
#pragma acc kernels async(1)
                {
#pragma acc loop independent collapse(2)
                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++)
                            U_n[i][j] = 0.25 * (U[i + 1][j] + U[i - 1][j] + U[i][j - 1] + U[i][j + 1]);
                }
            }

            // swap U_n and U arrays

            copy_pointer = U;
            U = U_n;
            U_n = copy_pointer;

            if (iteration % 100 == 0) {
#pragma acc wait(1)
    // synchronization point

#pragma acc update host(max_error)
    // update host(list) - for all variables from the list,
    // update their values in the CPU memory
    // with values on the GPU

                printf("%d %lf\n", iteration, max_error);
            }
        }
    }
    printf("%d %lf\n", iteration, max_error);


    return 0;
}
