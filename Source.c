#include <stdio.h>
#include<math.h>
#include <malloc.h>
#define  N 128

int main() {
    double** u = (double**)calloc(N, sizeof(double*));
    double** u_n = (double**)calloc(N, sizeof(double*)); 
    for (int i = 0; i < N; i++) {
        u[i] = (double*)calloc(N, sizeof(double));
        u_n[i] = (double*)calloc(N, sizeof(double));
    }


    u[0][0] = 10;
    u[N - 1][0] = 20;
    u[0][N - 1] = 20;
    u[N - 1][N - 1] = 30;

    u_n[0][0] = 10;
    u_n[N - 1][0] = 20;
    u_n[0][N - 1] = 20;
    u_n[N - 1][N - 1] = 30;

    double step = 10.0 / (N - 1);
    int iter = 0;
    double err = 1;

#pragma acc kernels
    for (int i = 1; i < N - 1; i++) {
        u[i][0] = 10 + step * i;
        u[0][i] = 10 + step * i;
        u[i][N - 1] = 20 + step * i;
        u[N - 1][i] = 20 + step * i;
        u_n[i][0] = 10 + step * i;
        u_n[0][i] = 10 + step * i;
        u_n[i][N - 1] = 20 + step * i;
        u_n[N - 1][i] = 20 + step * i;
    }

#pragma acc data copy(u[0:N][0:N], err) create (u_n[0:N][0:N]) 
    {
        while (err > 1e-6 && iter < 1e+6) {

            iter++;
            if (iter % 100 == 0) {
#pragma acc kernels async
                err = 0;
            }
#pragma acc data present(u, du)
#pragma acc parallel num_gangs(128) async
            {
                if (iter % 100 == 0) {

#pragma acc loop collapse(2) independent reduction(max:err) 
                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++) {

                            u_n[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j - 1] + u[i][j + 1]);
                            err = fmax(err, u_n[i][j] - u[i][j]);
                        }
                }
                else {
#pragma acc loop collapse(2) 

                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++)
                            u_n[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j - 1] + u[i][j + 1]);

                }
            }
#pragma acc parallel loop independent collapse(2) async
            for (int i = 1; i < N - 1; i++)
                for (int j = 1; j < N - 1; j++)
                    u[i][j] = u_n[i][j];

            if (iter % 100 == 0) {
#pragma acc wait 
#pragma acc update self(err) 
                printf("%d %e\n", iter, err);
            }


        }
    }


    return 0;
}




