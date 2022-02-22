#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define n 128

void one_dim_prog() {
    double u[n], up[n];
    u[0] = 0.0;  
    for (int i = 1; i < n - 1; i++) {
        u[i] = 0.0;
    }
    u[n - 1] = 1.0;
    up[0] = 0.0;
    up[n - 1] = 1.0;
    double x_max = 1.0;
    double h = x_max / (double)n;
    double a = 1.0;
    double tau = a / (n * n * n);


    int it_num = 0;
    int flag = 1;
    while (flag == 1) {
        flag = 0;
        for (int i = 1; i < n - 1; i++) {
            up[i] = u[i] + tau * a * (u[i - 1] - 2 * u[i] + u[i + 1]) / (h * h);
            if (fabs(u[i] - up[i]) > 1e-6)
                flag = 1;
        }
        for (int i = 1; i < n - 1; i++) {
            u[i] = up[i];
        }
        it_num++;
    }
    for (int i = 0; i < n; i++)
        printf("%e\n", u[i]);
    printf("number of iterations: %d", it_num);

}

void two_dim_prog() {

    double u[n][n] = {0};
    double u_n[n][n] = {0};
    u[0][0] = 10;
    u[0][n - 1] = 20;
    u[n - 1][0] = 20;
    u[n - 1][n - 1] = 30;

    double up = (double)(u[n - 1][0] - u[0][0]) / (n-1);
    double left = (double)(u[0][n - 1] - u[0][0]) / (n-1);
    double right = (double)(u[n - 1][n - 1] - u[n - 1][0]) /(n-1);
    double down = (double)(u[n - 1][n - 1] - u[0][n - 1]) / (n-1);

    for (int i = 1; i < n - 1; i++) {
        u[0][i] = u[0][i - 1] + up;
        u[n - 1][i] = u[n - 1][i - 1] + down;
        u[i][0] = u[i - 1][0] + left;
        u[i][n - 1] = u[i - 1][n - 1] + right;
    }
    int flag = 1;
    double x_max = 1.0;
    double h = x_max / (double)n;
    double a = 1.0;
    double tau = 30 * a / (n * n * n);
    int it = 0;
    double localmax = 0;

    while (flag == 1) {
        flag = 0;
        localmax = 0;
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                u_n[i][j] = u[i][j] + tau * a * (u[i][j - 1] - 4 * u[i][j] + u[i][j + 1] + u[i + 1][j] + u[i - 1][j]) / (h * h);
                double delta = fabs(u[i][j] - u_n[i][j]);
                if (delta > 1e-6) {
                    flag = 1;
                    if (localmax < delta)
                        localmax = delta;
                }
                   
                
            }
            for (int i = 1; i < n - 1; i++)
                for (int j = 0; j < n - 1; j++)
                    u[i][j] = u_n[i][j];

        }
        printf("%d %e\n", it, localmax);
        it++;
    }


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%e ", u[i][j]);
        printf("\n");
    }
    printf("iterations: %d", it);
}

int main() {

    two_dim_prog();
    return 0;
}