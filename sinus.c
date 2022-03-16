#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define pi M_PI

int main(void)
{
	double *sinus_arr = (double*)malloc(sizeof(double) * iter);
	double sum_sin = 0;

#pragma acc kernels
{
	for (int i = 0; i < 10000000; i++)
		sinus_arr[i] = sin(2 * (double)i * pi / iter);

	for (int i = 0; i < 10000000; i++)
		sum_sin += sinus_arr[i];
}
	printf("%lf\n", sum_sin);
	return 0;
}


