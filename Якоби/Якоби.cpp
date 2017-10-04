// Якоби.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#include "malloc.h"
#include <omp.h>

#define N 255
#define precision 1e-3f
#define ABSOLUTE

/*
*	функция записи результатов в файл
*/
void write_to_file(const char *fileName, float *u, float xStep, float yStep)
{
	int i, j;
	FILE* out = fopen(fileName, "w");
	float *p = u;
	float x = 0.0f, y = 0.0f;
	for (i = 0; i <= N; i++)
	{
		x += xStep;
		y = 0.0;
		for (j = 0; j <= N; j++)
		{
			y += yStep;
			fprintf(out, "%f %f %f\n", x, y, *p++);
		}
	}
	fclose(out);
}

/*
*	начальное приближение - нулевое
*/
void create_first_approximation(float *u)
{

	float xStep = 1.0f / N;
	float yStep = 1.0f / N;
	int i, j;
	float *p = u;
	for (i = 0; i <= N; i++)
	{
		for (j = 0; j <= N; j++)
		{
			*p++ = 0.0f;
		}
	}

	//ввод граничных условий
	{
		float x = 0.0f, y = 0.0f;
		float *px = u;
		float *py = u;
		for (i = 0; i <= N; i++)
		{
			*(px + i*(N + 1)) = x * x - x + 1.0f;
			*(px + i*(N + 1) + N) = x * x - x + 1.0f;
			*(py + i) = y * y - y + 1.0f;
			*(py + N*(N + 1) + i) = y * y - y + 1.0f;
			x += xStep;
			y += yStep;
		}
	}
}

/*
*   метод Якоби
*/
void jacobi_omp(float *u, float xStep, float yStep, float*uNext, float *ua, float* f, int N1, int N2)
{
	//FILE* resultCalc = fopen("res_calc.txt", "w");
	int i, j;
	float error = 1.0f + precision, e;
	int count = 1;

	while (error > precision)
	{
		error = 0.0f;
		for (i = N1; i <= N2; i++)
		{
			for (j = 1; j < N; j++)
			{
				e = fabs(*(u + i*(N + 1) + j) - *(ua + i*(N + 1) + j));	//погрешность относительно точного

				if (e > error)
					error = e;

				*(uNext + i*(N + 1) + j) = 0.25f * (*(u + (i - 1)*(N + 1) + j) + *(u + (i + 1)*(N + 1) + j) +
					*(u + i*(N + 1) + j - 1) + *(u + i*(N + 1) + j + 1) - xStep * yStep * *(f + i*(N + 1) + j));
			}
		}
		//u = uNext;
		for (i = N1; i <= N2; i++)
		{
			for (j = 1; j < N; j++)
			{
				*(u + i*(N + 1) + j) = *(uNext + i*(N + 1) + j);
			}
		}

		count++;
		//fprintf(resultCalc, "%d -- %f\n", count, error);
		//#pragma omp master
		{
			//printf("%d -- %f\n", count, error);
		}
	}
#pragma omp master
	printf("%d iterations of jacobi method\n", count);
	//fclose(resultCalc);
}

void jacobi(float *u, float xStep, float yStep, float *ua, float* f)
{
	float *uNext = (float *)malloc(sizeof(float) * (N + 1)*(N + 1));
	int i, j;
	float error = 0.05f, e;
	int count = 0;
	float *p = u;
	float x = 0.0f, y = 0.0f;

	time_t t0 = clock();

	create_first_approximation(uNext);
	while (error > precision)
	{
		error = 0.0f;
		for (i = 1; i < N; i++)
		{
			for (j = 1; j < N; j++)
			{
				*(uNext + i*(N + 1) + j) = 0.25f * (*(u + (i - 1)*(N + 1) + j) + *(u + (i + 1)*(N + 1) + j) +
					*(u + i*(N + 1) + j - 1) + *(u + i*(N + 1) + j + 1) - xStep * yStep * *(f + i*(N + 1) + j));

#ifdef ABSOLUTE
				e = fabs(*(uNext + i*(N + 1) + j) - *(ua + i*(N + 1) + j));	//погрешность относительно точного
#else
				e = fabs(*(uNext + i*(N + 1) + j) - *(u + i*(N + 1) + j)); //погрешность относительно предыдущего шага
#endif
				if (e > error)
					error = e;
			}
		}
		//u = uNext;
		for (i = 1; i < N; i++)
		{
			for (j = 1; j < N; j++)
			{
				*(u + i*(N + 1) + j) = *(uNext + i*(N + 1) + j);
			}
		}
		count++;
		//printf("%d -- %f\n", count, error);
	}
	printf("jacobi time =  %f sec, iterations = %d\n", (double)(clock() - t0) / CLOCKS_PER_SEC, count);


	write_to_file("result_jacobi.dat", u, xStep, yStep);
	free(uNext);
}

/*
*	метод гаусса-зейделя
*/
void gauss_zeidel_omp(float *u, float xStep, float yStep, float *ua, float* f, int Nx1, int Nx2)
{
	int i, j;
	float error = 0.05f, e, curValue;
	int count = 0;
	float *p = u;
	float x = 0.0f, y = 0.0f;

	while (error > precision)
	{
		error = 0.0f;
		for (i = Nx1; i <= Nx2; i++)
		{
			for (j = 1; j < N; j++)
			{
				curValue = *(u + i*(N + 1) + j);
				*(u + i*(N + 1) + j) = 0.25f * (*(u + (i - 1)*(N + 1) + j) + *(u + (i + 1)*(N + 1) + j) +
					*(u + i*(N + 1) + j - 1) + *(u + i*(N + 1) + j + 1) - xStep * yStep * *(f + i*(N + 1) + j));

				e = fabs(*(u + i*(N + 1) + j) - *(ua + i*(N + 1) + j));	 //погрешность относительно точного
				if (e > error)
					error = e;
			}
		}
		count++;
		//printf("%d -- %f\n", count, error);
	}
#pragma omp master
	printf("omp gauss-zeidel method %d iterations\n", count);
}


void gauss_zeidel(float *u, float xStep, float yStep, float *ua, float* f)
{
	int i, j;
	float error = 0.05f, e, curValue;
	int count = 0;
	float *p = u;
	float x = 0.0f, y = 0.0f;

	time_t t0 = clock();
	while (error > precision)
	{
		error = 0.0f;
		for (i = 1; i < N; i++)
		{
			for (j = 1; j < N; j++)
			{
				curValue = *(u + i*(N + 1) + j);
				*(u + i*(N + 1) + j) = 0.25f * (*(u + (i - 1)*(N + 1) + j) + *(u + (i + 1)*(N + 1) + j) +
					*(u + i*(N + 1) + j - 1) + *(u + i*(N + 1) + j + 1) - xStep * yStep * *(f + i*(N + 1) + j));
#ifdef ABSOLUTE	
				e = fabs(*(u + i*(N + 1) + j) - *(ua + i*(N + 1) + j));	 //погрешность относительно точного
#else
				e = fabs(*(u + i*(N + 1) + j) - curValue);	//погрешность относительно предыдущего шага
#endif
				if (e > error)
					error = e;
			}
		}
		count++;
		//printf("%d -- %f\n", count, error);
	}
	printf("gauss_zeidel time =  %f sec, iterations = %d\n", (double)(clock() - t0) / CLOCKS_PER_SEC, count);


	write_to_file("result_gz.dat", u, xStep, yStep);
}

/*
*   метод релаксации
*/
void relax_omp(float *u, float xStep, float yStep, float *ua, float* f, float w, int Nx1, int Nx2)
{
	int i, j;
	float error = 0.05f, e, curValue;
	int count = 0;
	float *p = u;
	float x, y;
	float cnst1 = w*0.25f, cnst2 = (1 - w);

	while (error > precision)
	{
		error = 0.0f;
		for (i = Nx1; i <= Nx2; i++)
		{
			for (j = 1; j < N; j++)
			{
				curValue = *(u + i*(N + 1) + j);
				*(u + i*(N + 1) + j) = cnst1 * (*(u + (i - 1)*(N + 1) + j) + *(u + (i + 1)*(N + 1) + j) +
					*(u + i*(N + 1) + j - 1) + *(u + i*(N + 1) + j + 1) - xStep * yStep * *(f + i*(N + 1) + j)) + *(u + i*(N + 1) + j)*cnst2;

				e = fabs(*(u + i*(N + 1) + j) - *(ua + i*(N + 1) + j));  //погрешность относительно точного
				if (e > error)
					error = e;
			}
		}
		count++;
		//printf("%d -- %f\n", count, error);
	}
#pragma omp master
	printf("omp relax %d iterations\n", count);
}


void relax(float *u, float xStep, float yStep, float *ua, float* f, float w)
{
	int i, j;
	float error = 0.05f, e, curValue;
	int count = 0;
	float *p = u;
	float x, y;
	float cnst1 = w*0.25f, cnst2 = (1 - w);
	time_t t0 = clock();
	while (error > precision)
	{
		error = 0.0f;
		for (i = 1; i < N; i++)
		{
			for (j = 1; j < N; j++)
			{
				curValue = *(u + i*(N + 1) + j);
				*(u + i*(N + 1) + j) = cnst1 * (*(u + (i - 1)*(N + 1) + j) + *(u + (i + 1)*(N + 1) + j) +
					*(u + i*(N + 1) + j - 1) + *(u + i*(N + 1) + j + 1) - xStep * yStep * *(f + i*(N + 1) + j)) + *(u + i*(N + 1) + j)*cnst2;
#ifdef ABSOLUTE
				e = fabs(*(u + i*(N + 1) + j) - *(ua + i*(N + 1) + j));  //погрешность относительно точного
#else
				e = fabs(*(u + i*(N + 1) + j) - curValue); //погрешность относительно предыдущего шага
#endif
				if (e > error)
					error = e;
			}
		}
		count++;
		//printf("%d -- %f\n", count, error);
	}
	printf("relax time =  %f sec, iterations = %d\n", (double)(clock() - t0) / CLOCKS_PER_SEC, count);

	write_to_file("result_relax2.dat", u, xStep, yStep);
}


/*
* расчет средней относительной погрешности
*/
float calculate_error(float *u, float *ua)
{
	int i, j;
	float err = 0.0f;
	float *pu = u, *pua = ua;
	int count = 0;
	for (i = 0; i <= N; i++)
	{
		for (j = 0; j <= N; j++)
		{
			err += fabs(*pu++ - *pua++) / *pua;
		}
	}
	return err / count;
}



int main()
{
	int i, j;
	float xStep = 1.0f / N;
	float yStep = 1.0f / N;

	float *ua = (float *)malloc(sizeof(float) * (N + 1)*(N + 1)); //одномерный массив, в котором хранится матрица точного решения
	float *u = (float *)malloc(sizeof(float) * (N + 1)*(N + 1)); //одномерный массив, в котором хранится матрица теущего приближения
	float *f = (float *)malloc(sizeof(float) * (N + 1)*(N + 1)); //массив значений правой части
	float *uNext = (float *)malloc(sizeof(float) * (N + 1)*(N + 1));

	float x = 0.0f, y = 0.0f, error;

	//float *pa = ua;
	//float *pf = f;

	//расчет точного решения, правой части
	{
		x = 0.0f;
		for (i = 0; i <= N; i++)
		{
			y = 0.0f;
			for (j = 0; j <= N; j++)
			{
				*(ua + i*(N + 1) + j) = (x * x - x + 1.0f) * (y * y - y + 1.0f);
				*(f + i*(N + 1) + j) = 4.0f + 2 * x * x - 2 * x + 2 * y * y - 2 * y;
				y += yStep;
			}
			x += xStep;
		}
	}

	printf("openmp threads number: %d\n\n", omp_get_max_threads());

	create_first_approximation(u);
	create_first_approximation(uNext);
	const int cores = 4;

	int edge[4];
	edge[0] = 1;
	for (i = 1; i < cores; i++) {
		edge[i] = i*N / cores;
	}

	create_first_approximation(u);
	printf("start jacobi method ...\n");
	jacobi(u, xStep, yStep, ua, f);

	printf("start jacobi omp method with %d threads...\n", cores);
	create_first_approximation(u);
	time_t t0 = clock();

#pragma omp parallel num_threads(cores)
	{
#pragma omp sections
	{
#pragma omp section
	{
		jacobi_omp(u, xStep, yStep, uNext, ua, f, edge[0], edge[1]);
	}
#pragma omp section
	{
		jacobi_omp(u, xStep, yStep, uNext, ua, f, edge[1], edge[2]);
	}
#pragma omp section
	{
		jacobi_omp(u, xStep, yStep, uNext, ua, f, edge[2], edge[3]);
	}
#pragma omp section
	{
		jacobi_omp(u, xStep, yStep, uNext, ua, f, edge[3], N - 1);
	}
	}
	}
	printf("jacobi omp time =  %f sec\n\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
	write_to_file("result_jacobi_omp.dat", u, xStep, yStep);

	create_first_approximation(u);
	printf("start gauss zeidel method\n");
	gauss_zeidel(u, xStep, yStep, ua, f);

	create_first_approximation(u);
	printf("start gauss zeidel OpenMP method\n");

	t0 = clock();

#pragma omp parallel num_threads(cores)
	{
#pragma omp sections
	{
#pragma omp section
	{
		gauss_zeidel_omp(u, xStep, yStep, ua, f, edge[0], edge[1]);
	}
#pragma omp section
	{
		gauss_zeidel_omp(u, xStep, yStep, ua, f, edge[1], edge[2]);
	}
#pragma omp section
	{
		gauss_zeidel_omp(u, xStep, yStep, ua, f, edge[2], edge[3]);
	}
#pragma omp section
	{
		gauss_zeidel_omp(u, xStep, yStep, ua, f, edge[3], N - 1);
	}
	}
	}

	printf("gauss_zeidel time =  %f sec\n\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
	write_to_file("result_gaus-zeidel_omp.dat", u, xStep, yStep);


	printf("start relax method\n");
	create_first_approximation(u);
	relax(u, xStep, yStep, ua, f, 1.9f);

	create_first_approximation(u);
	printf("start relax OpenMP method\n");

	t0 = clock();

#pragma omp parallel num_threads(cores)
	{
#pragma omp sections
	{
#pragma omp section
	{
		relax_omp(u, xStep, yStep, ua, f, 1.9f, edge[0], edge[1]);
	}
#pragma omp section
	{
		relax_omp(u, xStep, yStep, ua, f, 1.9f, edge[1], edge[2]);
	}
#pragma omp section
	{
		relax_omp(u, xStep, yStep, ua, f, 1.9f, edge[2], edge[3]);
	}
#pragma omp section
	{
		relax_omp(u, xStep, yStep, ua, f, 1.9f, edge[3], N - 1);
	}
	}
	}

	printf("relax omp time =  %f sec\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
	write_to_file("result_relax_omp.dat", u, xStep, yStep);


	free(u);
	free(ua);
	free(uNext);
	free(f);

	return 0;
}