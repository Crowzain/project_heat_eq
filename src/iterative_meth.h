#ifndef iterative_meth_2026_77e07cc7521743a6943ca61d60e29a0423f4c324dacef2619a2112f8b887e9ba
#define iterative_meth_2026_77e07cc7521743a6943ca61d60e29a0423f4c324dacef2619a2112f8b887e9ba
#include<stdio.h>

void print_vector(double*, int, int, FILE*);

void Jacobi(
    double *, double *, double, double *, int, int,
    double, double, double, double,
    int, double, FILE*);

void GaussSeidel(
    double *, double, double *, int, int,
    double, double, double, double,
    int, double, FILE*);


#endif