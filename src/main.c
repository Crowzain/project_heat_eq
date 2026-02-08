#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "iterative_meth.h"

enum Method_Ax_b {
    JACOBI, 
    GAUSS_SEIDEL
};

enum Method_EDO{
    EULER_IMPLICITE = 1,
    CRANCK_NICOLSON = 2,
};

int MAX_ITER = 10000;

double heat_source(double x, double y){
    return sin(M_PI*x)*sin(M_PI*y);
}

void heat_source_vector(double *vect, int m, int n, int init){
    int offset = (init) ? 0 : 1;
    for (int i=offset; i<m+1+offset; i++){
        for (int j=offset; j<n+1+offset; j++){
            double x = (double)i/(m+1);
            double y = (double)j/(n+1);
            vect[i*(n+2)+j] = heat_source(x, y);
        }
    }
}

 
int main(int argc, char **argv){

    int m = 3;
    int n = 3;
    int T = 100;
    double eps = 1e-10;

    double hx = 1.0/(m+1);
    double hy = 1.0/(n+1);
    double ht = 1.0/(T+1);

    enum Method_Ax_b method_ax_b = JACOBI;
    enum Method_EDO method_edo = 2;
    
    double nu;
    double I;
    if (argc < 3){
        nu = 0.3;
        I = 1.;
    }
    else{
        nu = atof(argv[1]);
        I = atof(argv[2]);
    }
    
    double nu_tilde = nu/method_edo;
    double lambda = ht + 2*nu_tilde/hx/hx + 2*nu_tilde/hy/hy; 
    
    // Allocate solution and heat source vectors
    double *u = (double *)malloc((m+2)*(n+2)*sizeof(double));
    double *f = (double *)malloc((m+2)*(n+2)*sizeof(double));
    
    // Initialize heat source vector
    heat_source_vector(f, m, n, 1);
    heat_source_vector(u, m, n, 1);
    
    FILE* fptr;
    const char * file_name = "data.txt";
    if (access(file_name, F_OK)){
        remove(file_name);
    }
    else{
        fptr = fopen("data.txt", "a+");
    }
    fptr = stdout;
    //fptr = fopen("data.txt", "a");

    switch (method_ax_b)
    {
    case JACOBI:
        jacobi_case: 
            double *new_u = (double *)malloc((m+2)*(n+2)*sizeof(double));
            Jacobi(u, new_u, I, f, m, n, hx, hy, nu_tilde, lambda, MAX_ITER, eps, fptr);
            new_u = NULL;
            free(new_u);
            break;
    case GAUSS_SEIDEL:
        GaussSeidel(u, I, f, m, n, hx, hy, nu_tilde, lambda, MAX_ITER, eps, fptr);
        break;
    default:
        goto jacobi_case;
        break;
    }
    fclose(fptr);
    
    f = NULL;
    u = NULL;

    free(f);
    free(u);
    
    return 0;
}