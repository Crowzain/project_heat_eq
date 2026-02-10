#include "iterative_meth.h"

inline void print_vector(double* u, int m, int n, FILE* f_ptr){
    int np2 = n+2;
    int size = (m+2)*np2;
    fwrite(u, sizeof(double), size, f_ptr?f_ptr:stdout);
}

void Jacobi(
    double *restrict u, double *restrict new_u, double I, double *restrict f, int m, 
    int n, double hx, double hy, double nu_tilde, 
    double lambda, int max_iter, double tol, FILE* fptr
){
    double tmp_val;
    double* tmp_ptr;
    double el_squared_sum;
    double diff_squared_sum;
    int iter = 0;
    int flag = 0;
    int np2  = n+2;
    do{
        iter++;
        el_squared_sum = 0.0;
        diff_squared_sum = 0.0;
        #pragma omp parallel
        #pragma omp for
        for (int i=1; i<m+1; i++){
            #pragma omp for
            for (int j=1; j<n+1; j++){
                
                tmp_val = 
                    I*f[i*np2+j]/lambda+
                    nu_tilde/lambda/hy/hy*u[(i-1)*np2+j] +
                    nu_tilde/lambda/hy/hy*u[(i+1)*np2+j] +
                    nu_tilde/lambda/hx/hx*u[i*np2+(j-1)] + 
                    nu_tilde/lambda/hx/hx*u[i*np2+(j+1)];
                    if (flag==0){
                        diff_squared_sum += (u[i*np2+j]-tmp_val)*(u[i*np2+j]-tmp_val);
                    }
                    else{
                        diff_squared_sum += (new_u[i*np2+j]-tmp_val)*(new_u[i*np2+j]-tmp_val);
                    }
                    el_squared_sum += tmp_val*tmp_val;
                new_u[i*np2+j] = tmp_val;
            }
        }
        flag = 1;
        print_vector(new_u, m, n, fptr);
        tmp_ptr = u;
        u = new_u;
        new_u = tmp_ptr;
    }while (diff_squared_sum>=tol*el_squared_sum && iter<max_iter);//
}



void GaussSeidel(
    double *restrict u, double I, double *restrict f, int m, int n,
    double hx, double hy, double nu_tilde, double lambda,
    int max_iter, double tol, FILE* fptr
){
    double tmp;
    double el_squared_sum;
    double diff_squared_sum;
    int iter = 0;
    int np2  = n+2;
    do{
        iter++;
        el_squared_sum = 0.0;
        diff_squared_sum = 0.0;
        for (int i=1; i<m+1; i++){
            for (int j=1; j<n+1; j++){
                tmp = 
                    I*f[i*np2+j]/lambda+
                    nu_tilde/lambda/hy/hy*u[(i-1)*np2+j] +
                    nu_tilde/lambda/hy/hy*u[(i+1)*np2+j] +
                    nu_tilde/lambda/hx/hx*u[i*np2+(j-1)] + 
                    nu_tilde/lambda/hx/hx*u[i*np2+(j+1)];
                
                    diff_squared_sum += (u[i*np2+j]-tmp)*(u[i*np2+j]-tmp);
                u[i*np2+j]=tmp;
                el_squared_sum += tmp*tmp;
            }
        }
        print_vector(u, m, n, fptr);
    }while (diff_squared_sum<tol*el_squared_sum && iter<max_iter);
}