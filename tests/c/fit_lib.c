#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Test of the Levenberg-Marquardt nonlinear least squares algorithm in C */

int GetM(double * A);
int GetN(double * A);

/* user function */

void Gaussian1d(double * params, double * xi, double * y)
{
    double g_0 = params[0];
    double K = params[1];
    double x_0 = params[2];
    double A = params[3];
    int M = GetM(y);
    int m;
    for(m = 0; m < M; m++)
    {
        double x1 = xi[m] - x_0;
        y[m] = g_0 + K * exp(-(A * x1 * x1));
    }
}

void Gaussian1dJacobian(double * params, double * xi, double * J)
{
    /* J(M, 4) */
    double K = params[1];
    double x_0 = params[2];
    double A = params[3];
    int M = GetM(J);
    int N = GetN(J);
    int m;
    for(m = 0; m < M; m++)
    {
        double x1 = xi[m] - x_0;
        double x2 = x1 * x1;
        double E = exp(-(A * x2));
        double KE = K * E;
        J[N * m + 0] = 1;
        J[N * m + 1] = E; 
        J[N * m + 2] = 2 * A * x1 * KE;
        J[N * m + 3] = - x2 * KE;
    }
}

/* 
   store the dimensions at the start of the array 
*/
double * MatrixAllocate(int m, int n)
{
    double * xs = (double *)calloc(m * n + 2, sizeof(double));
    xs[0] = m;
    xs[1] = n;
    return xs + 2;
}

int GetM(double * A)
{
    return A[-2];
}

int GetN(double * A)
{
    return A[-1];
}

void ShowMatrix(double * A)
{
    int M = GetM(A);
    int N = GetN(A);
    int m;
    int n;
    for(m = 0; m < M; m++)
    {
        for(n = 0; n < N; n++)
        {
            printf("%g ", A[m * N + n]);
        }
        printf("\n");
    }
}

/* 
   Matrix-Vector routines 
   (named after BLAS routines but standalone)
*/

void MatrixSquare(double * J, double * B)
{
    /* 
       A = J'J
       J(m,n)
       B(n,n)
    */
    int M = GetM(J);
    int N = GetN(J);
    int k;
    int m;
    int n;
    for(k = 0; k < N; k++)
    {
        for(n = 0; n < N; n++)
        {
            B[n + N * k] = 0;
            for(m = 0; m < M; m++)
            {
                B[n + N * k] += J[m * N + n] * J[m * N + k];
            }
        }
    }
}

void MatrixTVector(double * A, double * x, double * b)
{
    /* b = A'x */
    int M = GetM(A);
    int N = GetN(A);
    int m;
    int n;
    for(n = 0; n < N; n++)
    {
        b[n] = 0;
        for(m = 0; m < M; m++)
        {
            b[n] += A[m * N + n] * x[m];
        }
    }
}

void MatrixVector(double * A, double * x, double * b)
{
    /* b = Ax */
    int M = GetM(A);
    int N = GetN(A);
    int m;
    int n;
    for(m = 0; m < M; m++)
    {
        b[m] = 0;
        for(n = 0; n < N; n++)
        {
            b[m] += A[m * N + n] * x[n];
        }
    }
}

double VectorDot(double * a, double * b)
{
    /* b = aa */
    int size = GetM(a);
    int n;
    double sum = 0;
    for(n = 0; n < size; n++)
    {
        sum += a[n] * b[n];
    }
    return sum;
}

void Scal(double * y, double a)
{
    /* y = ay */
    int size = GetM(y);
    int n;
    for(n = 0; n < size; n++)
    {
        y[n] = a * y[n];
    }
}

void Axpy(double * y, double a, double * x)
{
    /* y = ax + y */
    int size = GetM(y);
    int n;
    for(n = 0; n < size; n++)
    {
        y[n] = a * x[n] + y[n];
    }
}

void Cg(double * A, double * b, double * x, double * p, double * r, double * w, double tol)
{
    
    /*
      Ax = b
      A is SPD
      solve for x
      Conjugate Gradient Method
    */

    int size = GetM(A);

    Scal(x, 0.0);
    Scal(p, 0.0);
    Scal(r, 0.0);
    Scal(w, 0.0);

    Axpy(r, 1.0, b);
    Axpy(p, 1.0, r);
    double p_0 = VectorDot(r, r);
    
    int k = 0;
    /* 
       if we don't converge in 2 * size(A)
       we are too ill-conditioned
    */
    while(k < size * 2)
    {
        MatrixVector(A, p, w);
        double alpha = p_0 / VectorDot(p, w);
        Axpy(x,  alpha, p);
        Axpy(r, -alpha, w);
        double p_1 = VectorDot(r, r);
        if(p_1 < tol * tol * VectorDot(b, b))
        {
            break;
        }
        double beta = p_1 / p_0;
        Scal(p, beta);
        Axpy(p, 1.0, r);
        p_0 = p_1;
        k++;
    }

}

/* Levenberg Marquardt */

void LmFit()
{

    /* load test data */

    int MAXP = 1024;
    double * xp = MatrixAllocate(MAXP, 1);
    char * filename = "xprofile.txt";
    FILE * f = fopen(filename, "r");
    if(f == NULL)
    {
        fprintf(stderr, "can't open file %s\n", filename);
        exit(1);
    }
    int n;
    for(n = 0; n < MAXP; n++)
    {
        double x = 0;
        int num = fscanf(f, "%lf", &x);
        if(num != 1)
        {
            break;
        }
        xp[n] = x;
    }

    int np = n;
    int N = 4;
    
    /* independent variable */
    double * xi = MatrixAllocate(np, 1);
    /* function value */
    double * yp = MatrixAllocate(np, 1);
    /* Jacobian matrix, gradient of function */
    double * J = MatrixAllocate(np, N);
    /* LM normal equation matrices */
    double * alpha = MatrixAllocate(N, N);
    double * beta = MatrixAllocate(N, 1);
    /* parameter and parameter change */
    double * a = MatrixAllocate(N, 1);
    double * da = MatrixAllocate(N, 1);
    /* storage for linear solver */
    double * p = MatrixAllocate(N, 1);
    double * r = MatrixAllocate(N, 1);
    double * w = MatrixAllocate(N, 1);

    /* guess */
    a[0] = 685.960000;
    a[1] = 5117.200000;
    a[2] = 68.000000;
    a[3] = 0.001920;
    
    for(n = 0; n < np; n++)
    {
        xi[n] = n;
    }
    
    double tol = 1e-9;
    double lam = 0.01;
    
    int i;
    for(i = 0; i < 10; i++)
    {

        /* evaluate function and gradient */
        Gaussian1d(a, xi, yp);
        Gaussian1dJacobian(a, xi, J);

        /* evaluate gradient of chi-squared */
        for(n = 0; n < np; n++)
        {
            yp[n] = xp[n] - yp[n];
        }
        MatrixSquare(J, alpha);
        MatrixTVector(J, yp, beta);
        
        /* Damping */
        for(n = 0; n < N; n++)
        {
            alpha[n + N * n] *= (1.0 + lam);
        }
        
        /* Linear Solver */
        Cg(alpha, beta, da, p, r, w, tol);
        
        /* Update Parameters */
        for(n = 0; n < N; n++)
        {
            a[n] = a[n] + da[n];
        }
        
        printf("iteration %d parameters %g %g %g %g\n", i, a[0], a[1], a[2], 1.0 / sqrt(2.0*a[3]));
    }

    /* 
       TODO
       function pointers
       convergence test and lambda update
    */
    
}

int main()
{
    LmFit();
    return 0;
}
