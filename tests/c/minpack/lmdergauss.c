#include <stdio.h>
#include <math.h>
#include <cminpack.h>

/* load data to fit */
#include "xprofile.h"

int Gaussian1d(void * p, int m, int n, const double * params, double * fvec, double * fjac, int ldfjac, int iflag)
{
    double * y = (double *)p;
    double g_0 = params[0];
    double K = params[1];
    double x_0 = params[2];
    double A = params[3];

    if(iflag == 1)
    {
        /* Function */
        int i;
        for(i = 0; i < m; i++)
        {
            double x1 = i - x_0;
            fvec[i] = g_0 + K * exp(-(A * x1 * x1)) - y[i];
        }
    }
    else
    {
        /* Gradient */
        int i;
        for(i = 0; i < m; i++)
        {
            double x1 = i - x_0;
            double x2 = x1 * x1;
            double E = exp(-(A * x2));
            double KE = K * E;
            fjac[i + ldfjac * 0] = 1;
            fjac[i + ldfjac * 1] = E; 
            fjac[i + ldfjac * 2] = 2 * A * x1 * KE;
            fjac[i + ldfjac * 3] = - x2 * KE;
        }
    }
    return 0;
}

#define M 150
#define N 4
#define LWA 5*N+M

int main()
{
    int j, info;
    int ipvt[N];
    double tol, fnorm;
    double x[N], fvec[M], fjac[M*N], wa[LWA];
    int ldfjac = M;
    
    /* guess */
    x[0] = 685.960000;
    x[1] = 5117.200000;
    x[2] = 68.000000;
    x[3] = 0.001920;

    /* set tol to the square root of the machine precision. */
    /* unless high solutions are required, */
    /* this is the recommended setting. */
    tol = sqrt(dpmpar(1));

    /* do fit */
    info = lmder1(Gaussian1d, xprofile, M, N, x, fvec, fjac, ldfjac, tol, ipvt, wa, LWA);

    /* calculate least squares error */
    fnorm = enorm(M, fvec);
    
    printf("final l2 norm of the residual %g\n", fnorm);
    printf("exit parameter %d\n", info);
    printf("final approximate solution\n");
    for (j=0; j<N; j++)
    {
        printf("%g ", x[j]);
    }
    printf("\n");
    
    printf("Known solution is:\n749.201 4329.47 70.9934 0.00131433\n");

    return 0;
}

