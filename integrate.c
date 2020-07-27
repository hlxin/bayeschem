#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void cumtrapz_1d(double *x, double *y, double *y_int, int N)
{
  double delta = x[1] - x[0];
  y_int[0] = (y[1] + y[0])*delta/2;
  for(int i = 1; i < N - 1; i++)
    {
      delta = x[i + 1] - x[i];
      y_int[i] = y_int[i - 1] + (y[i + 1] + y[i])*delta/2;
    }  
}

double trapz_1d(double *x, double *y, int N)
{
  double delta = x[1] - x[0];
  double sum_y_int = (y[1] + y[0])*delta/2;
  for(int i = 1; i < N - 1; i++)
    {
      delta = x[i + 1] - x[i];
      sum_y_int += (y[i + 1] + y[i])*delta/2;
    }
  return(sum_y_int);
}
