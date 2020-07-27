#include<stdio.h>
#include<math.h>
#include<complex.h> //This library is declared before fftw3.h
#include<fftw3.h>

void hilbt_imag(double *ht_output, int N, double *ht_input)
{
  // Output the imaginary components of Hilbert transform
  int i;
  fftw_complex *y_Hi, *y_H;
  fftw_plan plan, plan_i;
  
  y_Hi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N*4);          //allocating memory
  y_H  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N*4);
  plan = fftw_plan_dft_r2c_1d(N, ht_input, y_Hi, FFTW_ESTIMATE);
  plan_i = fftw_plan_dft_1d(N, y_Hi, y_H, FFTW_BACKWARD, FFTW_ESTIMATE);

  fftw_execute(plan);

  for(i = 0; i < N; i++)
    {
      if (N % 2 == 0) {
	if((i == 0) || (i == N/2)) {

	} else if((i > 0) && (i < N/2)) {
	  y_Hi[i] *= 2;
	} else {
	  y_Hi[i] *= 0;
	}
      } else {
	if(i == 0) {
	  
	} else if((i > 0) && (i < (N + 1)/2)) {
	  y_Hi[i] *= 2;
	} else {
	  y_Hi[i] *= 0;
	}
      }
    }

  fftw_execute(plan_i);

  for(i = 0; i < N; i++)
    {
      ht_output[i] = cimag(y_H[i])/N;
    }
  
  fftw_destroy_plan(plan);
  fftw_destroy_plan(plan_i);
  fftw_free(y_Hi);
  fftw_free(y_H);
} 
