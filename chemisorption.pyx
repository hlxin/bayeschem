#!/usr/bin/env python

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, M_PI, atan

# define function pointer                                                                                             
ctypedef double (*trapz_ptr)(double[:] y, double[:] x, np.intp_t f_ind)

# export function from an external .C file
cdef extern from "hilbert.h":
    void hilbt_imag(double *ht_output, int N, double *ht_input)

@cython.boundscheck(False) # turn off bound-checking for entire function                                              
@cython.wraparound(False) # turn off negative index wrapping for entire function                                       
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double trapz_1d(double[:] y, double[:] x, np.intp_t f_ind):
    cdef np.intp_t i
    cdef double trap_sum, delta
    trap_sum = 0
    for i in range(f_ind):
        delta = x[i + 1] - x[i]
        trap_sum += (y[i + 1] + y[i])*delta/2

    return trap_sum

@cython.boundscheck(False) # turn off bound-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef namodel(double effadse,
              double[:] wdos,
              double delta, # the sp constant
              np.ndarray[np.float64_t, ndim=1] ergy):

    cdef np.intp_t fermi = np.argmax(ergy>=0)
    cdef np.intp_t N_ergy = ergy.shape[0]
    cdef np.intp_t j

    cdef double[:] htwdos = np.empty(N_ergy)
    cdef double* htwdos_p = &htwdos[0]
    cdef double* wdos_p = &wdos[0]

    hilbt_imag(htwdos_p, N_ergy, wdos_p) # performing Hilbert transform on wdos and subsequently
                                         # taking its imaginary components

    cdef double[:] lorentzian = np.empty(N_ergy)
    cdef double[:] Avg_Lorz = np.empty(N_ergy)
    cdef double[:] dos_ads = np.empty(N_ergy)
    cdef double[:] integrand = np.empty(N_ergy)
    cdef double diff_e 
    cdef double diff_e_lorz

    for j in range(N_ergy):
        diff_e = ergy[j] - effadse - htwdos[j]
        diff_e_lorz = ergy[j] - effadse

# get the lorentzian
        lorentzian[j] = (1/M_PI) * (delta)/(diff_e_lorz*diff_e_lorz + delta*delta)

# multiple the lorenzian * e 
        Avg_Lorz[j] = lorentzian[j] * ergy[j]
        dos_ads[j] = wdos[j]/(diff_e*diff_e + wdos[j]*wdos[j])/M_PI
        integrand[j] = atan(wdos[j]/diff_e)

        if integrand[j] > 0:
            integrand[j] = integrand[j] - M_PI


    cdef trapz_ptr int_fcn = &trapz_1d

    na = int_fcn(lorentzian, ergy, fermi) # integrate of lorenzian de up to fermi level 
    Avg_val = int_fcn(Avg_Lorz, ergy, fermi) # integration of lorz * e de up to fermi level
    integ = int_fcn(integrand, ergy, fermi)/M_PI
    energy_NA = 2*(integ - Avg_val)

    return energy_NA, np.asarray(dos_ads), na
