#!/usr/bin/env python

import numpy as np
from scipy.signal import hilbert
from scipy import integrate


def namodel_py(effadse, wdos, delta, ergy):

    # This is  a python version of the "chemisorption.so" function it
    # is slower but give the same values is more convenient in case
    # only few calculations are required.
    
    fermi = np.argmax(ergy >= 0)
    htwdos = np.imag(hilbert(wdos, axis=0))
    lorentzian = (1/np.pi) * (delta)/((ergy - effadse)**2 + delta**2)
    dos_ads = wdos/((ergy-effadse-htwdos)**2+wdos**2)/np.pi
    chem = wdos/(ergy-effadse-htwdos)
    integrand_1 = np.arctan(chem)
    integrand_2 = [x-np.pi if x > 0 else x for x in integrand_1]
    integ = integrate.cumtrapz(integrand_2, ergy, axis=0)[fermi-1]/np.pi 
    na = integrate.cumtrapz(lorentzian, ergy, axis =0)[fermi-1]
    AvgVal = integrate.cumtrapz(lorentzian * ergy, ergy, axis = 0)[fermi - 1]
    energy_NA = 2*(integ-AvgVal)

    return energy_NA, dos_ads, na
