#!/usr/bin/env python

import numpy as np
from scipy.signal import hilbert
from scipy import integrate

def namodel_py(effadse,wdos,ergy):

    # effadse: effective adsorbate energy level
    # wdos: weighted dos of a substrate
    # ergy: energy grid
    # lambda_sol: energy of reorganization 
    # q: the generalized solvent coordinate
    # eta: overpotential
    # dos_ads: adsorbate dos after interaction                                                    

    fermi = np.argmax(ergy>=0)
    htwdos = np.imag(hilbert(wdos,axis=0))
    dos_ads = wdos/((ergy-effadse-htwdos)**2+wdos**2)/np.pi
    chem = wdos/(ergy-effadse-htwdos)
    integrand_1 = np.arctan(chem)
    integrand_2 = [x-np.pi if x > 0 else x for x in integrand_1]
    integ = integrate.cumtrapz(integrand_2,ergy,axis=0)[fermi-1]/np.pi #fermi-1 is the index to calculate until fermi level
    energy_NA = 2*(integ-effadse)

    return energy_NA,dos_ads, htwdos
