#!/usr/bin/env python

import numpy as np
from scipy.signal import hilbert
from scipy import integrate

def namodel_py(effadse, wdos, delta, ergy):

    fermi = np.argmax(ergy >= 0)
    htwdos = np.imag(hilbert(wdos, axis=0))
    lorentzian = (1/np.pi) * (delta)/((ergy - effadse)**2 + delta**2)
    dos_ads = wdos/((ergy - effadse - htwdos)**2 + wdos**2)/np.pi
    chem = wdos/(ergy - effadse - htwdos)
    integrand = [x - np.pi if x > 0 else x for x in np.arctan(chem)]
    integ = integrate.cumtrapz(integrand, ergy, axis=0)[fermi - 1]/np.pi #fermi-1 is the index to calculate until fermi level
    na = integrate.cumtrapz(lorentzian, ergy, axis =0)[fermi - 1]

    # need to correct for when beta = 0
    # wdos = delta

    wdos_ = np.array([delta for i in range(0, len(ergy))])
    chem_ = wdos_/(ergy - effadse) # htwdos is just 0
    integrand_ = [x - np.pi if x > 0 else x for x in np.arctan(chem_)]
    integ_ = integrate.cumtrapz(integrand_, ergy, axis=0)[fermi - 1]/np.pi #fermi-1 is the index to calculate until fermi level
    energy_NA = 2*(integ - integ_)

    return energy_NA, dos_ads, na
