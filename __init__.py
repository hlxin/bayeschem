#!/usr/bin/env python

import numpy as np
from scipy import integrate
import pymc as pm
import pickle, math
from chemisorption import namodel
from ase import *

List = ['Co', 'Ni', 'Cu',' Ru',
        'Rh', 'Pd', 'Ag', 'Ir',
        'Pt', 'Au']

Vak2_d0 = [1.34, 1.16, 1, 3.87, 
           3.32, 2.78, 2.26, 4.45, 
           3.9, 3.35]

f = [0.8, 0.9, 1, 0.7,
     0.8, 0.9, 1, 0.8,
     0.9, 1]

ergy = np.linspace(-15, 15, 3001)
fermiindex = np.argmax(ergy >= 0)

# load in data
dos_d = np.loadtxt('dos_d.txt')
y1_data = np.loadtxt('dos_ads.txt')
y2_data = np.load('E.npy')

# priors
Initeffadse = -5.0 #-5.0
Initbeta = 2.1 #2.0
Initdelta = 1.0 #1.0
InitAlpha = 0.036 #.5
InitEsp = -3.25 #-2.0

effadse = pm.Normal('effadse', -5.0, 1, value=Initeffadse)
beta = pm.Lognormal('beta', 2, 1, value=Initbeta)
delta = pm.Lognormal('delta', 1, 0.25, value=Initdelta)
alpha = pm.Uniform('alpha', 0, 1.0, value=InitAlpha)
Esp = pm.Normal('Esp', -3.25, 1, value=InitEsp)

var_1 = pm.InverseGamma('var_1', 2.0, 0.05, value=0.05)
var_2 = pm.InverseGamma('var_2', 2.0, 0.1, value=0.1)

a = len(ergy)
@pm.stochastic(observed=True)
def custom_stochastic(effadse=effadse, beta=beta, delta=delta,
                      alpha=alpha, Esp=Esp,
                      var_1=var_1, var_2=var_2, value=y1_data):
    logp1 = 0
    logp2 = 0
    for i in range(len(List)):
        Vak2_d = Vak2_d0[i]*beta
        wdos = np.pi*Vak2_d*dos_d[i]+delta
        ergy_NA,dos_NA, na = namodel(effadse, wdos,
                                     delta, ergy)
        dos_NA = dos_NA/integrate.cumtrapz(dos_NA, ergy, axis=0)[-1]

        BE = Esp + (ergy_NA + 2*(na+f[i])*alpha*Vak2_d)*3
        logp1 += -2*math.log(var_1) * len(ergy) - np.sum((dos_NA-value[i])**2 / (2*var_1))
        logp2 += -2*math.log(var_2) - (BE - y2_data[i])**2 / (2*var_2)
    return logp1 + a * logp2


M = pm.MCMC([effadse, beta, delta, alpha, Esp,
             var_1, var_2, custom_stochastic],
            db='pickle', dbname='M.pickle')

M.sample(iter=200000, burn=0, thin=1)
M.db.close()
