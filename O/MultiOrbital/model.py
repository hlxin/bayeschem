#!/usr/bin/env python

import pickle, math
import numpy as np
import pymc as pm
from scipy import integrate

from chemisorption import namodel

List = ['Au', 'Ag', 'Cu',
        'Pt', 'Pd', 'Ni',
        'Ir', 'Rh',
        'Co', 'Ru']

Vak2_d0 = [3.35, 2.26, 1,
           3.9, 2.78, 1.16,
           4.45, 3.32,
           1.34, 3.87]
f = [1, 1, 1,
     0.9, 0.9, 0.9,
     0.8, 0.8,
     0.8, 0.7]

ergy = np.linspace(-15, 15, 3001)
fermiindex = np.argmax(ergy >= 0)
dos_d = np.loadtxt('dos_d.txt')
y1_data = np.loadtxt('dos_ads_z.txt')
y2_data = np.loadtxt('dos_ads_x.txt')
y3_data = np.load('E.npy')

InitAdse1 = INIT_Eadse1
InitAdse2 = INIT_Eadse2

InitBeta1 = INIT_beta1
InitBeta2 = INIT_beta2

InitAlpha = INIT_alpha

InitEsp = INIT_Esp

# priors
adse_1 = pm.Normal('adse_1', -5.0, 1, value= InitAdse1)
adse_2 = pm.Normal('adse_2', -5.0, 1, value= InitAdse2)

beta_1 = pm.Lognormal('beta_1', 2, 1.0, value=InitBeta1)
beta_2 = pm.Lognormal('beta_2', 2, 1.0, value=InitBeta2)

delta_1 = pm.Lognormal('delta_1', 1, 0.25, value=1)
delta_2 = pm.Lognormal('delta_2', 1, 0.25, value=1)

alpha = pm.Uniform('alpha', 0, 1.0, value=InitAlpha)

Esp = pm.Normal('Esp', -3.25, 1, value=InitEsp)

var_1 = pm.InverseGamma('var_1', 2.0, 0.05, value=0.05)
var_2 = pm.InverseGamma('var_2', 2.0, 0.1, value=0.1)

a = len(ergy)
@pm.stochastic(observed=True)
def custom_stochastic(adse_1=adse_1, beta_1=beta_1, delta_1=delta_1,
                      adse_2=adse_2, beta_2=beta_2, delta_2=delta_2,
                      alpha=alpha, Esp=Esp,
                      var_1=var_1, var_2=var_2, value=y1_data):
    logp1 = 0
    logp2 = 0
    logp3 = 0
    for i in range(len(List)):
        Vak2_d_1 = Vak2_d0[i]*beta_1
        wdos_1 = np.pi * Vak2_d_1 * dos_d[i] + delta_1
        ergy_NA_1,dos_NA_1, na_1 = namodel(adse_1,wdos_1,
                                           delta_1, ergy)
        dos_NA_1 = dos_NA_1/integrate.cumtrapz(dos_NA_1,ergy,axis=0)[-1]
        Vak2_d_2 = Vak2_d0[i] * beta_2

        wdos_2 = np.pi * Vak2_d_2 * dos_d[i] + delta_2
        ergy_NA_2,dos_NA_2, na_2 = namodel(adse_2, wdos_2,
                                           delta_2, ergy)
        dos_NA_2 = dos_NA_2/integrate.cumtrapz(dos_NA_2,ergy,axis=0)[-1]

        logp1 += -2 * math.log(var_1) * len(ergy) - np.sum((dos_NA_1-value[i])**2 / (2*var_1))
        logp2 += -2 * math.log(var_1) * len(ergy) - np.sum((dos_NA_2-y2_data[i])**2 / (2*var_1))
        BE = Esp + (ergy_NA_1 + 2*(na_1+f[i])*alpha*Vak2_d_1)+(ergy_NA_2 + 2*(na_2+f[i])*alpha*Vak2_d_2)*2

        logp3 += -2*math.log(var_2) - (BE - y3_data[i])**2 / (2*var_2)
    return logp1+logp2+a*logp3


M = pm.MCMC([adse_1, adse_2,
             beta_1, beta_2,
             delta_1, delta_2,
             alpha, Esp,
             var_1, var_2, custom_stochastic],
            db='pickle', dbname='M.pickle')

M.sample(iter=200000, burn=0, thin=1)
M.db.close()
