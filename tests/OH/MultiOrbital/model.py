#!/usr/bin/env python

import numpy as np
from scipy import integrate
import pymc as pm
import pickle, math

from chemisorption import namodel

Vak2_d0 = np.loadtxt('Vad.txt')
f = np.loadtxt('f.txt')
ergy = np.linspace(-15,15,3001)
fermiindex=np.argmax(ergy>=0)
dos_d = np.loadtxt('dos_d.txt')
y1_data = np.loadtxt('dos_ads_1.txt')
y2_data = np.loadtxt('dos_ads_2.txt')
y3_data = np.loadtxt('dos_ads_3.txt')
y4_data = np.loadtxt('BE.txt')
# priors
Initeffadse1 = -6
Initeffadse2 = -2
Initeffadse3 = 4

Initbeta1 = 1.0
Initbeta2 = 2.0
Initbeta3 = 1.0

InitAlpha = 0.1

InitEsp = -3.15

adse_1 = pm.Normal('adse_1', -6.0, 1.0, value=Initeffadse1) 
adse_2 = pm.Normal('adse_2', -2.0, 1.0, value=Initeffadse2)
adse_3 = pm.Normal('adse_3', 4.0, 1.0, value=Initeffadse3)

beta_1 = pm.Lognormal('beta_1', 2.0, 0.25, value=Initbeta1)
beta_2 = pm.Lognormal('beta_2', 2.0, 0.25, value=Initbeta2)
beta_3 = pm.Lognormal('beta_3', 2.0, 0.25, value=Initbeta3)

delta_1 = pm.Lognormal('delta_1', 1, 0.25, value=1)
delta_2 = pm.Lognormal('delta_2', 1, 0.25, value=1)
delta_3 = pm.Lognormal('delta_3', 1, 0.25, value=1)

alpha = pm.Uniform('alpha', 0, 1.0, value=InitAlpha)

Esp = pm.Normal('Esp', -3.13, 1.0, value=InitEsp)

var_1 = pm.InverseGamma('var_1', 2.0, 0.05, value=0.1)
var_2 = pm.InverseGamma('var_2', 2.0, 0.1, value=0.1)
a = 100

@pm.stochastic(observed=True)
def custom_stochastic(adse_1 = adse_1, beta_1 = beta_1, delta_1 = delta_1,
                      adse_2 = adse_2, beta_2 = beta_2, delta_2 = delta_2,
                      adse_3 = adse_3, beta_3 = beta_3, delta_3 = delta_3,
                      alpha = alpha, Esp = Esp, 
                      var_1=var_1, var_2=var_2, value=y1_data):
                      
    logp1 = 0
    logp2 = 0
    logp3 = 0
    logp4 = 0
    for i in range(len(f)):
        Vak2_d_1 = Vak2_d0[i] * beta_1
        wdos_1 = np.pi * Vak2_d_1 * dos_d[i] + delta_1
        ergy_NA_1, dos_NA_1, na_1 = namodel(adse_1, wdos_1, 
                                            delta_1, ergy)
        dos_NA_1 = dos_NA_1/integrate.cumtrapz(dos_NA_1, ergy, axis=0)[-1]

        Vak2_d_2 = Vak2_d0[i] * beta_2
        wdos_2 = np.pi * Vak2_d_2 * dos_d[i] + delta_2
        ergy_NA_2, dos_NA_2, na_2 = namodel(adse_2, wdos_2, 
                                            delta_2, ergy)
        dos_NA_2 = dos_NA_2/integrate.cumtrapz(dos_NA_2, ergy, axis=0)[-1]

        Vak2_d_3 = Vak2_d0[i]*beta_3
        wdos_3 = np.pi * Vak2_d_3 * dos_d[i] + delta_3
        ergy_NA_3, dos_NA_3, na_3 = namodel(adse_3,wdos_3,
                                            delta_3 ,ergy)
        dos_NA_3 = dos_NA_3/integrate.cumtrapz(dos_NA_3, ergy, axis=0)[-1]

        logp1 += -2*math.log(var_1)*len(ergy) - np.sum((dos_NA_1-value[i])**2 / (2*var_1))
        logp2 += -2*math.log(var_1)*len(ergy) - np.sum((dos_NA_2-y2_data[i])**2 / (2*var_1))
        logp3 += -2*math.log(var_1)*len(ergy) - np.sum((dos_NA_3-y3_data[i])**2 / (2*var_1))

        BE = Esp + (ergy_NA_1 + 2*(na_1+f[i])*alpha*Vak2_d_1) + 2 * (ergy_NA_2 + 2*(na_2+f[i])*alpha*Vak2_d_2) + (ergy_NA_3 + 2*(na_3+f[i])*alpha*Vak2_d_3)
        logp4 += -2*math.log(var_2) - (BE - y4_data[i])**2 / (2*var_2)
    return logp1+logp2+logp3+a*logp4


M = pm.MCMC([adse_1, adse_2, adse_3,
             beta_1, beta_2, beta_3,
             delta_1, delta_2, delta_3,
             alpha, Esp,
             var_1, var_2, custom_stochastic],
            db='pickle', dbname='M.pickle')

M.sample(iter=200000, burn=0, thin=1)
M.db.close()
