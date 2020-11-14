#!/usr/bin/env python

import numpy as np
from scipy import integrate
from ase.db import connect
import math
import pymc as pm

from chemisorption import namodel

def interpolate_nrg(pdos, energy_, energy):
     dos = [np.interp(e, energy_, pdos) if np.max(energy_) > e > np.min(energy_) else 0 for e in energy]
     dos = dos/integrate.cumtrapz(dos, energy, axis=0)[-1]
     return dos
        

# load in data 
db = connect('bayeschem.db')
energy = np.linspace(-15, 15, 3001)
labels = []
dE_dfts = []
vad2s = []
fillings = []
dos_ds = []
dos_adss = []

for row in db.select():

    labels.append(row.label)
    dE_dfts.append(row.data['de'])
    vad2s.append(row.vad2)
    fillings.append(row.filling)

    dos_sub = row.data['dos_sub'][1][0] + row.data['dos_sub'][1][1]
    dos_sub_energy = row.data['dos_sub'][0]
    dos_sub = interpolate_nrg(dos_sub, dos_sub_energy, energy)
    dos_ds.append(dos_sub)

    dos_ads = row.data['dos_ads'][1][0] + row.data['dos_ads'][1][1]
    dos_ads_energy = row.data['dos_ads'][0]
    dos_ads = interpolate_nrg(dos_ads, dos_ads_energy, energy)
    dos_adss.append(dos_ads)

# priors
dE_0 = pm.Normal('dE_0', -3.25, 1, value = -3.25)
eps_a = pm.Normal('eps_a', -5.0, 1, value = -5.0)
delta_0 = pm.Lognormal('delta_0', 1, 0.25, value = 1.0)
alpha = pm.Uniform('alpha', 0, 1.0, value = 0.036)
beta = pm.Lognormal('beta', 2, 1, value = 2.1)

var_1 = pm.InverseGamma('var_1', 2.0, 0.05, value = 0.05)
var_2 = pm.InverseGamma('var_2', 2.0, 0.1, value = 0.1)

lamb = .01
@pm.stochastic(observed = True)
def custom_stochastic(eps_a = eps_a, beta = beta, delta_0 = delta_0,
                      alpha = alpha, dE_0 = dE_0, var_1 = var_1, var_2 = var_2,
                      value = dos_adss):

    logp_de = 0
    logp_ados = 0
    for count in range(len(labels)):
        vad2 = vad2s[count]
        filling = fillings[count]
        dos_d = dos_ds[count]
        dos_ads_dft = value[count]
        dE_dft = dE_dfts[count]

        delta = delta_0 + np.pi * beta * vad2 * dos_d
        delta_ = np.array([delta_0 for k in range(0, len(energy))])
        dE_d_hyb, dos_ads_na, na = namodel(eps_a, delta, delta_,
                                           delta_0, energy)
        dos_ads_na = dos_ads_na/integrate.cumtrapz(dos_ads_na, energy, axis=0)[-1]

        dE_d = dE_d_hyb + 2 * (na + filling) * alpha *  beta * vad2
        dE_na = dE_0 + 3 * dE_d # 3 degenerate orbitals 

        logp_ados += -2 * math.log(var_1) * len(energy) - np.sum((dos_ads_na - dos_ads_dft)**2 / (2*var_1))
        logp_dE += -2 * math.log(var_2) - (dE_na - dE_dft)**2 / (2*var_2)

    return logp_dE + lamb * logp_ados
    
M = pm.MCMC([eps_a, beta, delta_0,
             alpha, dE_0,
             var_1, var_2, custom_stochastic],
            db='pickle', 
            dbname='M.pickle')

M.sample(iter=200000, burn=0, thin=1)
M.db.close()
