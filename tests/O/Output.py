#!/usr/bin/env python

import numpy as np
from scipy import integrate
from ase.db import connect
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

for row in db.select():

    labels.append(row.label)
    dE_dfts.append(row.data['de'])
    vad2s.append(row.vad2)
    fillings.append(row.filling)

    dos_sub = row.data['dos_sub'][1][0] + row.data['dos_sub'][1][1]
    dos_sub_energy = row.data['dos_sub'][0]
    dos_sub = interpolate_nrg(dos_sub, dos_sub_energy, energy)
    dos_ds.append(dos_sub)

vars = ['dE_0', 'eps_a',
        'delta_0', 'alpha', 
        'beta']

db = pm.database.pickle.load('M.pickle')
num = len(db.trace(vars[0])[:])/2

vars_s = {}
for var in vars:
    vars_s[var] = [db.trace(var)[num:].mean(), db.trace(var)[num:].std()]

print vars_s

# priors                                                                        
dE_0 = vars_s['dE_0'][0]
eps_a = vars_s['eps_a'][0]
delta_0 = vars_s['delta_0'][0]
alpha = vars_s['alpha'][0]
beta = vars_s['beta'][0]

dE_nas = []

for count in range(len(labels)):
    vad2 = vad2s[count]
    filling = fillings[count]
    dos_d = dos_ds[count]

    delta = delta_0 + np.pi * beta * vad2 * dos_d
    delta_ = np.array([delta_0 for k in range(0, len(energy))])
    dE_d_hyb, dos_ads_na, na = namodel(eps_a, delta, delta_,
                                       delta_0, energy)
    dos_ads_na = dos_ads_na/integrate.cumtrapz(dos_ads_na, energy, axis=0)[-1]

    dE_d = dE_d_hyb + 2 * (na + filling) * alpha *  beta * vad2
    dE_na = dE_0 + 3 * dE_d 
    dE_nas.append(dE_na)

print dE_nas
print dE_dfts
np.savetxt('E_NA.txt', dE_nas)

for i in range(len(vars)):
    np.savetxt(vars[i]+'.txt', db.trace(vars[i])[num:])
