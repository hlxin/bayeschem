#!/usr/bin/env python

import pickle
import pymc as pm
import numpy as np
from scipy import integrate

from chemisorption import namodel

List = ['Au', 'Ag', 'Cu',
        'Pt', 'Pd', 'Ni',
        'Ir', 'Rh',
        'Co', 'Ru']

vars = ['adse_1', 'beta_1', 'adse_2', 'beta_2',
        'delta_1', 'delta_2', 'alpha', 
        'Esp', 'var_1', 'var_2']
db = pm.database.pickle.load('M.pickle')
num = len(db.trace(vars[0])[:])/2

vars_s = {}
for var in vars:
    vars_s[var] = [db.trace(var)[num:].mean(), db.trace(var)[num:].std()]
print vars_s

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
x_data = np.loadtxt('dos_d.txt')
y1_data = np.loadtxt('dos_ads_z.txt')
y2_data = np.loadtxt('dos_ads_x.txt')
y3_data = np.load('E.npy')

# priors                                                                        
adse_1 = vars_s['adse_1'][0]
beta_1 = vars_s['beta_1'][0]
adse_2 = vars_s['adse_2'][0]
beta_2 = vars_s['beta_2'][0]
delta_1 = vars_s['delta_1'][0]
delta_2 = vars_s['delta_2'][0]
alpha = vars_s['alpha'][0]
Esp = vars_s['Esp'][0]
dos_d = np.array(x_data)
E = []


def NA_model(adse_1=adse_1, beta_1=beta_1, delta_1=delta_1,
             adse_2=adse_2, beta_2=beta_2, delta_2=delta_2,
             alpha=alpha, Esp=Esp, value=y1_data):
    error1 = 0
    error2 = 0
    error3 = 0
    for i in range(len(List)):
        Vak2_d_1 = Vak2_d0[i] * beta_1
        wdos_1 = np.pi*Vak2_d_1*dos_d[i]+delta_1
        ergy_NA_1, dos_NA_1, na_1 = namodel(adse_1, wdos_1, 
                                           delta_1, ergy)
        dos_NA_1 = dos_NA_1/integrate.cumtrapz(dos_NA_1, ergy, axis=0)[-1]
        
        Vak2_d_2 = Vak2_d0[i] * beta_2
        wdos_2 = np.pi * Vak2_d_2 * dos_d[i] + delta_2
        ergy_NA_2, dos_NA_2, na_2 = namodel(adse_2, wdos_2,
                                           delta_2, ergy)
        dos_NA_2 = dos_NA_2/integrate.cumtrapz(dos_NA_2, ergy, axis=0)[-1]
        
        error1 += np.sum((dos_NA_1-value[i])**2)
        error2 += np.sum((dos_NA_2-y2_data[i])**2)
        BE = Esp + (ergy_NA_1 + 2*(na_1+f[i])*alpha*Vak2_d_1)+(ergy_NA_2 + 2*(na_2+f[i])*alpha*Vak2_d_2)*2
        E.append(BE)
        error3 += (BE-y3_data[i])**2
    return (error1/len(ergy)/len(List))**0.5,(error2/len(ergy)/len(List))**0.5,(error3/len(List))**0.5


print NA_model(adse_1=adse_1, beta_1=beta_1, delta_1=delta_1,
               adse_2=adse_2, beta_2=beta_2, delta_2=delta_2,
               alpha=alpha, Esp=Esp, value=y1_data)
print E
print y3_data
np.savetxt('E_NA.txt', E)
for i in range(len(vars)):
    np.savetxt(vars[i]+'.txt',db.trace(vars[i])[num:])
