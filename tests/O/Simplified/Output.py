#!/usr/bin/env python

import pymc as pm
import numpy as np
import pickle
from scipy import integrate

#from chemisorption_py import namodel_py
from chemisorption import namodel

List = ['Co', 'Ni', 'Cu', 'Ru',
        'Rh', 'Pd', 'Ag', 'Ir',
        'Pt', 'Au']

Vak2_d0 = [1.34, 1.16, 1, 3.87, 
           3.32, 2.78, 2.26, 4.45,
           3.9, 3.35]

f = [0.8, 0.9, 1, 0.7,
     0.8, 0.9, 1, 0.8,
     0.9, 1]

vars = ['effadse', 'beta', 'delta', 'alpha', 
        'Esp', 'var_1', 'var_2']

db = pm.database.pickle.load('M.pickle')
num = len(db.trace(vars[0])[:])/2

vars_s = {}
for var in vars:
    vars_s[var] = [db.trace(var)[num:].mean(), db.trace(var)[num:].std()]
print vars_s

ergy = np.linspace(-15, 15, 3001)
fermiindex=np.argmax(ergy >= 0)

x_data = np.loadtxt('dos_d.txt')
y1_data = np.loadtxt('dos_ads.txt')
y2_data = np.load('E.npy')

effadse = vars_s['effadse'][0]
beta = vars_s['beta'][0]
delta = vars_s['delta'][0]
alpha = vars_s['alpha'][0]
Esp = vars_s['Esp'][0]
dos_d = np.array(x_data)
E = []


def NA_model(effadse=effadse, beta=beta, delta=delta,
             alpha=alpha, Esp=Esp, value=y1_data):
    error1 = 0
    error2 = 0
    for i in range(len(List)):
        Vak2_d = Vak2_d0[i]*beta
        wdos = np.pi*Vak2_d*dos_d[i]+delta

#        ergy_NA, dos_NA, na = namodel_py(effadse,wdos, delta, ergy)
        ergy_NA, dos_NA, na = namodel(effadse, wdos,
                                      delta, ergy)

        dos_NA = dos_NA/integrate.cumtrapz(dos_NA, ergy, axis=0)[-1]
        error1 += np.sum((dos_NA-value[i])**2)
        BE = Esp + (ergy_NA + 2*(na+f[i])*alpha*Vak2_d)*3
        E.append(BE)
        error2 += (BE-y2_data[i])**2
    return (error1/len(ergy)/len(List))**0.5,(error2/len(List))**0.5


print NA_model(effadse=effadse, beta=beta, delta=delta,
               alpha=alpha, Esp=Esp, value=y1_data)
print E
print y2_data
np.savetxt('E_NA.txt', E)
for i in range(len(vars)):
    np.savetxt(vars[i]+'.txt',db.trace(vars[i])[num:])
