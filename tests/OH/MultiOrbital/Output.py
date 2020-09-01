#!/usr/bin/env python

import pymc as pm
import numpy as np
import pickle
from scipy import integrate
from chemisorption import namodel
#from chemisorption_py import namodel_py

vars = ['adse_1', 'beta_1', 'adse_2', 
        'beta_2', 'adse_3', 'beta_3', 
        'delta_1', 'delta_2', 'delta_3',
        'alpha',
        'Esp', 'var_1', 'var_2']

db = pm.database.pickle.load('M.pickle')
num = len(db.trace(vars[0])[:])/2

vars_s = {}
for var in vars:
    vars_s[var] = [db.trace(var)[num:].mean(), db.trace(var)[num:].std()]
print vars_s

Vak2_d0 = np.loadtxt('Vad.txt')
f = np.loadtxt('f.txt')
ergy = np.linspace(-15, 15, 3001)
fermiindex = np.argmax(ergy >= 0)
dos_d = np.loadtxt('dos_d.txt')
y1_data = np.loadtxt('dos_ads_1.txt')
y2_data = np.loadtxt('dos_ads_2.txt')
y3_data = np.loadtxt('dos_ads_3.txt')
y4_data = np.loadtxt('BE.txt')

# priors                                                                        
adse_1 = vars_s['adse_1'][0]
adse_2 = vars_s['adse_2'][0]
adse_3 = vars_s['adse_3'][0]

beta_1 = vars_s['beta_1'][0]
beta_2 = vars_s['beta_2'][0]
beta_3 = vars_s['beta_3'][0]

delta_1 = vars_s['delta_1'][0]
delta_2 = vars_s['delta_2'][0]
delta_3 = vars_s['delta_3'][0]

alpha = vars_s['alpha'][0]

Esp = vars_s['Esp'][0]

E = []


def NA_model(adse_1=adse_1,beta_1=beta_1,delta_1=delta_1,
             adse_2=adse_2,beta_2=beta_2,delta_2=delta_2,
             adse_3=adse_3,beta_3=beta_3,delta_3=delta_3,
             alpha=alpha, 
             Esp=Esp,value=y1_data):

    error1 = 0
    error2 = 0
    error3 = 0
    error4 = 0
    for i in range(len(f)):
        Vak2_d_1 = Vak2_d0[i]*beta_1
        wdos_1 = np.pi*Vak2_d_1*dos_d[i]+delta_1
        ergy_NA_1, dos_NA_1, na_1 = namodel(adse_1, wdos_1,
                                            delta_1, ergy)
        dos_NA_1 = dos_NA_1/integrate.cumtrapz(dos_NA_1, ergy, axis=0)[-1]

        Vak2_d_2 = Vak2_d0[i]*beta_2
        wdos_2 = np.pi*Vak2_d_2*dos_d[i]+delta_2
        ergy_NA_2, dos_NA_2, na_2 = namodel(adse_2, wdos_2,
                                            delta_2, ergy)
        dos_NA_2 = dos_NA_2/integrate.cumtrapz(dos_NA_2, ergy, axis=0)[-1]

        Vak2_d_3 = Vak2_d0[i]*beta_3
        wdos_3 = np.pi*Vak2_d_3*dos_d[i]+delta_3
        ergy_NA_3, dos_NA_3, na_3 = namodel(adse_3, wdos_3,
                                            delta_3, ergy)
        dos_NA_3 = dos_NA_3/integrate.cumtrapz(dos_NA_3, ergy, axis=0)[-1]
        print f[i], 2*ergy_NA_1, 2*ergy_NA_2, ergy_NA_3, na_3

        error1 += np.sum((dos_NA_1-value[i])**2)
        error2 += np.sum((dos_NA_2-y2_data[i])**2)
        error3 += np.sum((dos_NA_3-y3_data[i])**2)

        BE = Esp + (ergy_NA_1 + 2*(na_1+f[i])*alpha*Vak2_d_1) + 2 * (ergy_NA_2 + 2*(na_2+f[i])*alpha*Vak2_d_2) + (ergy_NA_3 + 2*(na_3+f[i])*alpha*Vak2_d_3)

        E.append(BE)
        error4 += (BE-y4_data[i])**2
    return (error1/len(ergy)/len(f))**0.5,(error2/len(ergy)/len(f))**0.5,(error3/len(ergy)/len(f))**0.5,(error4/len(f))**0.5


print NA_model(adse_1=adse_1, beta_1=beta_1, delta_1=delta_1,
               adse_2=adse_2, beta_2=beta_2, delta_2=delta_2,
               adse_3=adse_3, beta_3=beta_3, delta_3=delta_3,
               alpha=alpha, 
               Esp=Esp, value=y1_data)
print E
print y4_data
np.savetxt('E_NA.txt', E)
for i in range(len(vars)):
    np.savetxt(vars[i]+'.txt', db.trace(vars[i])[num:])
