#!/usr/bin/env python

import numpy as np
import pymc as pm
from pylab import *
import corner

# Plotting Parameters Setting
rcParams['figure.figsize'] = 2*1.67323, 1.9*1.67323
rcParams['ps.useafm'] = True
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

rcParams['pdf.fonttype'] = 42
matplotlib.rc('xtick.major', size=6)
matplotlib.rc('xtick.minor', size=3)
matplotlib.rc('ytick.major', size=6)
matplotlib.rc('ytick.minor', size=3)
matplotlib.rc('lines', markeredgewidth=0.5*2)
matplotlib.rc('font', size=7*2.0)

vars = ['dE_0', 'eps_a',
        'delta_0', 'alpha', 
        'beta']

db = pm.database.pickle.load('M.pickle')
ndim = len(vars)
nsamples = len(db.trace(vars[0])[:])/2

data = db.trace(vars[0])[-nsamples::5]
for var in vars[1:]:
    data = np.vstack([data,db.trace(var)[-nsamples::5]])

data = np.transpose(data)
print data.shape
np.savetxt('CornerData.txt',data)

data = np.loadtxt('CornerData.txt')
# Plot it.
figure = corner.corner(data, bins=20, labels=vars, quantiles=[0.16, 0.5, 0.84], 
                       show_titles=True, title_kwargs={"fontsize": 12})

figure.savefig("corner.pdf")

