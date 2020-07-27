#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python setup.py build_ext --inplace
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/opt/apps/Anaconda/2.3.0/envs/fenicsproject/include/python2.7 -I/opt/apps/Anaconda/2.3.0/envs/fenicsproject/lib/python2.7/site-packages/numpy/core/include -o chemisorption.so chemisorption.c

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

setup(cmdclass = {'build_ext': build_ext},
      ext_modules = [ Extension("chemisorption",
                                sources=["chemisorption.pyx","hilbert.c","integrate.c"],
                                extra_compile_args=["-std=c99"],
                                library_dirs=['/opt/apps/gcc6_1/mvapich22_2/fftw/3.3.4/lib/'],
                                include_dirs=[np.get_include(),
                                              '/opt/apps/gcc6_1/mvapich22_2/fftw/3.3.4/include/'],
                                libraries=["m","fftw3"])
                      ])
                                                                
