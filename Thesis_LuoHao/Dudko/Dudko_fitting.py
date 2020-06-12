import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import warnings

import pandas as pd
import lmfit
warnings.filterwarnings('ignore')

from scipy.optimize import curve_fit

def dudok_fit_forward(force:list, k0:float, deltaG_double:float, deltaX_double:float):
    kbt = 1/4.0636
    deltaG_double = deltaG_double / kbt
    
    k = 24.96
    v = 2/3
    beta = 1 + (1-v)*k*deltaX_double**2/(2*deltaG_double)
    
    phi = 1 + v*k*deltaX_double/(2*deltaG_double) - v*force*deltaX_double*beta/deltaG_double
    rate = np.exp(k0)*pow(phi, (1/v-1)) * np.exp(kbt*deltaG_double*(1-pow(phi, 1/v)))
    return rate

def dudok_fit_backward(force:list, k0:float, deltaG_double:float, deltaX_double:float):
    kbt = 1/4.0636
    deltaG_double = deltaG_double / kbt
    
    k = 24.96
    v = 2/3
    beta = 1 + (1-v)*k*deltaX_double**2/(2*deltaG_double)
    
    phi = 1 + v*k*deltaX_double/(2*deltaG_double) + v*force*deltaX_double*beta/deltaG_double
    rate = np.exp(k0)*pow(phi, (1/v-1)) * np.exp(kbt*deltaG_double*(1-pow(phi, 1/v)))
    return rate