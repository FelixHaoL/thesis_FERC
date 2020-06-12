#eWLC parameters
Lplanar = 0.358
Lhelical = 0.28
kT = 4.0636
deltaG = 12.1908
K = 10000
Lp = 0.4



import scipy.optimize as opt
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
from scipy.optimize import curve_fit
import lmfit

from scipy.signal import savgol_filter, find_peaks
import cmath
import numpy as np
def line(e,a,b):
    return a+e*b
#eWLC model
def coth(x):
    return np.cosh(x) / np.sinh(x)
def ewlc_lc(f, p0, p1, p2, p3):  # p0=Lk p1=Ks p2=Ns p3=Lc
    extension1 = p2 * (Lplanar / (np.exp(-deltaG / kT) + 1) + Lhelical / (np.exp(deltaG / kT) + 1)) * (
                coth(p0 * f / kT) - kT / (p0 * f)) + p2 * f / p1
    Q = 1 + f / K
    P = f / K + Lp * f / kT + 1 / 4
    l = -2 * Q - P
    u = (-1 / 9) * (Lp * f / kT - 3 / 4) ** 2
    v = (-1 / 27) * (Lp * f / kT - 3 / 4) ** 3 + (1 / 8) + 0.0000000000001
    if (v ** 2 + u ** 3) < 0:
        theta = np.arccos(np.sqrt(-1 * v ** 2 / u ** 3))
        if v < 0:
            result = extension1 + (2 * np.sqrt(-u) * np.cos(theta / 3 + 2 * np.pi / 3) - l / 3) * p3
        else:
            result = extension1 + (-2 * np.sqrt(-u) * np.cos(theta / 3 - 6 * np.pi / 3) - l / 3) * p3
    else:
        A = cmath.exp(
            (1 / 3) * cmath.log(-v + np.sqrt(1 / 64 - 1 / 4 * complex(np.power((Lp * f / kT - 3 / 4) / 3, 3), 0))))
        B = cmath.exp(
            (1 / 3) * cmath.log(-v - np.sqrt(1 / 64 - 1 / 4 * complex(np.power((Lp * f / kT - 3 / 4) / 3, 3), 0))))
        E = -np.abs((A + B)) - l / 3
        result = extension1 + E * p3
    return result

def ewlc(inp,p0,p1, p2, p3):  # p0=Lk p1=Ks p2=Ns p3=Lc
    output_E = []
    for f in inp:

        extension1 = p2 * (Lplanar / (np.exp(-deltaG / kT) + 1) + Lhelical / (np.exp(deltaG / kT) + 1)) * (
                    coth(p0 * f / kT) - kT / (p0 * f)) + p2 * f / p1

        Q = 1 + f / K
        P = f / K + Lp * f / kT + 1 / 4

        l = -2 * Q - P
        u = (-1 / 9) * (Lp * f / kT - 3 / 4) ** 2
        v = (-1 / 27) * (Lp * f / kT - 3 / 4) ** 3 + (1 / 8) + 0.0000000000001

        if (v ** 2 + u ** 3) < 0:
            theta = np.arccos(np.sqrt(-1 * v ** 2 / u ** 3))
            if v < 0:
                result = extension1 + (2 * np.sqrt(-u) * np.cos(theta / 3 + 2 * np.pi / 3) - l / 3) * p3
            else:
                result = extension1 + (-2 * np.sqrt(-u) * np.cos(theta / 3 - 6 * np.pi / 3) - l / 3) * p3
        else:
            A = cmath.exp(
                (1 / 3) * cmath.log(-v + np.sqrt(1 / 64 - 1 / 4 * complex(np.power((Lp * f / kT - 3 / 4) / 3, 3), 0))))
            B = cmath.exp(
                (1 / 3) * cmath.log(-v - np.sqrt(1 / 64 - 1 / 4 * complex(np.power((Lp * f / kT - 3 / 4) / 3, 3), 0))))
            E = -np.abs((A + B)) - l / 3
            result = extension1 + E * p3
        output_E.append(result)
    return output_E

def eWLC_Lc(f,p0, p1, p2, p3):  # p0=Lk p1=Ks p2=Ns p3=Lc
    extension1 = p2 * (Lplanar / (np.exp(-deltaG / kT) + 1) + Lhelical / (np.exp(deltaG / kT) + 1)) * (
                coth(p0 * f / kT) - kT / (p0 * f)) + p2 * f / p1

    Q = 1 + f / K
    P = f / K + Lp * f / kT + 1 / 4

    l = -2 * Q - P
    u = (-1 / 9) * (Lp * f / kT - 3 / 4) ** 2
    v = (-1 / 27) * (Lp * f / kT - 3 / 4) ** 3 + (1 / 8) + 0.0000000000001

    if (v ** 2 + u ** 3) < 0:
        theta = np.arccos(np.sqrt(-1 * v ** 2 / u ** 3))
        if v < 0:
            result = extension1 + (2 * np.sqrt(-u) * np.cos(theta / 3 + 2 * np.pi / 3) - l / 3) * p3
        else:
            result = extension1 + (-2 * np.sqrt(-u) * np.cos(theta / 3 - 6 * np.pi / 3) - l / 3) * p3
    else:
        A = cmath.exp(
            (1 / 3) * cmath.log(-v + np.sqrt(1 / 64 - 1 / 4 * complex(np.power((Lp * f / kT - 3 / 4) / 3, 3), 0))))
        B = cmath.exp(
            (1 / 3) * cmath.log(-v - np.sqrt(1 / 64 - 1 / 4 * complex(np.power((Lp * f / kT - 3 / 4) / 3, 3), 0))))
        E = -np.abs((A + B)) - l / 3
        result = extension1 + E * p3
    return result

def fe_convert_to_lc_wlc(force, extension, *p, initial_guess):
    Lc, _ = curve_fit(lambda e, lc: WLC(e, lc, *p), ydata=force, xdata=extension, p0=initial_guess)
    return Lc[0]
def fe_convert_to_lc_ewlc(force, extension, *p, initial_guess):
    Lc, _ = curve_fit(lambda f, lc: eWLC_Lc(f, *p, lc), xdata=force, ydata=extension, p0=initial_guess)
    return Lc[0]


#global fit tools: residual:  supose you have n sets of data(states) and m data points,
#you need to asure the array input have shape as (n,m,2)
def global_residual(FitParameters, *Array):
    for i in range(len(Array)):
        res = ewlc(Array[i][:, 1], FitParameters['Lk'+str(i)], FitParameters['Ks'+str(i)], FitParameters['Ns'+str(i)],
                   FitParameters['Lc'+str(i)]) - Array[i][:, 0]
        if i == 0:
            residual = res
        else:
            residual = np.concatenate((residual,res))
    return residual
# Names 参数名字n； num 几组参数 ； ini_guesses 各参数初始拟合值； bounds 2 X n
def global_parameters(Names, num, ini_guesses, bounds):
    for n in range(num):
        if n == 0:
            for i in range(len(Names)):
                paras.add(Names[i]+str(n), ini_guesses[i], min = bounds[0][i], max = bounds[1][i])
        else:
            for o in range(len(Names)):
                paras.add(Names[o]+str(n), expr = Names[o]+str(0))
paras = lmfit.Parameters()
Lp = 0.4
paras = lmfit.Parameters()
global_parameters(['Lk', 'Ks', 'Ns'], 3, [0.7, 150000, 40], [[0, 0, 1], [1, 200000, 100]])
paras.add('Lc0', 9, vary=True, min=0, max=12.24)
paras.add('Lc1', 12.24, vary=True, min=0)
paras.add('Lc2', 12.24, vary=True, min=0)
