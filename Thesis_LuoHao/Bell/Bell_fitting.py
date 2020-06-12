from scipy.optimize import curve_fit, minimize
def bell(f,k0,x):
    kbt = 4.06
    k = np.exp(k0)*np.exp(f*x/kbt)
    return k
def bell_r(f,k0,x):
    kbt = 4.06
    k = np.exp(k0)*np.exp(-f*x/kbt)
    return k

def bell_(p, f, kf):
    kbt = 4.06
    k = np.exp(p[0])*np.exp(f*p[1]/kbt)
    return np.sum((np.log(k)-np.log(kf))**2)

def bell_r_(p, f, kf):
    kbt = 4.06
    k = np.exp(p[0])*np.exp(-f*p[1]/kbt)
    return np.sum((np.log(k)-np.log(kf))**2)

if __name__ == "__main__":
    popt= minimize(bell_r_,(12,0.21),args=(np.array(force_fit), np.array(rate_fit)), bounds=((None,None), (0, None)),method='L-BFGS-B')