import matplotlib.pyplot as plt
from scipy import interpolate 
from hmmlearn import hmm
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, minimize

def make_drift(obs, states, interval_num):
    residual = obs.reshape(-1) - states.reshape(-1)
    nodes = [0]
    interval = int(len(residual)/interval_num)
    
    for seg in range(0,len(residual), interval):
        nodes.append(np.mean(residual[seg:seg+interval]))
    nodes.append(0)
        
    index = np.arange(0,len(residual), interval)+interval/2
    index = np.insert(index, 0,0)
    index = np.append(index, len(residual))
    f_quadratic = interpolate.interp1d(index, nodes, kind='quadratic')
    drift = f_quadratic(np.arange(len(residual)))
    
    return drift
def drift_hmm(obs, state_num, drift_num, drift_iter, trans_p=1e-150, min_covar=1, hmm_tol=0.01):
    transition_pro = trans_p
    obs = obs.reshape(-1,1)
    drift = np.zeros(len(obs))
    drift_check = []
    GHMM = hmm.GaussianHMM(n_components=state_num, init_params='scm', params='smc',n_iter=10000)
    for i in range(drift_iter):
        obs = obs.reshape(-1,1)
        GHMM._init(obs)
        #while not (GHMM.means_[2] == sorted(GHMM.means_)[1]):
        #    GHMM = hmm.GaussianHMM(n_components=state_num, init_params='sc', params='sc',n_iter=10000)
        #    GHMM._init(obs)
        #GHMM.means_ = [[float(sorted(GHMM.means_)[0])],[float(sorted(GHMM.means_)[2])],[0.5]]
        #GHMM.transmat_ = [[1 - (transition_pro), transition_pro],
        #                  [transition_pro, 1 - (transition_pro)]]
        if state_num == 2:
            GHMM.transmat_ = [[1 - (transition_pro), transition_pro],
                              [transition_pro, 1 - (transition_pro)]]
        if state_num == 3:
            GHMM.transmat_ = [[1 - (transition_pro+transition_pro), transition_pro, transition_pro],
                              [transition_pro, 1 - (transition_pro+transition_pro), transition_pro],
                              [transition_pro, transition_pro, 1 - (transition_pro+transition_pro)]]
        GHMM.fit(obs)

        states = np.zeros(len(obs))
        key = GHMM.decode(obs)[1]
        if state_num == 2:
            states[key==0]=GHMM.means_[0][0]
            states[key==1]=GHMM.means_[1][0]
        if state_num == 3:
            states[key==0]=GHMM.means_[0][0]
            states[key==1]=GHMM.means_[1][0]
            states[key==2]=GHMM.means_[2][0]
        
        drift_temp = make_drift(obs, states, drift_num)

        obs = obs.reshape(-1)-drift_temp
        drift+=drift_temp

        drift_check.append(drift_temp)
    return states, drift, drift_check

def sort_dic_by_key(dic):
    new_dic = {}
    dic = sorted(dic.items(),key=lambda x:x[0])
    for i in range(len(dic)):
        new_dic.update({dic[i][0]:dic[i][1]})
    return new_dic
def get_life_time_from_states(states, sample_rate):
    forces = np.unique(states)
    
    force_lifetime ={}
    for j in np.unique(states):
        force_lifetime.update({j:[]})
    
    count = 0
    last = states[0]
    for s in states:
        if s == last:
            count +=1
        if s != last:
            temp_list = force_lifetime[last]
            temp_list.append(count*sample_rate)
            force_lifetime.update({last:temp_list})
            count = 1
        last = s
    temp_list = force_lifetime[last]
    temp_list.append(count*sample_rate)
    force_lifetime.update({last:temp_list})
    return force_lifetime
