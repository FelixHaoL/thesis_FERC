import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import savgol_filter, find_peaks
from hmmlearn import hmm
from scipy.optimize import curve_fit
import sys
sys.path.append(r'C:\Users\hao\Documents\luohao\code')
sys.path.append(r'C:\Users\hao\Documents\luohao\code\HMM')
import HMM_drift as dhmm
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")

from dudkoClassification import *

def triple_eWLC_fit(data, trace_num, ext_name, f_name, parameters, direction, ext_cut, sg_filter):
    ## inner params
    ext_cut = ext_cut
    sg_filter = sg_filter
    ##

    e_ = data[ext_name].dropna()
    
    if direction == 'U':
        cut_index = np.min(np.where(e_>=ext_cut[0]))
        cut_index_ = np.max(np.where(e_<=ext_cut[1]))
        f_ = data[f_name].dropna()[cut_index:cut_index_]
        e_ = data[ext_name].dropna()[cut_index:cut_index_]
    else:
        cut_index = np.max(np.where(e_>=ext_cut[0]))
        cut_index_ = np.min(np.where(e_<=ext_cut[1]))
        f_ = data[f_name].dropna()[cut_index_:cut_index]
        e_ = data[ext_name].dropna()[cut_index_:cut_index]
    f_line = savgol_filter(f_, sg_filter, 2)
    e_line = savgol_filter(e_, sg_filter, 2)
    

    ## inner params
    std_height = 2
    std_distance = 500
    ##

    f_std = std_slider(f_line, len(f_line)/100)
    index = find_peaks(f_std, height=std_height, distance=std_distance)[0]    

    while len(index) < 2:
        std_height -= 0.1
        index = find_peaks(f_std, height=std_height, distance=std_distance)[0]


    ##inner params
    resample = 100 
    ##

    fit_data = []
    if direction == 'U':
        fe_fit1 = np.transpose([e_line[:index[0]:resample], f_line[:index[0]:resample]])
        fe_fit2 = np.transpose([e_line[index[-1]::resample], f_line[index[-1]::resample]])
    else: 
        fe_fit2 = np.transpose([e_line[:index[0]:resample], f_line[:index[0]:resample]])
        fe_fit1 = np.transpose([e_line[index[-1]::resample], f_line[index[-1]::resample]])
    fit_data.append(fe_fit1)
    fit_data.append(fe_fit2)

    ##inner params
    mid_state_locate_rate = 1.7
    ##

    fit = lmfit.minimize(GlobalFit_residual, parameters, args=fit_data, method='least_squares')
    p = np.array(fit.params)[:3]
    lc1 = np.array(fit.params)[-2]
    lc3 = np.array(fit.params)[-1]
    lc2 = lc1 + abs(lc1 - lc3) / mid_state_locate_rate

    ##inner params
    resample1 = 1000
    ##

    fe_lc = pd.DataFrame({'ext': e_[::int(len(e_line)/resample1)],'force':f_[::int(len(e_line)/resample1)]})
    for i in fe_lc.index:
        Lc, _ = curve_fit(lambda f, p3: eWLC_lc(f, *p, p3), [fe_lc.loc[i].values[1]], [fe_lc.loc[i].values[0]])
        fe_lc.loc[i, 'Lc'] = Lc
    for index, row in fe_lc.iterrows():
        distance = abs(np.array([lc1,lc2,lc3]) - row.Lc)
        fe_lc.at[index, 'belonging']='state' +str(np.argwhere(distance==min(distance)).item(0))
    fit_data = []
    for state in ['state0', 'state2']:
        fit_data.append(np.array(fe_lc.loc[fe_lc.belonging == state, ['ext', 'force']]))
    fit = lmfit.minimize(GlobalFit_residual, parameters, args=fit_data, method='least_squares')
    
    p = np.array(fit.params)[:3]
    lc1 = np.array(fit.params)[-2]
    lc3 = np.array(fit.params)[-1]
    lc2 = lc1 + (lc3 - lc1) / 1.7

    fit_p = {'trace':trace_num,
             'Lk':p[0],
             'Ks':p[1], 
             'Ns':p[2], 
             'lc1':lc1, 
             'lc2':lc2, 
             'lc3':lc3, 
             'cut_index':cut_index,
             'cut_index_':cut_index_}

    df_output = pd.DataFrame({'ext': data[ext_name].dropna(),
                              'force': data[f_name].dropna(),
                              'lc': np.zeros(len(data[ext_name].dropna()))})
    if direction == 'U':
        for index, row in df_output.iloc[cut_index:cut_index_].iterrows():
            Lc, _ = curve_fit(lambda f, p3: eWLC_lc(f, *p, p3), [row[1]], [row[0]])
            row[2] = Lc[0]
    else:
        for index, row in df_output.iloc[cut_index_:cut_index].iterrows():
            Lc, _ = curve_fit(lambda f, p3: eWLC_lc(f, *p, p3), [row[1]], [row[0]])
            row[2] = Lc[0]
      

    ##plot
    
    f = np.arange(40, 160, 0.01)
    fig = plt.figure(dpi=100)
    plt.plot(df_output.ext, df_output.force, 'k',label = 'Force-Extension Curve')
    plt.plot(eWLC(f, *p, lc1), f, '--r', label = 'Linker-WLC fit '+r'$I_{G}^{0}$')
    plt.plot(eWLC(f, *p, lc2), f, '--b', label = 'Linker-WLC fit '+r'$I_{G}^{1}$')
    plt.plot(eWLC(f, *p, lc3), f, '--g', label = 'Linker-WLC fit '+r'$I_{G}^{2}$')
    
    
    
    data = df_output.loc[df_output.lc != 0]
    lc2 = lc1 + abs(lc1 - lc3) / 1.6

    ratio = 1 / 10


    data.loc[data.lc < (lc1 + abs(lc2 - lc1) * ratio), 'belonging'] = 'N'
    data.loc[data.lc > (lc3 - abs(lc3 - lc2) * ratio), 'belonging'] = 'U'
    data.loc[(data.lc > (lc2 - abs(lc2 - lc1) * ratio * 2)) & (
              data.lc < (lc2 + abs(lc2 - lc3) * ratio * 2)), 'belonging'] = 'I'
    
    
    ### 2.1
    stateT_index = locate_state_index(data, np.nan)
    if stateT_index == 0:
        print('bad trace')
    else:
        for i in stateT_index:
            if i[0] == data.index[0]:
                if direction == 'U':
                    data.loc[i[0]:i[1] + 1, 'belonging'] = 'N'
                    continue
                if direction == 'R':
                    data.loc[i[0]:i[1] + 1, 'belonging'] = 'U'
                    continue
            if i[1] == data.index[-1]:
                if direction == 'U':
                    data.loc[i[0]:i[1] + 1, 'belonging'] = 'U'
                    continue
                if direction == 'R':
                    data.loc[i[0]:i[1] + 1, 'belonging'] = 'N'
                    continue
            if data.loc[i[0] - 1, 'belonging'] == data.loc[i[1] + 1, 'belonging']:
                data.loc[i[0]:i[1] + 1, 'belonging'] = data.loc[i[1] + 1, 'belonging']
                
    data_0 = df_output.iloc[:min(cut_index,cut_index_)]
    data_2 = df_output.iloc[max(cut_index,cut_index_):]
    
    data_0['belonging'] = np.nan
    data_2['belonging'] = np.nan


    data = pd.concat([data_0, data, data_2])
    #data = data.drop(columns=['ext','lc'])

    return data, fit_p

def force_tile(df_in, sample_rate, fit_p):

    df_in['time'] = df_in.index*sample_rate
    z_mean = np.zeros(2)
    for key in ['N','U']:
        f = df_in.loc[df_in.belonging==key, ['force']].values.reshape(-1)
        t = df_in.loc[df_in.belonging==key, ['time']].values.reshape(-1)
        z = np.polyfit(t, f, 1)
        if key!='I':
            z_mean += z/2
        fit_p.update({'loading_rate'+key:z[0]})
        fit_p.update({'intercept'+key:z[1]})
    fit_p.update({'loading_rateI':z_mean[0]})
    fit_p.update({'interceptI':z_mean[1]})    
    index_range = sorted([fit_p['cut_index'], fit_p['cut_index_']])
    f = df_in['force'][index_range[0]:index_range[1]]
    t = df_in['time'][index_range[0]:index_range[1]]
    df_in['flatten_force'] = f - (z_mean[0]*t + z_mean[1])
    
    return df_in, fit_p

from scipy import interpolate 
from hmmlearn import hmm


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
def drift_hmm(obs, state_num, drift_num, drift_iter, min_covar=1, hmm_tol=0.01):
    transition_pro = 1e-88
    obs = obs.reshape(-1,1)
    drift = np.zeros(len(obs))
    drift_check = []
    GHMM = hmm.GaussianHMM(n_components=state_num, init_params='sc', params='sc',n_iter=10000)
    for i in range(drift_iter):
        obs = obs.reshape(-1,1)
        GHMM._init(obs)
        while not (GHMM.means_[2] == sorted(GHMM.means_)[1]):
            GHMM = hmm.GaussianHMM(n_components=state_num, init_params='sc', params='sc',n_iter=10000)
            GHMM._init(obs)
        GHMM.means_ = [[float(sorted(GHMM.means_)[0])],[float(sorted(GHMM.means_)[2])],[0.5]]
        #GHMM.transmat_ = [[1 - (transition_pro), transition_pro],
        #                  [transition_pro, 1 - (transition_pro)]]
        GHMM.transmat_ = [[1 - (transition_pro+transition_pro), transition_pro, transition_pro],
                          [transition_pro, 1 - (transition_pro+transition_pro), transition_pro],
                          [transition_pro, transition_pro, 1 - (transition_pro+transition_pro)]]
        GHMM.fit(obs)

        states = np.zeros(len(obs))
        key = GHMM.decode(obs)[1]
        states[key==0]=GHMM.means_[0][0]
        states[key==1]=GHMM.means_[1][0]
        states[key==2]=GHMM.means_[2][0]
        
        drift_temp = make_drift(obs, states, drift_num)

        obs = obs.reshape(-1)-drift_temp
        drift+=drift_temp

        drift_check.append(drift_temp)
    return states, drift, drift_check


if if __name__ == "__main__":
    fit_paras = pd.DataFrame()
    direction = 'U'
    for trace in [12]:
        print(trace, end='__')
        ext = data['Ext_smth'+str(molecule)+direction+str(trace)]
        force = data['Force_smth'+str(molecule)+direction+str(trace)]
        df = pd.DataFrame({'ext':ext, 'force':force})
        df_out, fit_p = triple_eWLC_fit(data=df,direction=direction, ext_name='ext', f_name='force', ext_cut=[18.5,25], trace_num=trace, parameters=paras,sg_filter=3001)
        fit_paras = pd.concat([fit_paras, pd.DataFrame(fit_p,index=[trace])])
        df_tile, fit_p = force_tile(df_out, 0.00001/50, fit_p)
        #plt.savefig(r'E:\Eq_data\figures\Dudko_1.eps',dpi=600,format='eps')
        states, drift, model = drift_hmm(np.array(df_tile.flatten_force.dropna().tolist()),3, 8, 3, min_covar=1e-5, hmm_tol=0.01)
        df_tile['drift'] = np.nan
        df_tile['hmm'] = np.nan
        df_tile.loc[df_tile.flatten_force == df_tile.flatten_force, 'hmm'] = states
        df_tile.loc[df_tile.flatten_force == df_tile.flatten_force, 'drift'] = drift
        df_tile['state'] = df_tile['hmm'] + fit_p['loading_rateI']*df_tile['time'] + fit_p['interceptI']

        ##plt.plot(df_tile.time, df_tile.force)
        ##plt.plot(df_tile.time, df_tile.state)
        #fit_paras = pd.concat([fit_paras, pd.DataFrame(fit_p,index=[trace])])
        #
        #df_out = df_tile.loc[:,['flatten_force','hmm','drift']]
        #df_out.columns = ['flatten_force'+str(trace),'hmm'+str(trace),'drift'+str(trace)]
        #df_out.to_csv(r'E:\data\117\hmm_check\M117_hmm'+str(trace)+direction+'.csv',index=None)
        #
        #df_out = df_tile.loc[:,['force','state']]
        #df_out.to_csv(r'E:\data\117\M117_hmm'+str(trace)+direction+'.csv',index=None)
    #fit_paras.to_csv(r'E:\data\117\M117Ufitparas.csv', index=None)
    plt.ylabel('Force (pN)', fontdict=font_label)
    plt.xlabel('Extension (nm)', fontdict=font_label)

    plt.tick_params(axis='y', right=True, which='both', labelsize=lsize)
    plt.tick_params(axis='x', top=True, labelsize=lsize)
    plt.legend(frameon=False, prop=font_legend)
    plt.figtext(0.01, 0.90, 'a',family='Arial', fontsize=20)
    #plt.savefig(r'E:\thesis_figure\Dudko\Dudko_13552.png',dpi=600,format='png',pad_inches=0.01)