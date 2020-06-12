import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def locate_state_index(data_df, state):
    if state != state:
        ind = data_df.loc[data_df.state.isna()].index.tolist()
    else:
        ind = data_df.loc[data_df.state==state].index.tolist()
    if len(ind) == 0:
        return 0
    first = np.array(ind)[np.append(True,np.diff(ind)!=1)]
    last = np.array(ind)[np.append(np.diff(ind)!=1,True)]
    result = [value for value in zip(first, last)]
    return result


none_nan_index = np.where((data==data)==True)[0].tolist()
first = np.array(none_nan_index)[np.append(True,np.diff(none_nan_index)!=1)]
last = np.array(none_nan_index)[np.append(np.diff(none_nan_index)!=1 ,True)] + 1
result = [value for value in zip(first, last)]


def init_state(force:list):
    var = force[-1] - force[0]
    if var > 0:
        return 0
    if var < 0:
        return 2
def state_series(force, initial_state):
    index = np.arange(len(force))
    if initial_state == 0:
        z0 = np.polyfit(index[:2],force[:2],1)
        z0_ = z0 + np.array([0, -1])
        val0_ = np.polyval(z0_, index)
        
        z2 = np.polyfit(index[-2:],force[-2:],1)
        z2_ = z2 + np.array([0, +1])
        val2_ = np.polyval(z2_, index)
        state = np.array([np.nan]*len(force)) 
        
        state[np.array(force>val0_)] = initial_state
        state[np.array(force<val2_)] = abs(2 - initial_state)
        state[np.array(state != state)] = 1
        return state
    if initial_state == 2:
        z2 = np.polyfit(index[:2],force[:2],1)
        z2_ = z2 + np.array([0, +1])
        val2_ = np.polyval(z2_, index)
        
        z0 = np.polyfit(index[-2:],force[-2:],1)
        z0_ = z0 + np.array([0, -1])
        val0_ = np.polyval(z0_, index)
        
        state = np.array([np.nan]*len(force)) 
        state[np.array(force>val0_)] = abs(2 - initial_state)
        state[np.array(force<val2_)] = initial_state
        state[np.array(state != state)] = 1
        return state
def transition_info(force, state, sample_rate, index_range):
    force_ = []
    index_ = []
    direction_ = []
    life_time_ = []
    state_ = []
    path_ = []
    count = 0
    for i in range(len(state)-1):
        current_ = state[i]
        next_ = state[i+1]
        if current_ != next_:
            life_time_.append(count*sample_rate)
            force_.append(force[i])
            index_.append(i+index_range[0])
            state_.append(int(current_))
            path_.append(str(int(current_))+str(int(next_)))
            count = 0
        count += 1
    neibor_ = [str(int(state_[i]))+str(int(state_[i+2])) for i in range(len(state_)-2)]
    neibor_.insert(0, np.nan)
    if state[0] == 0:
        neibor_.append(str(state_[-2])+str(2))
    if state[0] == 2:
        neibor_.append(str(state_[-2])+str(0))
    rup_forces = pd.DataFrame({'force':force_, 'index':index_, 'path':path_, 'neibor':neibor_, 'life_time':life_time_,'state':state_})
    return rup_forces

def t_resolution_fielter(rup_forces, cutofftime, state, force, sample_rate, index_range):
    count = 0
    row = rup_forces.sort_values('life_time').iloc[count,:]
    while row.life_time < cutofftime:
        
        neibor = row.neibor
        #print(neibor)
        if neibor in ['02', '20']:
            state_new = int(neibor[1])
        if neibor in ['00', '11', '22']:
            state_new = int(neibor[1])
        if neibor in ['10', '01', '12', '21']:
            state_new = 1
        ind = [rup_forces.loc[row.name-1, 'index']+1, row['index']]
        state[ind[0]-index_range[0]: ind[1]-index_range[0]+1] = state_new
        rup_forces = transition_info(force, state, sample_rate, index_range)
        row = rup_forces.sort_values('life_time').iloc[count,:]
        if (row.name==0) or (row.name == rup_forces.index[-1]):
            count +=1
            row = rup_forces.sort_values('life_time').iloc[count,:]
    return rup_forces, state

def t_resolution_fielter_middle(rup_forces, cutofftime, state, force, sample_rate, index_range):
    count = 0
    row = rup_forces.sort_values('life_time').iloc[count,:]
    while row.life_time < cutofftime:
        
        neibor = row.neibor
        #print(neibor)
        if neibor in ['02', '20']:
            state_new = int(neibor[1])
        if neibor in ['00', '11', '22','10', '01', '12', '21', np.nan]:
            count +=1
            row = rup_forces.sort_values('life_time').iloc[count,:]
            continue
        ind = [rup_forces.loc[row.name-1, 'index']+1, row['index']]
        state[ind[0]-index_range[0]: ind[1]-index_range[0]+1] = state_new
        rup_forces = transition_info(force, state, sample_rate, index_range)
        row = rup_forces.sort_values('life_time').iloc[count,:]
        if (row.name==0) or (row.name == rup_forces.index[-1]):
            count +=1
            row = rup_forces.sort_values('life_time').iloc[count,:]
    return rup_forces, state

def continuous_tansition(trnsition_series):
    p = list(trnsition_series)
    temp = ''
    for i in range(0,len(p)):
        temp += p[i]
    continuous = all([temp[i]==temp[i+1] for i in range(1,len(temp)-1,2)])
    boundary = ((temp[0]=='0') & (temp[-1]=='2') | (temp[0]=='2') & (temp[-1]=='0'))
    return continuous & boundary

def extend_filed_force_state(force, state, initiral_state):
    index = np.arange(len(force))
    new_force = np.array([np.nan]*len(force))
    for s in np.unique(state):
        z0 = np.polyfit(index[state==s],force[state==s],1)
        val0_ = np.polyval(z0, index[state==s])
        new_force[state==s] = s
        new_force[new_force==s] = val0_
    if initial_state == 0:
        z0 = np.polyfit(index[state==0],new_force[state==0],1)
        low_force_index = (50 - z0[1]) / z0[0]
        val0 = np.polyval(z0, np.arange(low_force_index,0))
        
        z2 = np.polyfit(index[state==2],new_force[state==2],1)
        high_force_index = (150 - z2[1]) / z2[0]
        val2 = np.polyval(z2, np.arange(len(new_force), high_force_index))
        
        extended_force = np.concatenate([val0, new_force, val2])
        extended_state = np.concatenate([[0]*len(val0), state, [2]*len(val2)])
        return extended_force, extended_state, new_force
    if initial_state == 2:
        z0 = np.polyfit(index[state==2],new_force[state==2],1)
        high_force_index = (150 - z0[1]) / z0[0]
        val0 = np.polyval(z0, np.arange(high_force_index,0))
        
        z2 = np.polyfit(index[state==0],new_force[state==0],1)
        low_force_index = (50 - z2[1]) / z2[0]
        val2 = np.polyval(z2, np.arange(len(new_force), low_force_index))
        
        extended_force = np.concatenate([val0, new_force, val2])
        extended_state = np.concatenate([[2]*len(val0), state, [0]*len(val2)])
        return extended_force, extended_state, new_force
    
def force_segement_range_state(filed_extended_force, extended_state):
    force_segments = []
    force_head = filed_extended_force[0]
    count = 0
    for i in range(len(extended_state)-1):
        current_ = extended_state[i]
        next_ = extended_state[i+1]
        if current_ != next_:
            force_tail = filed_extended_force[i]
            force_segments.append(([force_head, force_tail], current_))
            force_head = filed_extended_force[i+1]
            last_index = i+1
        count +=1
    force_head = filed_extended_force[last_index]
    force_tail = filed_extended_force[-1]
    force_segments.append(([force_head, force_tail], current_))
    return force_segments      

def transition_force_density(df, bins):
    #forces_for_Nf = []
    bins_dic = {}
    df_density = pd.DataFrame()
    trans = np.unique(df.path)
    for t in trans:
        hist_counts, hist_bins = np.histogram(df.loc[df.path==str(t), 'force'], bins=bins)
        force_bin = hist_bins[:-1] + np.diff(hist_bins)/2 
        rup_density = hist_counts/np.diff(hist_bins)
        df_density[t+'_hist_counts'] = hist_counts
        df_density[t+'_force'] = force_bin
        df_density[t+'_rup_density'] = rup_density
        bins_dic.update({str(t): hist_bins})
        #forces_for_Nf.extend(force_bin)
    return df_density, bins_dic

def trajectories_under_forces(df_p, force_ranges_labels, force_bins_size):
    path = []
    for m in map(lambda x: x.split('_'), df_p.columns.tolist()):
        path.extend(m[:1])
    path = np.unique(path)
    for p in path:
        force_bins = df_p[p+'_force']
        size = force_bins_size
        state_c = pd.DataFrame({'force':force_bins,
                                p[0]:np.zeros_like(force_bins),p[0]+'label':np.zeros_like(force_bins)})
        for frl in force_ranges_labels:
            for force_range, label in np.array(frl)[:,:2]:
                if label != int(p[0]):
                    continue
                
                state_c.loc[((state_c['force']-size)<max(force_range)) & ((state_c['force']+size) > min(force_range)), p[0]+'label'] = 1
            state_c[p[0]] = state_c[p[0]] + state_c[p[0]+'label']
            state_c.loc[:,p[0]+'label'] = 0

        df_p[p+'_Nf'] = state_c[p[0]]
    return df_p

def state_time_under_forces(df_p, force_ranges_labels, sample_rate, force_range_density):
    path = []
    for m in map(lambda x: x.split('_'), df_p.columns.tolist()):
        path.extend(m[:1])
    path = np.unique(path)
    for p in path:
        force_bins = df_p[p+'_force']
        size = np.array([np.diff(force_bins)[0]/2]*len(force_bins))
        state_c = pd.DataFrame({'force':force_bins,
                                p[0]:np.zeros_like(force_bins),p[0]+'label':np.zeros_like(force_bins)})
        for frl in force_ranges_labels:
            for force_range, label in np.array(frl):
                if label != int(p[0]):
                    continue
                state_c.loc[((state_c.force-size)>min(force_range)) & 
                            ((state_c.force+size)<max(force_range)), p[0]+'label'] = int((2*size[0])/force_range_density)*sample_rate
                
                down_force = state_c.loc[((state_c.force-size)<min(force_range)) &
                                         ((state_c.force+size)>min(force_range))].force.values
                if down_force.size > 0:
                    down_ = abs(down_force[0]+size[0]-min(force_range))
                    state_c.loc[((state_c.force-size)<min(force_range)) & ((state_c.force+size)>min(force_range)),p[0]+'label'] = int(down_ / force_range_density)*sample_rate
                
                up_force = state_c.loc[((state_c.force-size)>min(force_range)) & 
                                      ((state_c.force+size)>max(force_range)) & 
                                      ((state_c.force-size)<max(force_range))].force.values
                if up_force.size > 0:
                    up_ = abs(up_force[0]-size[0]-max(force_range))
                    state_c.loc[((state_c.force-size)>min(force_range)) & ((state_c.force+size)>max(force_range)) & 
                                ((state_c.force-size)<max(force_range)), p[0]+'label'] = int(up_ / force_range_density)*sample_rate
                state_c[p[0]] = state_c[p[0]] + state_c[p[0]+'label']
                state_c.loc[:,p[0]+'label'] = 0

        df_p[p+'_Nf_time'] = state_c[p[0]]
    return df_p





if if __name__ == "__main__":

    sample_rate = 2e-7
    out_df = pd.DataFrame()
    loading_rate = []
    cutoff_t = sample_rate*500
    Nf_Franges_state_density = []
    force_range_densities = []
    print( 'total {}'.format(len(result)))
    for trace in range(len(result)):

        if trace==639:
            continue
        #if trace in [29,72,117]:
        #    continue
        print(trace+1, end='_')
        index_range = result[trace]
        force = data[index_range[0]: index_range[1]]
        loading_rate.append(abs(np.polyfit(np.arange(2)*sample_rate,force[:2], 1)[0]))
        initial_state = init_state(force)
        state = state_series(force,initial_state)    
        rup_forces = transition_info(force, state, sample_rate, index_range)
        rup_forces, state = t_resolution_fielter_middle(rup_forces, cutoff_t, state, force, sample_rate, index_range)
        filed_extended_force, extended_state, filed_force = extend_filed_force_state(force, state, initial_state)
        Nf_Franges_state_density.append(force_segement_range_state(filed_extended_force, extended_state))
        force_range_densities.append(abs(filed_force[0] - filed_force[-1]) / len(filed_force))
        #print(abs(filed_extended_force[0] - filed_extended_force[-1]) / len(filed_extended_force))
        out_df = pd.concat([out_df, rup_forces], axis=0)

    out_df = out_df.reset_index(drop=True)
    loading_rate = np.mean(loading_rate)
    df_P, bins= transition_force_density(out_df, np.arange(60,140, 4))
    force_range_density = np.mean(force_range_densities)
    df_P = trajectories_under_forces(df_P, Nf_Franges_state_density, 0.001)
    #df_P = state_time_under_forces(df_P, Nf_Franges_state_density, sample_rate,force_range_density)
    for p in path:
    rate = df_P[p+'_rate']
    rate_c = df_P[p+'_rate_corr0']
    tau = df_P[p+'_Nf_time']
    bin_size = np.diff(df_P[p+'_force'])[0]
    #plt.plot(df_P[p+'_force'],tau*200, '^', label='t_raw X 200')
    #plt.plot(df_P[p+'_force'],(rate_c/rate)*(rate_c - rate), '^', label='delta')
    #N_old = df_P[p+'_rup_density'] *bin_size
    #N_new = (rate_c/rate)*(rate_c - rate)*tau
    #
    #plt.plot(df_P[p+'_force'],N_old, '.', label='old')
    #F = out_df.loc[out_df.path==p, 'force']
    #plt.hist(F, bins=18,edgecolor='k', alpha=0.5)
    #plt.plot(df_P[p+'_force'], N_old+N_new, '*', label = 'new')
    #plt.title(p)

    df_P[p+'_rup_density_corr'] = df_P[p+'_rup_density']+ ((rate_c/rate)*(rate_c - rate)*tau).round(0)/ bin_size
    rate = loading_rate * df_P[p+'_rup_density_corr'] / df_P[p+'_Nf']
    df_P[p+'_rate_corr'] = rate
    
    #log_sd = (1/(np.diff(df_P[p+'_force'])[0]*df_P[p+'_rup_density_corr'])+(1/df_P[p+'_Nf']))**0.5
    #df_P[p+'_rate_SDupper'] = np.exp(np.log(df_P[p+'_rate_corr'])+log_sd)
    #df_P[p+'_rate_SDlower'] = np.exp(np.log(df_P[p+'_rate_corr'])-log_sd)
    
    
    #erro_bar = np.array([(abs(df_P[p+'_rate_corr']-df_P[p+'_rate_SDlower'])).tolist(),(abs(df_P[p+'_rate_corr']-df_P[p+'_rate_SDupper'])).tolist()])
    #plt.errorbar(x=df_P[p+'_force'], y=df_P[p+'_rate_corr'], yerr=erro_bar,fmt='.-', label=p)
