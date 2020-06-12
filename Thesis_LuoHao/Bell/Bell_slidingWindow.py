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

if if __name__ == "__main__":
    force_lifetime = get_life_time_from_states(states, 2e-7)
    force_lifetime = sort_dic_by_key(force_lifetime)