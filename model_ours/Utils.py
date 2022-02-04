import pandas as pd
import numpy as np
import jpholiday

def get_data(data_path, N_link, subdata_path, feature_list):
    data = pd.read_csv(data_path)[feature_list].values
    data = data.reshape(-1, N_link, data.shape[-1])
    data[data<0] = 0
    data[data>200.0] = 100.0
    sub_idx = np.loadtxt(subdata_path).astype(int)
    data = data[:, sub_idx, :]
    return data
    
def get_adj(adj_path, subroad_path):
    A = np.load(adj_path)
    if subroad_path is not None:
        sub_idx = np.loadtxt(subroad_path).astype(int)
        A = A[sub_idx, :][:, sub_idx]
    return A

def get_seq_data(data, seq_len):
    seq_data = [data[i:i+seq_len, ...] for i in range(0, data.shape[0]-seq_len+1)]
    return np.array(seq_data)

def getXSYS_single(data_list, his_len, seq_len):
    XS, YS = [], []
    for data in data_list:
        seq_data = get_seq_data(data, seq_len + his_len)
        XS_, YS_ = seq_data[:, :his_len, ...], seq_data[:, -seq_len:-seq_len+1, ...]
        XS.append(XS_)
        YS.append(YS_)
    XS, YS = np.vstack(XS), np.vstack(YS)    
    return XS, YS

def getXSYS(data_list, his_len, seq_len):
    XS, YS = [], []
    for data in data_list:
        seq_data = get_seq_data(data, seq_len + his_len)
        XS_, YS_ = seq_data[:, :his_len, ...], seq_data[:, -seq_len:, ...]
        XS.append(XS_)
        YS.append(YS_)
    XS, YS = np.vstack(XS), np.vstack(YS)
    return XS, YS

def get_onehottime(data_path):
    data = pd.read_csv(data_path)
    data['gps_timestamp'] = pd.to_datetime(data['gps_timestamp'])
    df = pd.DataFrame({'time':data['gps_timestamp'].unique()})  
    df['dayofweek'] = df.time.dt.weekday
    df['hourofday'] = df.time.dt.hour
    df['intervalofhour'] = df.time.dt.minute//10 
    df['isholiday'] = df.apply(lambda x: int(jpholiday.is_holiday(x.time) | (x.dayofweek==5) | (x.dayofweek==6)), axis=1)
    tmp1 = pd.get_dummies(df.dayofweek)
    tmp2 = pd.get_dummies(df.hourofday)
    tmp3 = pd.get_dummies(df.intervalofhour)
    tmp4 = df[['isholiday']]
    df_dummy = pd.concat([tmp1, tmp2, tmp3, tmp4], axis=1)
    return df_dummy.values
