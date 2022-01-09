import pandas as pd
import numpy as np

def generate_data_plus(months, months_path, road_path):
    for month in months:
        test_month = [month]
        train_month = months.copy()
        train_month.remove(month)
        
        df_train = pd.concat([pd.read_csv(months_path[month]) for month in train_month])
        df_train.loc[df_train['speed_typea']<0, 'speed_typea'] = 0
        df_train.loc[df_train['speed_typea']>200, 'speed_typea'] = 100
        df_train['gps_timestamp'] = pd.to_datetime(df_train['gps_timestamp'])
        df_train['weekdaytime'] = df_train['gps_timestamp'].dt.weekday * 144 + (df_train['gps_timestamp'].dt.hour * 60 + df_train['gps_timestamp'].dt.minute)//10
        df_train = df_train[['linkid', 'weekdaytime', 'speed_typea']]

        df_train_avg = df_train.groupby(['linkid', 'weekdaytime']).mean().reset_index()

        df_test = pd.concat([pd.read_csv(months_path[month]) for month in test_month])
        df_test.loc[df_test['speed_typea']<0, 'speed_typea'] = 0
        df_test.loc[df_test['speed_typea']>200, 'speed_typea'] = 100
        df_test['gps_timestamp'] = pd.to_datetime(df_test['gps_timestamp'])
        df_test['weekdaytime'] = df_test['gps_timestamp'].dt.weekday * 144 + (df_test['gps_timestamp'].dt.hour * 60 + df_test['gps_timestamp'].dt.minute)//10

        df = pd.merge(df_test, df_train_avg, on=['linkid', 'weekdaytime'], suffixes=(None, '_y'))
        df_capital_link = pd.read_csv(road_path)
        capital_linkid_list = df_capital_link['link_id'].unique()
        timeslices = df_test.gps_timestamp.unique() # must be datetime
        mux = pd.MultiIndex.from_product([capital_linkid_list, timeslices],names=['linkid', 'gps_timestamp'])
        df = df.set_index(['linkid', 'gps_timestamp']).reindex(mux).reset_index()
        df['weekdaytime'] = df['weekdaytime']/df['weekdaytime'].max()
    
        df.to_csv(f'../data/capitaltrafficplus_{month}.csv.gz', index=False)
        print('generate capital traffic plus over', month, df.shape)
        

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