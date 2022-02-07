import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import argparse
from configparser import ConfigParser
import logging
        
parser = argparse.ArgumentParser()
parser.add_argument('--month', type=str, default='202112', help='In gen_dataplus, it must be set to 202112.')
opt = parser.parse_args()
config = ConfigParser()
config.read('params.txt', encoding='UTF-8')
train_month = eval(config[opt.month]['train_month'])
test_month = eval(config[opt.month]['test_month'])
traffic_path = config[opt.month]['traffic_path']
road_path = config['common']['road_path']

# training months are used for calculating the avg speed for all months.

def generate_data_plus(train_month, months, months_path, road_path):
    df_train = pd.concat([pd.read_csv(months_path[month]) for month in train_month])
    df_train.loc[df_train['speed_typea']<0, 'speed_typea'] = 0
    df_train.loc[df_train['speed_typea']>200, 'speed_typea'] = 100
    df_train['gps_timestamp'] = pd.to_datetime(df_train['gps_timestamp'])
    df_train['weekdaytime'] = df_train['gps_timestamp'].dt.weekday * 144 + (df_train['gps_timestamp'].dt.hour * 60 + df_train['gps_timestamp'].dt.minute)//10
    df_train = df_train[['linkid', 'weekdaytime', 'speed_typea']]
    df_train_avg = df_train.groupby(['linkid', 'weekdaytime']).mean().reset_index()
        
    for month in months:
        df_test = pd.read_csv(months_path[month])
        df_test.loc[df_test['speed_typea']<0, 'speed_typea'] = 0
        df_test.loc[df_test['speed_typea']>200, 'speed_typea'] = 100
        df_test['gps_timestamp'] = pd.to_datetime(df_test['gps_timestamp'])
        df_test['weekdaytime'] = df_test['gps_timestamp'].dt.weekday * 144 + (df_test['gps_timestamp'].dt.hour * 60 + df_test['gps_timestamp'].dt.minute)//10

        df = pd.merge(df_test, df_train_avg, on=['linkid', 'weekdaytime'], suffixes=(None, '_y'))
        df_capital_link = pd.read_csv(road_path)
        capital_linkid_list = df_capital_link['link_id'].unique()
        timeslices = df_test.gps_timestamp.unique() # must be datetime
        mux = pd.MultiIndex.from_product([timeslices, capital_linkid_list],names = ['gps_timestamp', 'linkid'])
        df = df.set_index(['gps_timestamp', 'linkid']).reindex(mux).reset_index()
        df['weekdaytime'] = df['weekdaytime']/df['weekdaytime'].max()
    
        df.to_csv(f'../data/capitaltrafficplus_{month}.csv.gz', index=False)
        print('generate capital traffic plus over', month, df.shape)
        
def main():
    if not os.path.exists(config[opt.month]['trafficplus_path']):
        months = train_month+test_month
        months_path = {month:config[month]['traffic_path'] for month in months}
        print('train_month, test_month, months', train_month, test_month, months)
        generate_data_plus(train_month, months, months_path, road_path)

if __name__ == '__main__':
    main()

