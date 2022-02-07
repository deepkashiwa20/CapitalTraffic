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
import Metrics
from Utils import *
import filecmp

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=6, help='sequence length of values, which should be even nums (2,4,6,12)')
parser.add_argument('--his_len', type=int, default=6, help='sequence length of observed historical values')
parser.add_argument('--month', type=str, default='202112', help='which experiment setting (month) to run')
parser.add_argument('--city', type=str, default='tokyo', help='which experiment setting (city) to run')
parser.add_argument('--model', type=str, default='MMGCRN', help='which model to use')
opt = parser.parse_args()

config = ConfigParser()
config.read('params.txt', encoding='UTF-8')
train_month = eval(config[opt.month]['train_month'])
test_month = eval(config[opt.month]['test_month'])
traffic_path = config[opt.month]['traffic_path']
subroad_path = config[opt.city]['subroad_path']
road_path = config['common']['road_path']
adj_path = config['common']['adjdis_path'] 
# adj_path = config['common']['adj01_path']
num_variable = len(np.loadtxt(subroad_path).astype(int))
N_link = config.getint('common', 'N_link')

print('experiment_city', opt.city)
print('experiment_month', opt.month)
print('model_name', opt.model)
##############################################################

# data columns of capitaltrafficplus_202112.csv.gz
# gps_timestamp, linkid, speed_typea, accident_flag, real_accident_flag, weekdaytime, speed_typea_y

def main():
    test_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, ['speed_typea']) for month in test_month]
    _, testYS = getXSYS(test_data, opt.his_len, opt.seq_len)
    
    test_linktime = [get_data_tmp(config[month]['traffic_path'], N_link, subroad_path, ['gps_timestamp', 'linkid']) for month in test_month]
    _, testYS_linktime = getXSYS(test_linktime, opt.his_len, opt.seq_len)
    
    print('TEST YS.shape, YS_linktime.shape', testYS.shape, testYS_linktime.shape)
    # (4453, 6, 1843, 1) 
    # (4453, 6, 1843, 2)
    print(testYS_linktime[0,0,0,0], testYS_linktime[0,0,0,1])
    print(testYS_linktime[0,0,-1,0], testYS_linktime[0,0,-1,1])
    
    
if __name__ == '__main__':
    main()

