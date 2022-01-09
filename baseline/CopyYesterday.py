import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from configparser import ConfigParser
import time
import sys
import logging
import shutil
import pandas as pd
import numpy as np
from Utils import *
import Metrics

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=6, help='sequence length of values, which should be even nums (2,4,6,12)')
parser.add_argument('--his_len', type=int, default=6, help='sequence length of observed historical values')
parser.add_argument('--month', type=str, default='202111', help='which experiment setting (month) to run')
parser.add_argument('--city', type=str, default='tokyo', help='which experiment setting (city) to run')
parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
opt = parser.parse_args()

config = ConfigParser()
config.read('params.txt', encoding='UTF-8')
train_month = eval(config[opt.month]['train_month'])
test_month = eval(config[opt.month]['test_month'])
traffic_path = config[opt.month]['traffic_path']
subroad_path = config[opt.city]['subroad_path']
adj_path = config['common']['adjdis_path']
N_link = config.getint('common', 'N_link')
feature_list = ['speed_typea']
opt.channelin = len(feature_list)

_, filename = os.path.split(os.path.abspath(sys.argv[0]))
filename = os.path.splitext(filename)[0]
model_name = filename.split('_')[-1]
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'./save/{opt.city}{opt.month}_{model_name}_c{opt.channelin}to{opt.channelout}_{timestring}' 
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
    
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info('experiment_city', opt.city)
logger.info('experiment_month', opt.month)
logger.info('model_name', model_name)
logger.info('channnel_in', opt.channelin)
logger.info('channnel_out', opt.channelout)

def get_seq_data_idx(data_list, seq_len):
    data = np.vstack(data_list)
    seq_data_idx = [np.arange(i, i+seq_len) for i in range(0, data.shape[0]-seq_len+1)]
    return np.array(seq_data_idx)

def getXSYS_idx(data, his_len, seq_len):
    seq_data = get_seq_data_idx(data, seq_len + his_len)
    XS, YS = seq_data[:, :his_len, ...], seq_data[:, seq_len:, ...]
    return XS, YS

def CopyYesterday(data_list, YS_index):
    data = np.vstack(data_list)
    print(data.shape)
    DAYTIMESTEP = 144
    YS, YS_pred = [], []
    for i in range(YS_index.shape[0]):
        index = YS_index[i, 0]
        if index >= DAYTIMESTEP:
            YS_pred.append(data[index-DAYTIMESTEP:index-DAYTIMESTEP+opt.seq_len, :, 0]) # 0 is the traffic speed.
            YS.append(data[index:index+opt.seq_len, :, 0])
        else:
            pass
    YS, YS_pred = np.array(YS), np.array(YS_pred)
    return YS, YS_pred

def testModel(name, mode, data, YS_index):
    logger.info(opt.city, opt.month, 'TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    YS, YS_pred = CopyYesterday(data, YS_index)
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    # np.save(path + f'/{name}_prediction.npy', YS_pred)
    # np.save(path + f'/{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    logger.info('*' * 40)
    f = open(score_path, 'a')
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(opt.seq_len):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, ...], YS_pred[:, i, ...])
        logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()

def main():
    test_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, feature_list) for month in test_month]
    _, testYS_idx = getXSYS_idx(test_data, opt.his_len, opt.seq_len)
    testModel(model_name, 'test', test_data, testYS_idx)
    
if __name__ == '__main__':
    main()
