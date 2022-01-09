import os
import argparse
from configparser import ConfigParser
import time
import sys
import logging
import shutil
import pandas as pd
import numpy as np
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
road_path = config['common']['road_path']
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

def get_seq_data(data, seq_len):
    seq_data = [data[i:i+seq_len, ...] for i in range(0, data.shape[0]-seq_len+1)]
    return np.array(seq_data)

def getXSYS(data, his_len, seq_len):
    seq_data = get_seq_data(data, seq_len + his_len)
    XS, YS = seq_data[:, :his_len, ...], seq_data[:, -seq_len:, ...]
    return XS, YS

def MonthlyAverage():
    df_train = pd.concat([pd.read_csv(config[month]['traffic_path']) for month in train_month])
    df_train.loc[df_train['speed_typea']<0, 'speed_typea'] = 0
    df_train.loc[df_train['speed_typea']>200, 'speed_typea'] = 100
    df_train['gps_timestamp'] = pd.to_datetime(df_train['gps_timestamp'])
    df_train['weekdaytime'] = df_train['gps_timestamp'].dt.weekday * 144 + (df_train['gps_timestamp'].dt.hour * 60 + df_train['gps_timestamp'].dt.minute)//10
    df_train = df_train[['linkid', 'weekdaytime', 'speed_typea']]
    df_train_avg = df_train.groupby(['linkid', 'weekdaytime']).mean().reset_index()
    
    df_test = pd.concat([pd.read_csv(config[month]['traffic_path']) for month in test_month])
    df_test.loc[df_test['speed_typea']<0, 'speed_typea'] = 0
    df_test.loc[df_test['speed_typea']>200, 'speed_typea'] = 100
    df_test['gps_timestamp'] = pd.to_datetime(df_test['gps_timestamp'])
    df_test['weekdaytime'] = df_test['gps_timestamp'].dt.weekday * 144 + (df_test['gps_timestamp'].dt.hour * 60 + df_test['gps_timestamp'].dt.minute)//10
    df_test = df_test[['linkid', 'gps_timestamp', 'speed_typea', 'weekdaytime']]

    df = pd.merge(df_test, df_train_avg, on=['linkid', 'weekdaytime'])
    df_capital_link = pd.read_csv(road_path)
    capital_linkid_list = df_capital_link['link_id'].unique()
    timeslices = df_test.gps_timestamp.unique() # must be datetime
    mux = pd.MultiIndex.from_product([capital_linkid_list, timeslices],names=['linkid', 'gps_timestamp'])
    df = df.set_index(['linkid', 'gps_timestamp']).reindex(mux).reset_index()
    
    test_data = df['speed_typea_x'].values.reshape(-1, N_link, 1)
    pred_data = df['speed_typea_y'].values.reshape(-1, N_link, 1)
    sub_idx = np.loadtxt(subroad_path).astype(int)
    test_data = test_data[:, sub_idx, :]
    pred_data = pred_data[:, sub_idx, :]
    _, YS = getXSYS(test_data, opt.his_len, opt.seq_len) 
    _, YS_pred = getXSYS(pred_data, opt.his_len, opt.seq_len) 
    return YS, YS_pred

def testModel(name, mode):
    logger.info(opt.city, opt.month, 'TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    YS, YS_pred = MonthlyAverage()
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    # np.save(path + f'/{name}_prediction.npy', YS_pred)
    # np.save(path + f'/{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    logger.info('*' * 40)
    f = open(score_path, 'a')
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(opt.seq_len):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()

def main():
    testModel(model_name, 'test')
    
if __name__ == '__main__':
    main()
