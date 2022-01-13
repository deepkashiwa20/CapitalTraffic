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
from GMAN import *
from Utils import *

def refineXSYS(XS, YS):
    XS, YS = XS[:, :, :, :opt.channelin], YS[:, :, :, :opt.channelout]
    return XS, YS


#If use whole graph embedding and sub_graph.
# def getModel():
    # def loadSE(file_path, sub_index):
    #     with open(file_path, mode='r') as f:
    #         lines = f.readlines()
    #         temp = lines[0].split(' ')
    #         num_vertex, dims = int(temp[0]), int(temp[1])
    #         SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
    #         for line in lines[1:]:
    #             temp = line.split(' ')
    #             index = int(temp[0])
    #             SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    #     return SE[sub_index, :]
    # SE_path = f'../data/{opt.city}_road_embedding.txt'
    # subroad_path = f'../data/{opt.city}_road_idx.csv'
    # sub_idx = np.loadtxt(subroad_path).astype(int)
    # SE = loadSE(SE_path, sub_idx).to(device=device)
    # logger.info('SE.shape', SE.size())
    # model = GMAN(SE, SE_dim=64, TE_dim=32, timestep_in=opt.his_len, channel_in=opt.channelin, channel_out=opt.channelout, device=device).to(device)
    # summary(model, [(opt.his_len, num_variable), (opt.his_len+opt.seq_len, 32)], device=device)
    # return model

def loadSE(file_path):
    with open(file_path, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    return SE

def getModel():
    SE_path = f'../data/{opt.city}_road_embedding.txt'
    SE = loadSE(SE_path).to(device)
    logger.info('SE.shape', SE.size())
    model = GMAN(SE, SE_dim=32, TE_dim=32, timestep_in=opt.his_len, channel_in=opt.channelin, channel_out=opt.channelout, 
                 device=device, statt_layers=1, att_heads=4, att_dims=8, bn_decay=0.1)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)
    # summary(model, [(opt.his_len, num_variable, opt.channelin), (opt.his_len+opt.seq_len, 32)], device=device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, te, y in data_iter:
            y_pred = model(x, te)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, te, y in data_iter:
            YS_pred_batch = model(x, te)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS, TE):
    logger.info('Model Training Started ...', time.ctime())
    logger.info('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    model = getModel()
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    TE_torch = torch.Tensor(TE).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - opt.val_ratio))
    logger.info('XS_torch.shape:  ', XS_torch.shape)
    logger.info('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, opt.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, opt.batch_size, shuffle=True)
    if opt.loss == 'MSE': criterion = nn.MSELoss()
    if opt.loss == 'MAE': criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    min_val_loss = np.inf
    wait = 0   
    for epoch in range(opt.epoch):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, te, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x, te)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        else:
            wait += 1
            if wait == opt.patience:
                logger.info('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        logger.info("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(epochlog_path, 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, opt.batch_size, shuffle=False))
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    YS, YS_pred = YS.reshape(-1, opt.seq_len, YS.shape[-1]), YS_pred.reshape(-1, opt.seq_len, YS_pred.shape[-1])
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    logger.info('*' * 40)
    logger.info("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    logger.info("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    logger.info('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS, TE):
    logger.info('Model Testing Started ...', time.ctime())
    logger.info('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    TE_torch = torch.Tensor(TE).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, opt.batch_size, shuffle=False)
    model = getModel()
    model.load_state_dict(torch.load(modelpt_path))
    if opt.loss == 'MSE': criterion = nn.MSELoss()
    if opt.loss == 'MAE': criterion = nn.L1Loss()
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    YS, YS_pred = YS.reshape(-1, opt.seq_len, YS.shape[-1]), YS_pred.reshape(-1, opt.seq_len, YS_pred.shape[-1])
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    # np.save(path + f'/{name}_prediction.npy', YS_pred)
    # np.save(path + f'/{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    logger.info('*' * 40)
    logger.info("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    with open(score_path, 'a') as f:
        f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
        for i in range(opt.seq_len):
            MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
            logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    logger.info('Model Testing Ended ...', time.ctime())
        
#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default='MAE', help="MAE, MSE, SELF")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--patience", type=float, default=10, help="patience used for early stop")
parser.add_argument('--val_ratio', type=float, default=0.25, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--seq_len', type=int, default=6, help='sequence length of values, which should be even nums (2,4,6,12)')
parser.add_argument('--his_len', type=int, default=6, help='sequence length of observed historical values')
parser.add_argument('--gpu', type=int, default=3, help='which gpu to use')
parser.add_argument('--month', type=str, default='202111', help='which experiment setting (month) to run')
parser.add_argument('--city', type=str, default='tokyo', help='which experiment setting (city) to run')
parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
parser.add_argument('--incident', type=bool, default=False, help='whether to use incident flag')
parser.add_argument('--accident', type=bool, default=False, help='whether to use accident flag')
parser.add_argument('--time', type=bool, default=False, help='whether to use float time embedding')
parser.add_argument('--history', type=bool, default=False, help='whether to use historical data')
opt = parser.parse_args()

config = ConfigParser()
config.read('params.txt', encoding='UTF-8')
train_month = eval(config[opt.month]['train_month'])
test_month = eval(config[opt.month]['test_month'])
traffic_path = config[opt.month]['traffic_path']
subroad_path = config[opt.city]['subroad_path']
road_path = config['common']['road_path']
adj_path = config['common']['adjdis_path'] # adj_path = config['common']['adj01_path']
num_variable = len(np.loadtxt(subroad_path).astype(int))
N_link = config.getint('common', 'N_link')
feature_list = ['speed_typea']
if opt.incident: feature_list.append('accident_flag')
if opt.accident: feature_list.append('real_accident_flag')
if opt.time: feature_list.append('weekdaytime')
if opt.history: feature_list.append('speed_typea_y')
opt.channelin = len(feature_list)
# feature_list = ['speed_typea', 'accident_flag', 'real_accident_flag', 'weekdaytime', 'speed_typea_y']

_, filename = os.path.split(os.path.abspath(sys.argv[0]))
filename = os.path.splitext(filename)[0]
model_name = filename.split('_')[-1]
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'./save/{opt.city}{opt.month}_{model_name}_c{opt.channelin}to{opt.channelout}_{timestring}' 
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
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
logger.info('feature_incident', opt.incident)
logger.info('feature_accident', opt.accident)
logger.info('feature_time', opt.time)
logger.info('feature_history', opt.history)
#####################################################################################################

# Multi-GPU "cuda"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)

scaler = StandardScaler()

def main():
    train_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, feature_list) for month in train_month]
    test_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, feature_list) for month in test_month]
    train_onehottime = [get_onehottime(config[month]['traffic_path']) for month in train_month]
    test_onehottime= [get_onehottime(config[month]['traffic_path']) for month in test_month]
    
    speed_data = []
    for data in train_data:
        speed_data.append(data[:,:,0])
    for data in test_data:
        speed_data.append(data[:,:,0])
    speed_data = np.vstack(speed_data)    
    scaler.fit(speed_data)
    
    for data in train_data:
        logger.info('train_data', data.shape)
        data[:,:,0] = scaler.transform(data[:,:,0])
    for data in test_data:
        logger.info('test_data', data.shape)
        data[:,:,0] = scaler.transform(data[:,:,0])
    
    logger.info(opt.city, opt.month, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(train_data, opt.his_len, opt.seq_len)
    trainXS, trainYS = refineXSYS(trainXS, trainYS)  
    trainXS_TE, trainYS_TE = getXSYS(train_onehottime, opt.his_len, opt.seq_len)
    trainTE = np.concatenate([trainXS_TE, trainYS_TE], axis=1)
    logger.info('TRAIN XS.shape YS,shape, TE.shape', trainXS.shape, trainYS.shape, trainTE.shape)
    trainModel(model_name, 'train', trainXS, trainYS, trainTE)
    
    logger.info(opt.city, opt.month, 'testing started', time.ctime())
    testXS, testYS =getXSYS(test_data, opt.his_len, opt.seq_len)
    testXS, testYS = refineXSYS(testXS, testYS)
    testXS_TE, testYS_TE = getXSYS(test_onehottime, opt.his_len, opt.seq_len)
    testTE = np.concatenate([testXS_TE, testYS_TE], axis=1)
    logger.info('TEST XS.shape, YS.shape, TE.shape', testXS.shape, testYS.shape, testTE.shape)
    testModel(model_name, 'test', testXS, testYS, testTE)

    
if __name__ == '__main__':
    main()

