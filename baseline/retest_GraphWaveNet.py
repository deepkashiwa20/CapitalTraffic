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
from GraphWaveNet import *
from Utils import *
import filecmp

def refineXSYS(XS, YS):
    XS, YS = XS[:, :, :, :opt.channelin], YS[:, :, :, :opt.channelout]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'\n In total: {param_count} trainable parameters. \n')
    return

def getModel(mode):
    # dist_mx = get_adj(adj_path, subroad_path)
    # A = norm_dist_mx(dist_mx)
    # adj_mx = [sym_adj(A)]
    # supports = [torch.tensor(i).to(device) for i in adj_mx]
    # print('supports', len(supports))
    # model = gwnet(device, num_nodes=num_variable, in_dim=opt.channelin, out_dim=opt.seq_len, supports=supports).to(device)
    model = gwnet(device, num_nodes=num_variable, in_dim=opt.channelin, out_dim=opt.seq_len).to(device)
    if mode == 'train':
        summary(model, (opt.channelin, num_variable, opt.his_len), device=device)
        print_params(model)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred
       
def testModel(name, mode, XS, YS, Mask=None):
    def testScore(YS, YS_pred, message):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
        print(message)
        print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
        print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
        print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
        with open(score_path, 'a') as f:
            f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
            for i in range(opt.seq_len):
                MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[..., i], YS_pred[..., i])
                print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
                f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        return None
    
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, opt.batch_size, shuffle=False)
    model = getModel(mode)
    model.load_state_dict(torch.load(modelpt_path))
    if opt.loss == 'MSE': criterion = nn.MSELoss()
    if opt.loss == 'MAE': criterion = nn.L1Loss()
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    YS, YS_pred = YS.reshape(-1, opt.seq_len, YS.shape[-1]), YS_pred.reshape(-1, opt.seq_len, YS_pred.shape[-1])
    YS, YS_pred = YS.transpose(0, 2, 1), YS_pred.transpose(0, 2, 1)
    # np.save(path + f'/{name}_prediction.npy', YS_pred)
    # np.save(path + f'/{name}_groundtruth.npy', YS)
    # np.save(path + f'/{name}_Mask_t1.npy', Mask)
    testScore(YS, YS_pred, '********* Evaluation on the whole testing dataset *********')
    testScore(YS[Mask], YS_pred[Mask], '********* Evaluation on the selected testing dataset when incident happen at t+1 *********')
    print('Model Testing Ended ...', time.ctime())
    print('Two Score Files are Same', filecmp.cmp(old_score_path, score_path))
    if filecmp.cmp(old_score_path, score_path):
        print('Because the files are same, I return the reproducible model to the main().')
        return model
    else:
        print('Because the files are not same, please kindly check the _scores_retest.txt file.')
        return None
    
parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default='MAE', help="MAE, MSE, SELF")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--patience", type=float, default=10, help="patience used for early stop")
parser.add_argument('--val_ratio', type=float, default=0.25, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--seq_len', type=int, default=6, help='sequence length of values, which should be even nums (2,4,6,12)')
parser.add_argument('--his_len', type=int, default=6, help='sequence length of observed historical values')
parser.add_argument('--gpu', type=int, default=3, help='which gpu to use')
parser.add_argument('--month', type=str, default='202112', help='which experiment setting (month) to run')
parser.add_argument('--city', type=str, default='tokyo', help='which experiment setting (city) to run')
parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
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

# all we need to do is to specify this path.
path = f'../save/tokyo202112_GraphWaveNet_c2to1_20220208044941_time'

keywords = path.split('_')
model_name = keywords[1]
timestring = keywords[3]
xcov_name = keywords[4] if len(keywords) > 4 else ''
old_score_path = f'{path}/{model_name}_{timestring}_scores.txt'
score_path = f'{path}/{model_name}_{timestring}_scores_retest.txt'
if os.path.exists(score_path): os.remove(score_path)
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if xcov_name == 'time': opt.time=True
if xcov_name == 'history': opt.history=True 
feature_list = ['speed_typea']
if opt.time: feature_list.append('weekdaytime')
if opt.history: feature_list.append('speed_typea_y')
opt.channelin = len(feature_list)

print('experiment_city', opt.city)
print('experiment_month', opt.month)
print('model_name', model_name)
print('channnel_in', opt.channelin)
print('channnel_out', opt.channelout)
print('feature_time', opt.time)
print('feature_history', opt.history)

device = torch.device("cuda:{}".format(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)

scaler = StandardScaler()

def main():
    train_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, feature_list) for month in train_month]
    test_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, feature_list) for month in test_month]
    test_flag = [get_data(config[month]['traffic_path'], N_link, subroad_path, ['accident_flag']) for month in test_month]
    
    speed_data = []
    for data in train_data:
        speed_data.append(data[:,:,0])
    for data in test_data:
        speed_data.append(data[:,:,0])
    speed_data = np.vstack(speed_data)    
    scaler.fit(speed_data)
    
    for data in train_data:
        print('train_data', data.shape)
        data[:,:,0] = scaler.transform(data[:,:,0])
    for data in test_data:
        print('test_data', data.shape)
        data[:,:,0] = scaler.transform(data[:,:,0])
           
    print(opt.city, opt.month, 'testing started', time.ctime())
    testXS, testYS = getXSYS(test_data, opt.his_len, opt.seq_len)
    testXS, testYS = refineXSYS(testXS, testYS)
    _, testYSFlag = getXSYS(test_flag, opt.his_len, opt.seq_len)
    testYMask = testYSFlag[:, 0, :, 0] > 0 # (B, N) incident happen at the first prediction timeslot, t+1.
    print('TEST XS.shape, YS.shape, YMask.shape', testXS.shape, testYS.shape, testYMask.shape)
    model = testModel(model_name, 'test', testXS, testYS, testYMask)
    
    #########################
    # I return the model here
    #########################
    
    
if __name__ == '__main__':
    main()

