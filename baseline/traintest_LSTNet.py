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
from LSTNet import *
from Utils import *

def refineXSYS(XS, YS):
    XS, YS = XS[:, :, :, :opt.channelin], YS[:, :, :, :opt.channelout]
    XS = XS.reshape(XS.shape[0], XS.shape[1], -1)
    YS = YS.reshape(YS.shape[0], YS.shape[1], -1)
    return XS, YS

def refineYS_ex(YS):
    YS = YS[:, :, :, 1:]
    YS = YS.reshape(YS.shape[0], YS.shape[1], -1)
    return YS

def getModel():
    if opt.his_len >= 168:
        model = LSTNet(num_variable=num_variable,
                     in_dim = opt.channelin,
                     out_dim = opt.channelout,
                     window=opt.his_len,
                     hidRNN=64,
                     hidCNN=64,
                     CNN_kernel=3,
                     skip=3,
                     highway_window=24,
                     dropout=0, 
                     output_fun='tanh').to(device)
    else:
        model = LSTNet(num_variable=num_variable,
                     in_dim = opt.channelin,
                     out_dim = opt.channelout,
                     window=opt.his_len,
                     hidRNN=64,
                     hidCNN=64,
                     CNN_kernel=3,
                     skip=3,
                     highway_window=3,
                     dropout=0.2, 
                     output_fun=None).to(device)
    summary(model, (opt.his_len, num_variable * opt.channelin), device=device)
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

def predictModel_multi(model, data_iter):
    YS_pred_multi = []
    model.eval()
    with torch.no_grad():
        if opt.channelin > 1:
            for x, y, ex in data_iter:
                XS_pred_multi_batch, YS_pred_multi_batch = [x], []
                for i in range(opt.seq_len):
                    tmp_torch = torch.cat(XS_pred_multi_batch, axis=1)[:, i:, :]
                    yhat = model(tmp_torch)
                    YS_pred_multi_batch.append(yhat)
                    XS_pred_multi_batch.append(torch.cat((yhat, ex[:, i:i+1, :]), axis=2))
                YS_pred_multi_batch = torch.cat(YS_pred_multi_batch, axis=1).cpu().numpy()
                YS_pred_multi.append(YS_pred_multi_batch)
            YS_pred_multi = np.vstack(YS_pred_multi)
        else:
            for x, y in data_iter:
                XS_pred_multi_batch, YS_pred_multi_batch = [x], []
                for i in range(opt.seq_len):
                    tmp_torch = torch.cat(XS_pred_multi_batch, axis=1)[:, i:, :]
                    yhat = model(tmp_torch)
                    YS_pred_multi_batch.append(yhat)
                    XS_pred_multi_batch.append(yhat)
                YS_pred_multi_batch = torch.cat(YS_pred_multi_batch, axis=1).cpu().numpy()
                YS_pred_multi.append(YS_pred_multi_batch)
            YS_pred_multi = np.vstack(YS_pred_multi)
    return YS_pred_multi
        
def trainModel(name, mode, XS, YS):
    logger.info('Model Training Started ...', time.ctime())
    logger.info('opt.his_len, opt.seq_len', opt.his_len, opt.seq_len)
    model = getModel()
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - opt.val_ratio))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, opt.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, opt.batch_size, shuffle=True)
    if opt.loss == 'MSE':
        criterion = nn.MSELoss()
    if opt.loss == 'MAE':
        criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    min_val_loss = np.inf
    wait = 0
    for epoch in range(opt.epoch):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        # scheduler.step()
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
        logger.info("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, ", validation loss:", val_loss)
        with open(epochlog_path, 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, opt.batch_size, shuffle=False))
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    logger.info('*' * 40)
    logger.info("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    logger.info("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    logger.info('Model Training Ended ...', time.ctime())

    
def testModel(name, mode, XS, YS, YS_multi, YS_ex_multi):
    logger.info('Model Testing Started ...', time.ctime())
    logger.info('opt.his_len, opt.seq_len', opt.his_len, opt.seq_len)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, opt.batch_size, shuffle=False)    
    model = getModel()
    model.load_state_dict(torch.load(modelpt_path))
    if opt.loss == 'MSE': criterion = nn.MSELoss()
    if opt.loss == 'MAE': criterion = nn.L1Loss()
    torch_score = evaluateModel(model, criterion, test_iter)
    
    if opt.channelin > 1: 
        YS_ex_torch = torch.Tensor(YS_ex_multi).to(device)
        test_data1 = torch.utils.data.TensorDataset(XS_torch, YS_torch, YS_ex_torch)
        test_iter1 = torch.utils.data.DataLoader(test_data1, opt.batch_size, shuffle=False)
        YS_pred_multi = predictModel_multi(model, test_iter1)
    else:
        YS_pred_multi = predictModel_multi(model, test_iter)
            
    logger.info('YS_multi.shape, YS_pred_multi.shape,', YS_multi.shape, YS_pred_multi.shape)
    YS_multi, YS_pred_multi = np.squeeze(YS_multi), np.squeeze(YS_pred_multi)
    for i in range(YS_multi.shape[1]):
        YS_multi[:, i, :] = scaler.inverse_transform(YS_multi[:, i, :])
        YS_pred_multi[:, i, :] = scaler.inverse_transform(YS_pred_multi[:, i, :])
    logger.info('YS_multi.shape, YS_pred_multi.shape,', YS_multi.shape, YS_pred_multi.shape)
    # np.save(path + f'/{name}_prediction.npy', YS_pred_multi)
    # np.save(path + f'/{name}_groundtruth.npy', YS_multi)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS_multi, YS_pred_multi)
    logger.info('*' * 40)
    logger.info("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f = open(score_path, 'a')
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(opt.seq_len):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS_multi[:, i, :], YS_pred_multi[:, i, :])
        logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    logger.info('Model Testing Ended ...', time.ctime())
        
#########################################################################################    
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

device = torch.device("cuda:{}".format(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)

scaler = StandardScaler()

def main():
    train_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, feature_list) for month in train_month]
    test_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, feature_list) for month in test_month]

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
    trainXS, trainYS = getXSYS_single(train_data, opt.his_len, opt.seq_len)
    trainXS, trainYS = refineXSYS(trainXS, trainYS) # new added
    logger.info('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(model_name, 'train', trainXS, trainYS)
    
    logger.info(opt.city, opt.month, 'testing started', time.ctime())
    testXS, testYS = getXSYS_single(test_data, opt.his_len, opt.seq_len)
    testXS, testYS = refineXSYS(testXS, testYS)
    testXS_multi, testYS_multi = getXSYS(test_data, opt.his_len, opt.seq_len)
    testXS_multi, testYS_y_multi = refineXSYS(testXS_multi, testYS_multi)
    if opt.channelin > 1:
        testYS_ex_multi = refineYS_ex(testYS_multi)
        logger.info('TEST XS.shape, YS.shape, XS_multi.shape, YS_y_multi.shape, YS_ex_multi.shape', testXS.shape, testYS.shape, testXS_multi.shape, testYS_y_multi.shape, testYS_ex_multi.shape)
        testModel(model_name, 'test', testXS, testYS, testYS_y_multi, testYS_ex_multi)
    else:
        logger.info('TEST XS.shape, YS.shape, XS_multi.shape, YS_y_multi.shape', testXS.shape, testYS.shape, testXS_multi.shape, testYS_y_multi.shape)
        testModel(model_name, 'test', testXS, testYS, testYS_y_multi, None)
    

if __name__ == '__main__':
    main()

