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
from MMGCRN import MMGCRN
from Utils import *

def refineXSYS(XS, YS):
    assert opt.time or opt.history, 'it should have one covariate time or history'
    XCov, YCov = XS[..., -1:], YS[..., -1:]
    if opt.channelin == 1 and opt.incident:
        XS, YS = XS[:, :, :, :opt.channelin+1], YS[:, :, :, :opt.channelout]
    else:
        XS, YS = XS[:, :, :, :opt.channelin], YS[:, :, :, :opt.channelout]
    return XS, YS, XCov, YCov

def print_params(model):
    # print trainable params
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f' \n In total: {param_count} trainable parameters. \n')
    return

def getModel():
    model = MMGCRN(num_nodes=num_variable, input_dim=opt.channelin, output_dim=opt.channelout, horizon=opt.seq_len, 
                        rnn_units=opt.hiddenunits, num_layers=opt.num_layers, mem_num=opt.mem_num, mem_dim=opt.mem_dim, 
                        memory_type=opt.memory, meta_type=opt.meta, decoder_type=opt.decoder, go_type=opt.go).to(device)
    tmp_channelin = opt.channelin if opt.channelin > 1 else opt.channelin+int(opt.incident)
    logger.info('tmp_channelin: ', tmp_channelin) 
    summary(model, [(opt.his_len, num_variable, tmp_channelin), (opt.seq_len, num_variable, opt.channelout)], device=device)   
    print_params(model)
        
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

def evaluateModel(model, data_iter, ycov_flag):
    if opt.loss == 'MSE': criterion = nn.MSELoss()
    if opt.loss == 'MAE': 
        criterion = nn.L1Loss()
        # compact_loss = nn.MSELoss()
        separate_loss = nn.TripletMarginLoss(margin=1.0)
        
    model.eval()
    loss_sum, n, YS_pred = 0.0, 0, []
    loss_sum1, loss_sum2 = 0.0, 0.0
    with torch.no_grad():
        if ycov_flag:
            for x, y, y_cov in data_iter:
                y_pred, h_att, query, pos, neg = model(x, y_cov)
                loss1 = criterion(y_pred, y)
                loss2 = separate_loss(query, pos.detach(), neg.detach())
                loss = loss1 + opt.lamb * loss2
                loss_sum += loss.item() * y.shape[0]
                loss_sum1 += loss1.item() * y.shape[0]
                loss_sum2 += loss2.item() * y.shape[0]
                n += y.shape[0]
                YS_pred.append(y_pred.cpu().numpy())     
        else:
            for x, y in data_iter:
                y_pred, h_att, query, pos, neg = model(x)
                loss1 = criterion(y_pred, y)
                loss2 = separate_loss(query, pos.detach(), neg.detach())
                loss = loss1 + opt.lamb * loss2
                loss_sum += loss.item() * y.shape[0]
                loss_sum1 += loss1.item() * y.shape[0]
                loss_sum2 += loss2.item() * y.shape[0]
                n += y.shape[0]
                YS_pred.append(y_pred.cpu().numpy())
    loss = loss_sum / n
    loss1 = loss_sum1 / n
    loss2 = loss_sum2 / n    
    YS_pred = np.vstack(YS_pred)
    return loss, loss1, loss2, YS_pred

def trainModel(name, mode, XS, YS, YCov):
    logger.info('Model Training Started ...', time.ctime())
    logger.info('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    model = getModel()
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    logger.info('XS_torch.shape:  ', XS_torch.shape)
    logger.info('YS_torch.shape:  ', YS_torch.shape)
    if YCov is not None:
        YCov_torch = torch.Tensor(YCov).to(device)
        logger.info('YCov_torch.shape:  ', YCov_torch.shape)
        trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, YCov_torch)
    else:    
        trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - opt.val_ratio))

    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, opt.batch_size, shuffle=False) # drop_last=True
    val_iter = torch.utils.data.DataLoader(val_data, opt.batch_size, shuffle=False) # drop_last=True
    trainval_iter = torch.utils.data.DataLoader(trainval_data, opt.batch_size, shuffle=False) # drop_last=True
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.loss == 'MSE': criterion = nn.MSELoss()
    if opt.loss == 'MAE': 
        criterion = nn.L1Loss()
        # compact_loss = nn.MSELoss()
        separate_loss = nn.TripletMarginLoss(margin=1.0)
    min_val_loss = np.inf
    wait = 0   
    for epoch in range(opt.epoch):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        loss_sum1, loss_sum2 = 0.0, 0.0
        model.train()
        if YCov is not None:
            for x, y, ycov in train_iter:
                optimizer.zero_grad()
                y_pred, h_att, query, pos, neg = model(x, ycov)
                loss1 = criterion(y_pred, y)
                loss2 = separate_loss(query, pos.detach(), neg.detach())
                loss = loss1 + opt.lamb * loss2
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * y.shape[0]
                loss_sum1 += loss1.item() * y.shape[0]
                loss_sum2 += loss2.item() * y.shape[0]
                n += y.shape[0]
        else:
            for x, y in train_iter:
                optimizer.zero_grad()
                y_pred, h_att, query, pos, neg = model(x)
                loss1 = criterion(y_pred, y)
                loss2 = separate_loss(query, pos.detach(), neg.detach())
                loss = loss1 + opt.lamb * loss2
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * y.shape[0]
                loss_sum1 += loss1.item() * y.shape[0]
                loss_sum2 += loss2.item() * y.shape[0]
                n += y.shape[0]
        train_loss = loss_sum / n
        train_loss1 = loss_sum1 / n
        train_loss2 = loss_sum2 / n
        val_loss, val_loss1, val_loss2, _ = evaluateModel(model, val_iter, YCov is not None)
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
        logger.info("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, train_loss1, train_loss2, "validation loss:", val_loss, val_loss1, val_loss2)
        with open(epochlog_path, 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.6f, %s, %.6f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    # torch_score = train_loss
    loss, loss1, loss2, YS_pred = evaluateModel(model, trainval_iter, YCov is not None)
    logger.info('trainval loss, loss1, loss2', loss, loss1, loss2)
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS = YS[:YS_pred.shape[0], ...]
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    YS, YS_pred = YS.reshape(-1, opt.seq_len, YS.shape[-1]), YS_pred.reshape(-1, opt.seq_len, YS_pred.shape[-1])
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    logger.info('*' * 40)
    logger.info("%s, %s, Torch MSE, %.6f, %.6f, %.6f" % (name, mode, train_loss, train_loss1, train_loss2))
    logger.info("%s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (name, mode, MSE, RMSE, MAE, MAPE))
    logger.info('Model Training Ended ...', time.ctime())
    
def testModel(name, mode, XS, YS, YCov, YMask=None):
    def testScore(YS, YS_pred, message):
        logger.info(message)
        logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
        logger.info("%s, %s, Torch MSE, %.6f, %.6f, %.6f" % (name, mode, loss, loss1, loss2))
        logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (name, mode, MSE, RMSE, MAE, MAPE))
        with open(score_path, 'a') as f:
            f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
            for i in range(opt.seq_len):
                MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[..., i], YS_pred[..., i])
                logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
                f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        return None

    logger.info('Model Testing Started ...', time.ctime())
    logger.info('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)

    model = getModel()
    model.load_state_dict(torch.load(modelpt_path))

    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    if YCov is not None:
        YCov_torch = torch.Tensor(YCov).to(device)
        test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, YCov_torch)
    else:
        test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, opt.batch_size, shuffle=False) # drop_last=True
    
    loss, loss1, loss2, YS_pred = evaluateModel(model, test_iter, YCov is not None)
    logger.info('test loss, loss1, loss2', loss, loss1, loss2)
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS = YS[:YS_pred.shape[0], ...]
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    YS, YS_pred = YS.reshape(-1, opt.seq_len, YS.shape[-1]), YS_pred.reshape(-1, opt.seq_len, YS_pred.shape[-1])
    YS, YS_pred = YS.transpose(0, 2, 1), YS_pred.transpose(0, 2, 1)
    # np.save(path + f'/{name}_prediction.npy', YS_pred)
    # np.save(path + f'/{name}_groundtruth.npy', YS)
    # np.save(path + f'/{name}_mask.npy', YMask)
    testScore(YS, YS_pred, '********* Evaluation on the whole testing dataset *********')
    testScore(YS[YMask], YS_pred[YMask], '********* Evaluation on the selected/masked testing dataset (when incident happen) *********')
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
parser.add_argument('--month', type=str, default='202111', help='which experiment setting (month) to run')
parser.add_argument('--city', type=str, default='tokyo', help='which experiment setting (city) to run')
parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
parser.add_argument('--incident', type=bool, default=False, help='whether to use incident flag')
parser.add_argument('--accident', type=bool, default=False, help='whether to use accident flag')
parser.add_argument('--time', type=bool, default=False, help='whether to use float time embedding')
parser.add_argument('--history', type=bool, default=False, help='whether to use historical data')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--hiddenunits', type=int, default=32, help='number of hidden units')
parser.add_argument('--mem_num', type=int, default=10, help='number of memory')
parser.add_argument('--mem_dim', type=int, default=32, help='dimension of memory')
parser.add_argument("--memory", type=str, default='local', help="which type of memory: local or any other")
parser.add_argument("--meta", type=str, default='yes', help="whether to use meta-graph: yes or any other")
parser.add_argument("--decoder", type=str, default='stepwise', help="which type of decoder: stepwise or stepwise")
parser.add_argument('--ycov', type=str, default='time', help='which ycov to use: time or history')
parser.add_argument('--exchannel', type=int, default=0, help='how many extra/external channels for the encoder, 0 means extra input for memory')
parser.add_argument('--go', type=str, default='last', help='which type of decoder go: random or last')
parser.add_argument('--model', type=str, default='MMGCRN', help='which model to use')
parser.add_argument('--gpu', type=int, default=3, help='which gpu to use')
parser.add_argument('--lamb', type=float, default=0.5, help='lamb value to balance the MAE loss and Triplet loss')
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
feature_list = ['speed_typea']
if opt.ycov=='time':
    opt.time = True
elif opt.ycov=='history':
    opt.history = True
else:
    assert False, 'ycov type must be float time or float history value'
if opt.incident: feature_list.append('accident_flag')
if opt.accident: feature_list.append('real_accident_flag')
if opt.time: feature_list.append('weekdaytime')
if opt.history: feature_list.append('speed_typea_y')
opt.channelin += opt.exchannel
# if opt.xcov: opt.channelin = len(feature_list) # Here is important, input for the encoder is just speed, w/o xcov is always better.
# feature_list = ['speed_typea', 'accident_flag', 'real_accident_flag', 'weekdaytime', 'speed_typea_y']

_, filename = os.path.split(os.path.abspath(sys.argv[0]))
filename = os.path.splitext(filename)[0]
model_name = opt.model
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

logger.info('lamb', opt.lamb)
logger.info('experiment_city', opt.city)
logger.info('experiment_month', opt.month)
logger.info('model_name', opt.model)
logger.info('memory_type', opt.memory)
logger.info('mem_num', opt.mem_num)
logger.info('mem_dim', opt.mem_dim)
logger.info('meta_type', opt.meta)
logger.info('decoder_type', opt.decoder)
logger.info('go_type', opt.go)
logger.info('ycov_type', opt.ycov)
logger.info('exchannel', opt.exchannel)
logger.info('batch_size', opt.batch_size)
logger.info('rnn_units', opt.hiddenunits)
logger.info('num_layers', opt.num_layers)
logger.info('channnel_in', opt.channelin)
logger.info('channnel_out', opt.channelout)
logger.info('feature_incident', opt.incident)
logger.info('feature_accident', opt.accident)
logger.info('feature_time', opt.time)
logger.info('feature_history', opt.history)
#####################################################################################################

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
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
    # train_onehottime = [get_onehottime(config[month]['traffic_path']) for month in train_month]
    # test_onehottime = [get_onehottime(config[month]['traffic_path']) for month in test_month]
    # trainXTE, trainYTE = getXSYS(train_onehottime, opt.his_len, opt.seq_len)
    # testXTE, testYTE = getXSYS(test_onehottime, opt.his_len, opt.seq_len)
        
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
    trainXS, trainYS, trainXCov, trainYCov = refineXSYS(trainXS, trainYS)
    logger.info('TRAIN XS.shape YS.shape, XCov.shape, YCov.shape', trainXS.shape, trainYS.shape, trainXCov.shape, trainYCov.shape)
    trainModel(model_name, 'train', trainXS, trainYS, trainYCov)
 
    logger.info(opt.city, opt.month, 'testing started', time.ctime())
    testXS, testYS = getXSYS(test_data, opt.his_len, opt.seq_len)
    _, testYSFlag = getXSYS(test_flag, opt.his_len, opt.seq_len)
    testYMask = testYSFlag[:, 0, :, 0] > 0 # (B, N) incident happen at the first prediction timeslot.
    testXS, testYS, testXCov, testYCov = refineXSYS(testXS, testYS)
    logger.info('TEST XS.shape, YS.shape, XCov.shape, YCov.shape, testYMask.shape', testXS.shape, testYS.shape, testXCov.shape, testYCov.shape, testYMask.shape)
    testModel(model_name, 'test', testXS, testYS, testYCov, testYMask) 

    
if __name__ == '__main__':
    main()

