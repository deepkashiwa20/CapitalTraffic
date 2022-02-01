import numpy as np

def evaluate(y_true, y_pred, precision=10):
    # print('MSE:', round(MSE(y_true, y_pred), precision))
    # print('RMSE:', round(RMSE(y_true, y_pred), precision))
    # print('MAE:', round(MAE(y_true, y_pred), precision))
    # print('MAPE:', round(MAPE(y_true, y_pred), precision), '%')
    # print('PCC:', round(PCC(y_true, y_pred), precision))
    return MSE(y_true, y_pred), RMSE(y_true, y_pred), MAE(y_true, y_pred), MAPE(y_true, y_pred)

def MAPE(y_pred:np.array, y_true:np.array, epsilon=1.0):
    y_pred[y_pred < 0] = 0
    return np.mean(np.abs(y_pred - y_true) / np.clip(np.abs(y_true), epsilon, None)) * 100

def MSE(y_true, y_pred):
    y_pred[y_pred < 0] = 0
    return np.mean(np.square(y_pred - y_true))

def RMSE(y_true, y_pred):
    y_pred[y_pred < 0] = 0
    return np.sqrt(MSE(y_pred, y_true))

def MAE(y_true, y_pred):
    y_pred[y_pred < 0] = 0
    return np.mean(np.abs(y_pred - y_true))

# def MAPE(y_pred:np.array, y_true:np.array, epsilon=1e-3):       # avoid zero division
#     return np.mean(np.abs(y_pred - y_true) / np.clip((np.abs(y_pred) + np.abs(y_true)) * 0.5, epsilon, None))
    
# def PCC(y_pred:np.array, y_true:np.array):      # Pearson Correlation Coefficient
#     return np.corrcoef(y_pred.flatten(), y_true.flatten())[0,1]