#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os
import time
import datetime as dt
MAY_CAN_PATH = '/data2/jiang/Toyota/CAN_day_data_1122ver/'
OCT_CAN_PATH = '/data2/jiang/Toyota/CAN_day_data_202110/'
NOV_CAN_PATH = '/data2/jiang/Toyota/CAN_day_data_202111/'
CAN_AGG_PATH = '/data2/jiang/workToyota/data/CAN_Aggregated/'
CAPITAL_LINK_FILE = '/data2/jiang/Toyota/graph_data/capital_graph_link_info.csv'
SAMPLING_RATE = '10min'      #Defining time-interval for aggregation
LAT_MIN = 35.36
LAT_MAX = 35.90
LON_MIN = 139.537
LON_MAX = 139.947


# In[3]:


#Function to extract linkids in give rectangular coordinates
def clip_CAN(df):
    #df = pd.read_csv(os.path.join(NEW_LINK_PATH, 'link_connect_all.csv'))
    #df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['start_lon'], df['start_lat'])).set_crs(epsg=4326)
    dft = df[(df['mmlatitude']>=LAT_MIN) & (df['mmlatitude']<=LAT_MAX) & (df['mmlongitude']>=LON_MIN) & 
            (df['mmlongitude']<=LON_MAX)]
    #df = gpd.clip(df, mask)
    return dft


# In[2]:


#Function1- (Aggregating the avg speed for each linkid and saving in monthly files (without reindexing))
def Aggregate_AvgSpeed_CAN(PATH_VAR, month_var):
    
    #mask = Polygon([(LON_MIN,LAT_MAX), (LON_MAX,LAT_MAX), (LON_MAX,LAT_MIN), (LON_MIN,LAT_MIN)])
    
    files = [os.path.join(PATH_VAR+filename) for filename in os.listdir(PATH_VAR)]
    files.sort()
    
    tik = time.time()
    df_month = []
    for filename in files:
        #Reading the file, dropping NA values, and clipping it
        df = pd.read_csv(filename, compression='gzip')
        df = df.dropna(subset=['speed_typea'])
        df = clip_CAN(df)
        
        #Sampling and getting avg speed after aggregation
        df['gps_timestamp'] = pd.to_datetime(df['gps_timestamp'])
        df['gps_timestamp'] = df['gps_timestamp'].dt.floor(SAMPLING_RATE)    
        df = df.groupby(['linkid','gps_timestamp'], as_index=False)['speed_typea'].mean()[['linkid',"gps_timestamp",
                                                                                         "speed_typea"]]

        #Appending the one day result
        df_month.append(df)
    
    del df
    df_month = pd.concat(df_month)
    df_month.to_csv(CAN_AGG_PATH+month_var+'_CAN.csv.gz', compression='gzip', index=False)
    
    tok = time.time()
    #print(tok-tik)


# In[7]:


#Function2
#Getting combined list of all unique linkids from all monthly files
def get_linkid_list(can_files):
    files = [os.path.join(CAN_AGG_PATH,filename) for filename in can_files]
    
    linkid_list = []

    for filename in files:
        tik = time.time()
        print('Processing '+filename)
        df = pd.read_csv(filename, compression='gzip')
        linkid_list = linkid_list + df.linkid.unique().tolist()
        tok = time.time()
        #print(tok-tik)
    
    linkid_list = np.unique(np.array(linkid_list))
    return linkid_list


# In[8]:


#Function 3
#Doing the reindexing for all linkids (saving in day-wise file)
def reindex_can_day(filename, linkid_list):
    #files = [os.path.join(CAN_AGG_PATH,filename) for filename in can_files]
    filename = os.path.join(CAN_AGG_PATH, filename)
    
    tik = time.time()
    print('Processing '+filename)
    df = pd.read_csv(filename, compression='gzip')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    df['gps_timestamp'] = pd.to_datetime(df['gps_timestamp'])
    first_date = df['gps_timestamp'].iloc[0].date()
    last_date = df['gps_timestamp'].iloc[-1].date()
    dayslices = pd.date_range(first_date, last_date, freq='1D')
    
    for day in dayslices:
        df_tmp = df[df['gps_timestamp'].dt.date==day]
        timeslices = pd.date_range(day, day+dt.timedelta(days=1), freq=SAMPLING_RATE)[:-1]
        #mux = pd.MultiIndex.from_product([linkid_list, timeslices],names=['linkid', 'gps_timestamp'])
        #df_tmp = df_tmp.set_index(['linkid', 'gps_timestamp']).reindex(mux).reset_index()
        y = pd.DataFrame([], index=pd.MultiIndex.from_product([linkid_list, timeslices],names=['linkid', 'gps_timestamp']))
        y = y.merge(df_tmp, on=['linkid', 'gps_timestamp'], how='left').reset_index().drop(['index'],axis=1)
        y.to_csv(os.path.join(CAN_AGG_PATH,'CAN_Daywise_Reindexed',day.strftime('%Y%m'),day.strftime('%Y%m%d')+
                                   '_CAN_Reindexed.csv.gz'), compression='gzip', index=False)

    tok = time.time()
    #print(tok-tik)


# In[11]:


#Function4
#Extracting capital linkids only and doing the reindexing (saving in monthly files)
def reindex_can_capital(filename, capital_linkid_list):
    
    filename = os.path.join(CAN_AGG_PATH, filename)

    tik = time.time()
    print('Processing '+filename)
    
    df = pd.read_csv(filename, compression='gzip')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df = df[df['linkid'].isin(capital_linkid_list)]
    #print(len(df))
    
    df['gps_timestamp'] = pd.to_datetime(df['gps_timestamp'])
    first_date = df['gps_timestamp'].iloc[0].date()
    last_date = df['gps_timestamp'].iloc[-1].date()
    timeslices = pd.date_range(first_date, last_date+dt.timedelta(days=1), freq=SAMPLING_RATE)[:-1]
    
    mux = pd.MultiIndex.from_product([capital_linkid_list, timeslices],names=['linkid', 'gps_timestamp'])
    df = df.set_index(['linkid', 'gps_timestamp']).reindex(mux).reset_index()
    df.to_csv(filename[:-7]+'_CAPITAL_Reindexed.csv.gz', compression='gzip', index=False)
    
    #print(df.linkid.nunique())
    #print(len(df))
    #print(df.speed_typea.isna().sum())
    tok = time.time()
    #print(tok-tik)


# In[12]:


#Function5
#Creating Accident Tensor based on reindexed Capital linkids (monthly files)
def create_accident_tensor_capital(filename, capital_linkid_list):

    tik = time.time()
    print('Processing '+filename)
    
    try:
        df = pd.read_csv(filename)
    except:
        df = pd.read_csv(filename, encoding='shift-jis')
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['start_time'] = df['start_time'].dt.floor(SAMPLING_RATE)
    df['end_time'] = df['end_time'].dt.floor(SAMPLING_RATE)
    
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df = df[df['coord_start_upstream_nearestlink'].isin(capital_linkid_list)]
    
    #first_date = df['start_time'].iloc[0].date()
    month = df['start_time'].iloc[0].month
    year = df['start_time'].iloc[0].year
    first_date = pd.to_datetime(str(year)+'-'+str(month)+'-01')
    last_date = df['start_time'].max().date()
    timeslices = pd.date_range(first_date, last_date+dt.timedelta(days=1), freq=SAMPLING_RATE)[:-1]
    #print(timeslices[0], timeslices[-1])
    
    #mux = pd.MultiIndex.from_product([capital_linkid_list, timeslices],names=['linkid', 'gps_timestamp'])
    y = pd.DataFrame([], index=pd.MultiIndex.from_product([capital_linkid_list, timeslices],
                                                          names=['linkid', 'gps_timestamp'])).reset_index()
    y['accident_flag'] = 0
    for _,row in df.iterrows():
        #y.loc[(row['coord_start_upstream_nearestlink'],row['start_time']):(row['coord_start_upstream_nearestlink'],
        #                                                                   row['end_time']), 'accident_flag'] = 1
        y['accident_flag'] = np.where((y['linkid']==row['coord_start_upstream_nearestlink']) & 
                (y['gps_timestamp']>=row['start_time']) & (y['gps_timestamp']<=row['end_time']), 1, y['accident_flag'])
        
    y.to_csv(CAN_AGG_PATH+'ACCIDENT_'+first_date.strftime('%Y-%m')+'_CAPITAL.csv.gz', compression='gzip', index=False)
    
    #print(df.linkid.nunique())
    #print(len(df))
    #print(df.speed_typea.isna().sum())
    tok = time.time()
    #print(tok-tik)


# In[13]:


#Function6
#Filling NaN values => (2) followed by (1)
##(1): xxInterpolationxx Forward Fill followed by bfill (daily basis) based on last xxand nextxx available values (but it 
##     wouldn't be very accurate)
##(2): Taking average of speed for that time period across all days (wherever data is available) for that particular linkid 
##     - combined with weekday and accident_flag information (there is possibility of having no value available)
def fill_nan_can_capital(can_file, acc_file):

    tik = time.time()
    print('Processing '+can_file)
    df = pd.read_csv(can_file)
    df['accident_flag'] = pd.read_csv(acc_file)['accident_flag']
    
    df['gps_timestamp'] = pd.to_datetime(df['gps_timestamp'])
    df['weekday'] = df['gps_timestamp'].dt.weekday
    df['weekday'] = np.where(df['weekday']>=5, 0, 1)
    df['interval'] = df.index%144
    df['time'] = df['gps_timestamp'].dt.time
    
    #Method (2)
    df_mean = df[df.accident_flag==0].groupby(by=['linkid','time','weekday']).speed_typea.mean().reset_index(name='speed_avg')
    df = df.merge(df_mean, on=['linkid','time','weekday'], how='left')
    df['speed_typea'] = np.where(df['accident_flag']==0, df['speed_typea'].fillna(df['speed_avg']), df['speed_typea'])
    
    #Method (1)
    df['speed_typea'] = np.where(df['speed_typea'].isna() & (df['interval']==0), -2, df['speed_typea'])
    df['speed_typea'] = np.where(df['accident_flag']==0, df['speed_typea'].fillna(method='ffill'), df['speed_typea'])
    df['speed_typea'] = np.where(df['speed_typea']==-2, np.nan, df['speed_typea'])
    df['speed_typea'] = np.where(df['accident_flag']==0, df['speed_typea'].fillna(method='bfill'), df['speed_typea'])
    df['speed_typea'] = np.where(df['speed_typea'].isna() & (df['accident_flag']==1), -1, df['speed_typea'])
    
    #Saving File
    df[['linkid','gps_timestamp','speed_typea','accident_flag']].to_csv(can_file[:-7]+'_FilledNA.csv.gz',compression='gzip',
                                                                       index=False)
    tok = time.time()
    #print(tok-tik)


# In[14]:


#Function7
#Creating Real Accident (filtered from cause column) Tensor based on reindexed Capital linkids (monthly files)
def create_real_accident_tensor_capital(filename, capital_linkid_list):
    tik = time.time()
    print('Processing '+filename)
    
    df = pd.read_csv(filename)
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['start_time'] = df['start_time'].dt.floor(SAMPLING_RATE)
    df['end_time'] = df['end_time'].dt.floor(SAMPLING_RATE)
    
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df = df[df['coord_start_upstream_nearestlink'].isin(capital_linkid_list)]
    
    #first_date = df['start_time'].iloc[0].date()
    month = df['start_time'].iloc[0].month
    year = df['start_time'].iloc[0].year
    first_date = pd.to_datetime(str(year)+'-'+str(month)+'-01')
    last_date = df['start_time'].max().date()
    timeslices = pd.date_range(first_date, last_date+dt.timedelta(days=1), freq=SAMPLING_RATE)[:-1]
    print(timeslices[0], timeslices[-1])
    
    #mux = pd.MultiIndex.from_product([capital_linkid_list, timeslices],names=['linkid', 'gps_timestamp'])
    y = pd.DataFrame([], index=pd.MultiIndex.from_product([capital_linkid_list, timeslices],
                                                          names=['linkid', 'gps_timestamp'])).reset_index()
    y['real_accident_flag'] = 0
    #df['accident_flag'] = 1
    for _,row in df.iterrows():
        #y.loc[(row['coord_start_upstream_nearestlink'],row['start_time']):(row['coord_start_upstream_nearestlink'],
        #                                                                   row['end_time']), 'accident_flag'] = 1
        y['real_accident_flag'] = np.where((y['linkid']==row['coord_start_upstream_nearestlink']) & 
                (y['gps_timestamp']>=row['start_time']) & (y['gps_timestamp']<=row['end_time']), 1, y['real_accident_flag'])
        
    y.to_csv(CAN_AGG_PATH+'REAL_ACCIDENT_'+first_date.strftime('%Y-%m')+'_CAPITAL.csv.gz', compression='gzip', index=False)
    
    tok = time.time()
    #print(tok-tik)


# In[15]:


#Function8
#Merging the FilledNA files with real_accident_flag data
#columns -> linkid, gps_timestamp, speed_typea, accident_flag, realaccident_flag
#※here the accident_flag corresponds to the original incident data, while real_accident_flag is the newly added 
#  real_accident flag.
#if realaccident_flag=1, accident_flag must =1, but not vise versa. 
#Because the realaccident data is a pure subset of accident data (incident data).
def merge_fillednan_realaccident(can_file, acc_file):
    tik = time.time()
    print('Processing '+can_file)
    df = pd.read_csv(can_file)
    df['real_accident_flag'] = pd.read_csv(acc_file)['real_accident_flag']
    
    #Saving File
    df.to_csv(can_file[:-7]+'_IncidentAccident.csv.gz',compression='gzip', index=False)
    tok = time.time()
    #print(tok-tik)


# In[1]:


#Main Function
if __name__=='__main__':
    
    #1-Aggregating the average speed for each linkid in CAN files and saving them in monthly files without reindexing
    Aggregate_AvgSpeed_CAN(MAY_CAN_PATH, 'MAY')
    Aggregate_AvgSpeed_CAN(OCT_CAN_PATH, 'OCT')
    Aggregate_AvgSpeed_CAN(NOV_CAN_PATH, 'NOV')
    
    #2-Get combined unique LinkID list
    can_files = ['MAY_CAN.csv.gz', 'OCT_CAN.csv.gz', 'NOV_CAN.csv.gz']
    linkid_list = get_linkid_list(can_files)
    
    #3-Reindex the monthly aggregated CAN files with above linkid list for all possible time-slots (day-wise)
    reindex_can_day(can_files[0], linkid_list)
    reindex_can_day(can_files[1], linkid_list)
    reindex_can_day(can_files[2], linkid_list)
    
    #4-Reindexing the monthly aggregated CAN for caital links only (month-wise) 
    df_capital_link = pd.read_csv(CAPITAL_LINK_FILE)
    capital_linkid_list = df_capital_link['link_id'].unique()
    reindex_can_capital(can_files[0], capital_linkid_list)
    reindex_can_capital(can_files[1], capital_linkid_list)
    reindex_can_capital(can_files[2], capital_linkid_list)

    #5-Creating accident tensors for capital linkids (monthly files)
    acc_files = ['/data2/jiang/Toyota/JARTIC_data_202105/vics_regulation_202105_shutokou.csv',
            '/data2/jiang/Toyota/JARTIC_data_202110/vics_regulation_202110_C01_2.csv',
            '/data2/jiang/Toyota/JARTIC_data_202111/vics_regulation_202111_C01.csv']
    create_accident_tensor_capital(acc_files[0], capital_linkid_list)
    create_accident_tensor_capital(acc_files[1], capital_linkid_list)
    create_accident_tensor_capital(acc_files[2], capital_linkid_list)
    
    #6-Filling NaN values in reindexed can capital files
    can_capital_files = [CAN_AGG_PATH+'MAY_CAN_CAPITAL_Reindexed.csv.gz', CAN_AGG_PATH+'OCT_CAN_CAPITAL_Reindexed.csv.gz',
             CAN_AGG_PATH+'NOV_CAN_CAPITAL_Reindexed.csv.gz']
    acc_tensor_files = [CAN_AGG_PATH+'ACCIDENT_2021-05_CAPITAL.csv.gz', CAN_AGG_PATH+'ACCIDENT_2021-10_CAPITAL.csv.gz',
             CAN_AGG_PATH+'ACCIDENT_2021-11_CAPITAL.csv.gz']
    fill_nan_can_capital(can_capital_files[0], acc_tensor_files[0])
    fill_nan_can_capital(can_capital_files[1], acc_tensor_files[1])
    fill_nan_can_capital(can_capital_files[2], acc_tensor_files[2])
    
    #7-Create Real Accident Tensor for capital linkids (monthly files)
    real_acc_files = ['/data2/jiang/Toyota/JARTIC_data_202105/vics_accident_202105.csv', 
                      '/data2/jiang/Toyota/JARTIC_data_202110/vics_accident_202110.csv',
                      '/data2/jiang/Toyota/JARTIC_data_202111/vics_accident_202111.csv']
    create_real_accident_tensor_capital(real_acc_files[0], capital_linkid_list)
    create_real_accident_tensor_capital(real_acc_files[1], capital_linkid_list)
    create_real_accident_tensor_capital(real_acc_files[2], capital_linkid_list)
    
    #8-Merging the filledNA files with real_accident_flag data
    filledna_can_files = [CAN_AGG_PATH+'MAY_CAN_CAPITAL_Reindexed_FilledNA.csv.gz', 
                          CAN_AGG_PATH+'OCT_CAN_CAPITAL_Reindexed_FilledNA.csv.gz',
                          CAN_AGG_PATH+'NOV_CAN_CAPITAL_Reindexed_FilledNA.csv.gz']
    real_acc_tensor_files = [CAN_AGG_PATH+'REAL_ACCIDENT_2021-05_CAPITAL.csv.gz', 
                             CAN_AGG_PATH+'REAL_ACCIDENT_2021-10_CAPITAL.csv.gz',
                             CAN_AGG_PATH+'REAL_ACCIDENT_2021-11_CAPITAL.csv.gz']
    merge_fillednan_realaccident(filledna_can_files[0], real_acc_tensor_files[0])
    merge_fillednan_realaccident(filledna_can_files[1], real_acc_tensor_files[1])
    merge_fillednan_realaccident(filledna_can_files[2], real_acc_tensor_files[2])

