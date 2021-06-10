
import sys
sys.path.insert(1, 'C:\\Users\\RIO\\Desktop\RIO_LUCA\\')
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from datetime import datetime, timedelta 
import time
from Utils.InfluxDB_Reader import InfluxDB_Reader
from Utils.InfluxDB_Writer import InfluxDB_Writer
from matplotlib import pyplot as plt
import seaborn as sns
from Utils.mapminmax import mapminmax, mapminmax_a
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib
import imageio
import time
import torch.nn as nn
import torch.nn.functional as F
from model import ConvAutoencoder



model = ConvAutoencoder()
print(model)
model.load_state_dict(torch.load('C:\\Users\\RIO\\Desktop\\RIO_LUCA\\Pre-processing\\CNNAE_big_f.pt'))
scaler_time = joblib.load('C:\\Users\\RIO\\Desktop\\RIO_LUCA\\Pre-processing\\scaler_time.pkl')
scaler_freq = joblib.load('C:\\Users\\RIO\\Desktop\\RIO_LUCA\\Pre-processing\\scaler_freq.pkl')
quant = np.load('C:\\Users\\RIO\\Desktop\\RIO_LUCA\\Pre-processing\\quant.pkl.npy')





source_db_param = {'address':'138.131.217.85',
                    'port':8086,
                    'user':'admin',
                    'password':'1234'}
source_db = InfluxDB_Reader(source_db_param['address'], source_db_param['port'], 
                            source_db_param['user'], source_db_param['password'], 
                            data_frame_client_flag = True)




dest_db_param = {'address':'138.131.217.85',
                    'port':8086,
                    'user':'admin',
                    'password':'1234'}
destination_db = InfluxDB_Writer('Try', 
                                 dest_db_param['address'], dest_db_param['port'], 
                                 dest_db_param['user'], dest_db_param['password'], 
                                 data_frame_client_flag = True)
unit = '53'
field_names = ['time','signal']
tag_names = ['unit','meas']
columns = field_names + tag_names



#database = source_db.databases[-1] 
database = 'APP3DB1_0' 
print(database)
measurmeent_keys = source_db.measurements[database][:]
keys = ['FeederAction1','FeederAction2', 'FeederAction3', 'FeederAction4',
'NumberEmptyFeeder', 'NumberFuseDetected', 'FusePicked',
'NumberFuseEstimated', 'DurationPickToPick', 'FeederBackgroundIlluminationIntensity', 
'IntensityTotalImage', 'SharpnessImage', 'FuseIntoFeeder', 'Pressure',  'Vacuum', 
 'VacuumFusePicked', 'VacuumValveClosed', 'FuseHeatSlope', 'FuseTestResult', 'FuseCycleDuration', 
'SmartMotorSpeed']


# list[k][j]

# [B, 2*n_signals, 68] n_signals = 32


#TODO dynamic session ID
start_, end_ = source_db.get_time_range('DurationPickToPick',database, where = {'SessionID': '73'})
end_time = datetime.fromisoformat(end_[:26])
print(end_)
print(end_time)





length = 0
count = 0
condition = False
initial_size = 80
signals = np.ones([initial_size,1])




#TODO pbar
while condition == False:
    time.sleep(2)
    print('################# iteration number:  ' + str(count) + '  ####################')
    print()
    end_time = end_time + timedelta(seconds = 5)
    #within a session of measurements have the same initial and start times, so I've just picked one randomly
    all_signals = source_db.query('DurationPickToPick', {'*': ''}, database, time_range = [str(end_), str(end_time)])
    count = count + 1
    #this condition is inserted in order to discard the first few datapoints that ofthen contain nans and other anomalies
    if count >7:
        length = len(all_signals['value'].values)
        print('################# vector length:  ' + str(length) + ' ####################')
        print()
        if length==initial_size:
            for k in keys:
                temp = source_db.query(k, {'*': ''}, database, time_range = [str(end_), str(end_time)])
                if 'value' in temp.keys():
                    temp_ = np.expand_dims(temp['value'].values, axis = 1)
                else:
                    temp_ = np.expand_dims(temp['vCnt'].values, axis = 1)
                signals = np.concatenate([signals,temp_], axis = 1)
            signals_f = signals[initial_size-70:,1:]
            fft_H_U = np.zeros([68,21])
            times = np.zeros([68,21])
            for j in range(21):
                temp1 = signals_f[:,j]
                temp_ = np.mean(temp1[np.isnan(temp1) == False])
                temp1[np.isnan(temp1)] = temp_
                fft_H_U[:,j] = abs(np.fft.fft(temp1[1:70]))[1:70]
                times[:,j] = temp1[1:69]
            fft_H_U_n = scaler_freq.transform(fft_H_U)
            time_n = scaler_time.transform(times)
            train = torch.tensor(np.concatenate([fft_H_U_n, time_n],axis = 1).T).unsqueeze(0).float()
            train = torch.clamp(train, min=-1e5, max=5e1)
            condition = True





keys2 = ['FeederAction1_F','FeederAction2_F', 'FeederAction3_F', 'FeederAction4_F',
'NumberEmptyFeeder_F', 'NumberFuseDetected_F', 'FusePicked_F',
'NumberFuseEstimated_F', 'DurationPickToPick_F', 'FeederBackgroundIlluminationIntensity_F', 
'IntensityTotalImage_F', 'SharpnessImage_F', 'FuseIntoFeeder_F', 'Pressure_F',  'Vacuum_F', 
 'VacuumFusePicked_F', 'VacuumValveClosed_F', 'F
 
 useHeatSlope_F', 'FuseTestResult_F', 'FuseCycleDuration_F', 
'SmartMotorSpeed_F', 'FeederAction1','FeederAction2', 'FeederAction3', 'FeederAction4',
'NumberEmptyFeeder', 'NumberFuseDetected', 'FusePicked',
'NumberFuseEstimated', 'DurationPickToPick', 'FeederBackgroundIlluminationIntensity', 
'IntensityTotalImage', 'SharpnessImage', 'FuseIntoFeeder', 'Pressure',  'Vacuum', 
 'VacuumFusePicked', 'VacuumValveClosed', 'FuseHeatSlope', 'FuseTestResult', 'FuseCycleDuration', 
'SmartMotorSpeed']
dest_db_dict = {k:[] for k in columns}





images = []
fig, axs = plt.subplots(10, 4, figsize=(18,18))
fig.tight_layout(pad=3.0)
count = 0
length = 0
size = 70
cnt = initial_size-size
interval = 5

#TODO investigate block twice
m = 0
condition = False
signals = np.ones([size,1])
indices = temp.index
plt.ion()
plt.show()
while condition == False:
    count +=1
    start_time = datetime.fromisoformat(str(indices[cnt]))
#    print(start_time)
    end_time = datetime.fromisoformat(str(end_time)[:26])
    end_time = end_time + timedelta(seconds = interval)
#    print(end_time)
    all_signals = source_db.query('DurationPickToPick', {'*': ''}, database, time_range = [str(start_time)[:26], str(end_time)])
    length = len(all_signals['value'].values)
    current_time = all_signals.index.values[-1]
    if length==size:
        starts_time = time.time()
        #print('################# vector length:  ' + str(length) + ' ####################')
        cnt += 1
        signals = np.ones([size,1])
        for k in keys:
            temp = source_db.query(k, {'*': ''}, database, time_range = [str(start_time)[:26], str(end_time)])
            if 'value' in temp.keys():
                temp_ = np.expand_dims(temp['value'].values, axis = 1)
            else:
                temp_ = np.expand_dims(temp['vCnt'].values, axis = 1)
            signals = np.concatenate([signals,temp_], axis = 1)
        index = temp.index[-1]
        indices = np.append(indices,index)
        signals_f = signals[:,1:]
        fft_H_U = np.zeros([68,21])
        times = np.zeros([68,21])
        for j in range(21):
            temp = signals_f[:,j]
            temp_ = np.mean(temp[np.isnan(temp) == False])
            temp[np.isnan(temp)] = temp_
            fft_H_U[:,j] = abs(np.fft.fft(temp[1:70]))[1:70]
            times[:,j] = temp[1:69]
        fft_H_U_n = scaler_freq.transform(fft_H_U)
        time_n = scaler_time.transform(times)
        train = torch.tensor(np.concatenate([fft_H_U_n, time_n],axis = 1).T).unsqueeze(0).float()
        train = torch.clamp(train, min=-1e5, max=5e1)
        #name = input("stop? ")
        #if name == 'stop':
        #    condition = True
         
    ################### Inference and Plot part ###########################################################
        
        
        model.eval()
        T_O, T_ENC = model(train)
        diff = T_O.detach() - train
        diff_sq = np.square(diff.detach().numpy())
        tra = np.array([np.sum(diff_sq[:,i,:], axis =1) for i in range(42)])
        cnt2 = 0
        if m%1 == 0:
            for s in range(10):
                for j in range(4):
                    if cnt2 == 19 or cnt2 == 40:
                        cnt2+=1
                    #axs[s,j].set_xlim(-1,102)
                    axs[s,j].axhline(y=1.35*quant[cnt2], color='g', linestyle='-')
                    #axs[s,j].plot(np.arange(-1,102,103/1000),1.35*quant[cnt2]*np.ones(1000), color = 'g')
                    axs[s,j].scatter(m,tra[cnt2], color = 'b')
                    axs[s,j].set_title(keys2[cnt2])
                    if tra[cnt2]>1.35*quant[cnt2]:
                        current_signal = float(1)
                        axs[s,j].set_facecolor('xkcd:salmon')
                        axs[s,j].set_facecolor((1.0, 0.47, 0.42))
                    else:
                        current_signal = float(0)
                        axs[s,j].set_facecolor('xkcd:white')                
                    dest_db_dict['unit'].append(unit)
                    dest_db_dict['meas'].append(keys2[cnt2])
                    dest_db_dict['signal'].append(current_signal)
                    dest_db_dict['time'].append(current_time)
                    dest_measurement_name = 'labels'
                    destination_db.create_new_measurement('Try', dest_measurement_name, dest_db_dict, tag_names, delete_previous_measurement = False)                    
                    cnt2+=1
            fig.canvas.draw()
            plt.draw()
            fig.canvas.start_event_loop(0.001)
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
        m+=1
        interval = time.time() - starts_time
    if m == 1000:   #change this to set the duration of the experiment----> need to change x-axis size accordingly
        condition = True
kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./try2.gif', images, fps=1)


    
