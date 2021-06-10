import numpy as np
from datetime import datetime
from datetime import datetime, timedelta 
import time
from InfluxDB_Reader import InfluxDB_Reader
from InfluxDB_Writer import InfluxDB_Writer
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib
import imageio
import time
import torch.nn as nn
import torch.nn.functional as F
from models import ConvAutoencoder,ResNet
import pdb



model = ConvAutoencoder()
print(model)
model.load_state_dict(torch.load('C://Users//RIO//Desktop//RIO_LUCA//June21//RIO//weights//CNNAE_May.pt'))
scaler_time = joblib.load('C://Users//RIO//Desktop//RIO_LUCA//June21//RIO//scalers//scalers')
quant = np.load('C://Users//RIO//Desktop//RIO_LUCA//June21//RIO//quants//quants', allow_pickle=True)





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


database = 'APP3DB1_0' 
print(database)
measurmeent_keys = source_db.measurements[database][:]
print(np.max(np.array(source_db.tags[database]['FusePicked']['SessionID']).astype(int)))
keys = ['FeederAction1','FeederAction2', 'FeederAction3', 'FeederAction4',
'NumberEmptyFeeder', 'NumberFuseDetected', 'FusePicked',
'NumberFuseEstimated', 'DurationPickToPick', 'FeederBackgroundIlluminationIntensity', 
'IntensityTotalImage', 'SharpnessImage', 'FuseIntoFeeder', 'Pressure',  'Vacuum', 
 'VacuumFusePicked', 'VacuumValveClosed', 'FuseHeatSlope', 'FuseTestResult', 'FuseCycleDuration', 
'SmartMotorSpeed']


idsess = str(26)
start_, end_ = source_db.get_time_range('DurationPickToPick',database, where = {'SessionID': idsess})
end_time = datetime.fromisoformat(end_[:26])
print(end_)
print(end_time)



length = 0
count = 0
condition = False
initial_size = 10
signals = np.ones([initial_size,1])


while condition == False:
    time.sleep(2)
    print('################# iteration number:  ' + str(count) + '  ####################')
    print()
    end_time = end_time + timedelta(seconds = 2)
    print(end_time)
    #within a session of measurements have the same initial and start times, so I've just picked one randomly
    all_signals = source_db.query('DurationPickToPick', {'*': ''}, database, time_range = [str(end_), str(end_time)])
    #this condition is inserted in order to discard the first few datapoints that ofthen contain nans and other anomalies
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
        signals_f = signals[initial_size-10:,1:]
        times = np.zeros([10,21])
        for j in range(21):
            temp1 = signals_f[:,j]
            temp1[np.isnan(temp1)] = 0
            times[:,j] = temp1[:]
        condition = True


dest_db_dict = {k:[] for k in columns}


images = []
fig, axs = plt.subplots(7, 3, figsize=(11,11))
fig.tight_layout(pad=3.0)
count = 0
length = 0
size = 10
cnt = initial_size-size
interval = 5

m = 0
condition = False
signals = np.ones([size,1])
indices = temp.index
plt.ion()
plt.show()
while condition == False:
     count +=1
     start_time = datetime.fromisoformat(str(indices[cnt]))
     end_time = datetime.fromisoformat(str(end_time)[:26])
     end_time = end_time + timedelta(seconds = interval)
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
         times = np.zeros([10,21])
         for j in range(21):
             temp = signals_f[:,j]
             temp[np.isnan(temp)] = 0
             times[:,j] = scaler_time[j].transform(np.expand_dims(temp,1))[:,0]  
         train = torch.tensor(times.T).unsqueeze(0).float()
         train = torch.clamp(train, min=-1e5, max=5e1)
         
     ################### Inference and Plot part ###########################################################
        
         model.eval()
         T_O, T_ENC = model(train)
         diff = T_O.detach() - train
         diff_sq = np.abs(diff.detach().numpy())
         tra = np.sum(diff_sq,axis=2)
         cnt2 = 0
         if m%1 == 0:
             for s in range(7):
                 for j in range(3):
                     #axs[s,j].set_xlim(-1,102)
                     axs[s,j].axhline(y=1.*quant[cnt2], color='g', linestyle='-')
                     #axs[s,j].plot(np.arange(-1,102,103/1000),1.35*quant[cnt2]*np.ones(1000), color = 'g')
                     axs[s,j].scatter(m,tra[:,cnt2], color = 'b')
                     axs[s,j].set_title(keys[cnt2])
                     if tra[:,cnt2]>1.*quant[cnt2]:
                         current_signal = float(1)
                         axs[s,j].set_facecolor('xkcd:salmon')
                         axs[s,j].set_facecolor((1.0, 0.47, 0.42))
                     else:
                         current_signal = float(0)
                         axs[s,j].set_facecolor('xkcd:white')                
                     dest_db_dict['unit'].append(unit)
                     dest_db_dict['meas'].append(keys[cnt2])
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


    
