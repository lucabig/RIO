import numpy as np
from numpy import nan
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



classes = ['class_ 0_', 'class_ 2_', 'class_ 3_', 'class_ 4_', 'class_ 5_', 'class_ 7_',
 'class_ 9_', 'class_11_', 'class_12_']

m_subset = ['FeederAction1','FeederAction2', 'FeederAction3', 'FeederAction4',
'NumberEmptyFeeder', 'NumberFuseDetected', 'FusePicked',
'NumberFuseEstimated', 'DurationPickToPick', 'FeederBackgroundIlluminationIntensity', 
'IntensityTotalImage', 'SharpnessImage', 'FuseIntoFeeder', 'Pressure',  'Vacuum', 
 'VacuumFusePicked', 'VacuumValveClosed', 'FuseHeatSlope', 'FuseTestResult', 'FuseCycleDuration', 
'SmartMotorSpeed']


def read_full_data():
    print('Loading full dataset')
    DATA = defaultdict()
    for c in classes:
        base_path = 'C://Users//lbg//OneDrive - CSEM S.A//Bureau//RIO_Data_Challenge//src//data//'
        paths = ['release1dev//','release2dev//','testdev//']
        full_data_t = []
        for p in paths:
            mypath = base_path+p
            fields = pd.read_csv(mypath[:-5]+'//fields.csv')
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            classes_ = np.unique([onlyfiles[i][:9] for i in range(len(onlyfiles))])
            idxs = defaultdict()
            for s,j in enumerate(classes_):
                idxs[j] = np.where([j in onlyfiles[i] for i in range(len(onlyfiles))])[0]
            idxs = dict(idxs)

            try:
                index = idxs[c]
                file = onlyfiles[index[0]]
                path_csv = join(mypath,file)
                data = pd.read_csv(path_csv)
                data = data.set_index(data[list(data.keys())[0]].values).iloc[:,1:]
                index = idxs[c][1:]
                for i in index:
                    file = onlyfiles[i]
                    path_csv = join(mypath,file)
                    data_t = pd.read_csv(path_csv)
                    data_t = data_t.set_index(data_t[list(data_t.keys())[0]].values).iloc[:,1:]
                    data = pd.concat([data,data_t],axis = 1)
                full_data_t.append(data)
            except KeyError:
                pass
                #print('data class', c, 'not present in the dataset', p)
        full_data = pd.concat(full_data_t,axis=1)    
        DATA[c]=full_data
    return DATA,fields



def extract_healthy(DATA,m_subset=m_subset):
    print('Extracting healthy data')
    train_data=DATA[classes[0]].iloc[:,(DATA[classes[0]].iloc[-1,:]=='0').values]
    del train_data['class_ 0_54_data']
    kk = list(train_data.keys())
    train_data_subset = train_data.loc[m_subset]
    return train_data_subset,kk


def extract_unhealthy(DATA,class_n,m_subset=m_subset):
    print('Extracting class: ', classes[class_n])
    faulty_data = DATA[classes[class_n]]
    kk_f = list(faulty_data.keys())
    faulty_data_subset = faulty_data.loc[m_subset]
    return faulty_data_subset,kk_f



def fill_nans_fields(train_data_subset,kk,fields):
    print('Nans filling and field selection')
    cnt = 0
    filtered_data = pd.DataFrame(index=m_subset,columns=kk)
    for k in kk:
        for m in m_subset:
            cnt+=1
            fields_new = fields.set_index(fields[list(fields.keys())[0]].values).iloc[:,1:]
            ff = fields_new.loc[m].values
            if 'value' in ff:
                try:
                    filtered_data.loc[m,k]=np.nan_to_num(np.array(eval(train_data_subset.loc[m,k]))[:,np.where(ff == 'value')[0].item()])
                except IndexError:
                    filtered_data.loc[m,k]=np.nan_to_num(np.array(eval(train_data_subset.loc[m,k]))[:,np.where(ff == 'value')[0].item()-1])
            elif 'vCnt' in ff:
                filtered_data.loc[m,k]=np.nan_to_num(np.array(eval(train_data_subset.loc[m,k]))[:,np.where(ff == 'vCnt')[0].item()])
    return filtered_data





def normalize(filtered_data,scaler_saved=None,m_subset=m_subset,type_scaling='minmax'):
    print('Normalizing')
    if scaler_saved==None:
        print('Using new scaler')
        scalers=[]
    else:
        print('Using input scaler')
    norma = defaultdict(list)
    norm_data = pd.DataFrame(columns=m_subset)
    lengths = []
    for i in range(len(m_subset)):
        for j in range(filtered_data.shape[1]):
            norma[m_subset[i]]=norma[m_subset[i]]+list(filtered_data.iloc[i,j])
        lengths.append(len(np.array(norma[m_subset[i]])))
    for i in range(len(m_subset)):    
        x = np.expand_dims(np.array(norma[m_subset[i]]),1)
        if scaler_saved == None:
            if type_scaling=='minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            scaler.fit(x)
            scalers.append(scaler)
            x_new = scaler.transform(x)
            norm_data[m_subset[i]] = x_new[:np.min(lengths),0]
        else:
            scaler = scaler_saved[i]
            x_new = scaler.transform(x)
            norm_data[m_subset[i]] = x_new[:np.min(lengths),0]
            scalers = scaler
    return norm_data,scalers



def chunking(norm_data,stride=5,chunk_size=20,sensors=len(m_subset)):
    print('Chunking')
    train_norm_data = np.zeros([1,sensors,chunk_size])
    cnt=0
    n = norm_data.T.iloc[0,:].shape[0]
    while cnt<n:
        try:
            train_norm_data=np.concatenate([train_norm_data,np.expand_dims(norm_data.T.iloc[:,cnt:cnt+chunk_size].values,0)])
            cnt+=stride
            if cnt%1000==0:
                print(cnt/n)
        except ValueError:
            break
    return train_norm_data[1:,:,:]


 







