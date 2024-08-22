import os
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, BatchNormalization, TimeDistributed,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from keras.layers import PReLU
%matplotlib inline
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import keras
from keras import backend as K
from keras.layers import Lambda
from tensorflow.keras.layers import PReLU
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

model = load_model(r"E:\a0130\my_model_CCI_0324_4_最終.h5")

# 讀取訓練數據
data = pd.read_csv(r'E:\1125_各因子重新升尺度\CCI\1201_SM&因子_大檔案\0322_trainX_CCI.csv')
# data = pd.read_csv(r'E:\1125_各因子重新升尺度\9K\日期\SM&因子_大檔案\0313_trainX_SMAP.csv')
# data = pd.read_csv(r'E:\1125_各因子重新升尺度\25K\1201_SM&因子_大檔案\0229_trainX.csv')

data['time'] = pd.to_datetime(data['time'])
data.sort_values(['lon', 'lat', 'time'], inplace=True)

# 讀取測試數據
# testX = np.load('E:/1125_各因子重新升尺度/5K/1201_SMAP_因子_大檔案_保存檔/0229_testX_plus_Ks.npy', allow_pickle=True)
testX = np.load(r"E:\1125_各因子重新升尺度\3K\3KM_test_datasets.npy", allow_pickle=True)
column_names = ['lon', 'lat', 'time', 'ks', 'manning', 'carbon', 'clay', 'ph', 'elevation', 'slope', 'landuse', 'band4', 'band7', 'ndvi', 'api', 'tsk', 'tslb', 'temp', 'sr']
testX = pd.DataFrame(testX, columns=column_names)
testX['time'] = pd.to_datetime(testX['time'])
testX.sort_values(['lon', 'lat', 'time'], inplace=True)


# 合併 data 和 testX 進行歸一化
combined = pd.concat([data.drop(columns=['SM']), testX])
scaler = MinMaxScaler(feature_range=(0, 1))
features_to_normalize = combined.columns.difference(['lon', 'lat', 'time']).tolist()
combined[features_to_normalize] = scaler.fit_transform(combined[features_to_normalize])

# 拆回 data 和 testX
data[features_to_normalize] = combined.iloc[:len(data), :][features_to_normalize]
testX[features_to_normalize] = combined.iloc[len(data):, :][features_to_normalize]

testX_values = testX.values
column_1 = testX_values[:, 1]

min_value = np.min(column_1)
max_value = np.max(column_1)

print(f"第一列的最小值是：{min_value}")
print(f"第一列的最大值是：{max_value}")


data_lon = data['lon'].values
data_lat = data['lat'].values
data_time = data['time'].values
time = data['time'].values
time = pd.to_datetime(time) 

 
# 定義 trainX  trainY
all_X = data.drop(['SM', 'time', 'lon', 'lat','hum'], axis=1).values
all_Y = data['SM'].values

time_steps = 1461
feature_num = 16

#所有網格按照經緯度分組
num_grids = len(data.groupby(['lon', 'lat']))

# Reshape all_X 和 all_Y 為 ConvLSTM2D需要的維度
all_X = all_X.reshape(num_grids, time_steps, 1, 1, feature_num)
all_Y = all_Y.reshape(num_grids, time_steps)


trainX = all_X[:, :-365]
trainY = all_Y[:, :-365]

valX = all_X[:, -365:]
valY = all_Y[:, -365:]

# valX = all_X[:, 730:1096]  # 因为Python索引是从0开始的
# valY = all_Y[:, 730:1096]

# # 使用除了第731筆到第1096筆以外的數據作為訓練集
# trainX = np.concatenate([all_X[:, :730], all_X[:, 1096:]], axis=1)
# trainY = np.concatenate([all_Y[:, :730], all_Y[:, 1096:]], axis=1)


trainY_flatten = trainY.flatten()
valY_flatten = valY.flatten()
# =============================================================================
model = Sequential()
model.add(ConvLSTM2D(filters = 16 , kernel_size=(7,7), activation='PReLU'
                     , return_sequences=True, padding='same', input_shape=(None, 1, 1, feature_num)))
model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.35))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.35))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.35))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.summary()
# =============================================================================

# 學習率
#第一次
learning_rate = 0.0008
#第二次
# learning_rate = 0.00005

# learning_rate = 0.00005
# print(model.optimizer.get_config())
# #優化器
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='mean_squared_error', optimizer = optimizer)
callback = EarlyStopping(monitor='val_loss', patience=600, restore_best_weights=True, mode='min')
history = model.fit(trainX, trainY, epochs=600, batch_size=16, validation_data=(valX, valY)
                    , callbacks=[callback], verbose=1)

model_path = "E:/a0130/SMAP_0323_2.h5"
model.save(model_path)


#訓練預測跟驗證預測
trainPredict = model.predict(trainX)
valPredict = model.predict(valX)

#squeeze將多餘維度去除 保留2維即可
trainPredict = np.squeeze(trainPredict)
valPredict = np.squeeze(valPredict)




#保存訓練預測數據
val_time_mask = (data['time'] >= '2018-01-01') & (data['time'] <= '2020-12-31')  # Adjust this to your validation time range

val_lon = data[val_time_mask]['lon'].values
val_lat = data[val_time_mask]['lat'].values
val_time = data[val_time_mask]['time'].values


valPredict_flatten = trainPredict.flatten()
val_result_pred = pd.DataFrame({
    'lon': val_lon,
    'lat': val_lat,
    'time': val_time,
    'SM_val_pred': valPredict_flatten
})
val_result_pred.sort_values(['time'], inplace=True)
#按網格保存
val_grouped_pred = val_result_pred.groupby(['lon', 'lat'])
for name, group in tqdm(val_grouped_pred):
    # Updated to include the last subfolder in the path
    directory = "E:/降尺度重採樣數據/0.25度/feature/模型預測結果/網格/CCI_0324_4_訓練/"
    
    # Check if the directory exists, create if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Use the directory in the filename
    filename = f"{directory}grid_val_{name[0]}_{name[1]}.csv"
    group.sort_values('time', inplace=True)
    group.to_csv(filename, index=False)
    
#保存驗證集預測數據
val_time_mask = (data['time'] >= '2021-01-01') & (data['time'] <= '2021-12-31')  # Adjust this to your validation time range

val_lon = data[val_time_mask]['lon'].values
val_lat = data[val_time_mask]['lat'].values
val_time = data[val_time_mask]['time'].values


valPredict_flatten = valPredict.flatten()
val_result_pred = pd.DataFrame({
    'lon': val_lon,
    'lat': val_lat,
    'time': val_time,
    'SM_val_pred': valPredict_flatten
})
val_result_pred.sort_values(['time'], inplace=True)

#按網格保存
val_grouped_pred = val_result_pred.groupby(['lon', 'lat'])
for name, group in tqdm(val_grouped_pred):
    # Updated to include the last subfolder in the path
    directory = "E:/降尺度重採樣數據/0.25度/feature/模型預測結果/網格/CCI_0324_4/"
    
    # Check if the directory exists, create if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Use the directory in the filename
    filename = f"{directory}grid_val_{name[0]}_{name[1]}.csv"
    group.sort_values('time', inplace=True)
    group.to_csv(filename, index=False)
    
#訓練相關係數
trainsize = 418
corrcoefs_train = np.zeros(trainsize)
for i in range(trainsize):
    observed = trainY[i, :]
    predicted = trainPredict[i, :]
    corrcoef, _ = pearsonr(observed, predicted)
    corrcoefs_train[i] = corrcoef
print(corrcoefs_train)
avg_corrcoefs_train = np.mean(corrcoefs_train)
print(avg_corrcoefs_train)

#訓練bias
biases_train = np.zeros(trainsize)
for i in range(trainsize):
    observed = trainY[i]
    predicted = trainPredict[i]
    bias = np.mean(observed - predicted)
    biases_train[i] = bias
print(biases_train)
avg_bias_train =np.mean(biases_train)
print(avg_bias_train)

#訓練RMSE
rmse_train = np.zeros(trainsize)
for i in range(trainsize):
    observed = trainY[i]
    predicted = trainPredict[i]
    mse = np.mean((observed - predicted)**2)
    rmse_train[i] = mse  #將mse存入rmse_train陣列中
rmse_train = np.sqrt(rmse_train)  #取平方根得到RMSE
print(rmse_train)
avg_rmse_train = np.mean(rmse_train)
print(avg_rmse_train)

#訓練MAE
mae_train = np.zeros(trainsize)
for i in range(trainsize):
    observed = trainY[i]
    predicted = trainPredict[i]
    mae = np.mean(np.abs(observed - predicted)) 
    mae_train[i] = mae
print(mae_train)
avg_mae_train = np.mean(mae_train)
print(avg_mae_train)

#驗證correlation
valsize = 418
corrcoefs_val = np.zeros(valsize)
for i in range(valsize):
    observed = valY[i, :]
    predicted = valPredict[i, :]
    corrcoef, _ = pearsonr(observed, predicted)
    corrcoefs_val[i] = corrcoef
avg_corrcoef_val = np.mean(corrcoefs_val)
print(avg_corrcoef_val)
#驗證bias
biases_val = np.zeros(valsize)
for i in range(valsize):
    observed = valY[i]
    predicted = valPredict[i]
    bias = np.mean(observed - predicted)
    biases_val[i] = bias
print(biases_val)
avg_bias_val =np.mean(biases_val)
print(avg_bias_val)
 
#驗證RMSE
rmse_val = np.zeros(valsize)
for i in range(valsize):
    observed = valY[i]
    predicted = valPredict[i]
    mse = np.mean((observed - predicted)**2)
    rmse_val[i] = np.sqrt(mse)
print(rmse_val)   
avg_rmse_val =np.mean(rmse_val)
print(avg_rmse_val)

#驗證MAE
mae_val = np.zeros(valsize)
for i in range(valsize):
    observed = valY[i]
    predicted = valPredict[i]
    mae = np.mean(np.abs(observed - predicted))
    mae_val[i] = mae
print(mae_val)
avg_mae_val = np.mean(mae_val)
print(avg_mae_val)


# 預測所有時間##################################################################################
allPredict = model.predict(all_X)
allPredict = np.squeeze(allPredict)
allPredict_flatten = allPredict.flatten()

# 創建包含預測結果的DataFrame
result_pred = pd.DataFrame({
    'lon': data_lon,  # 必須是與allPredict大小相符的NumPy陣列或列表
    'lat': data_lat,  # 同上
    'time': data_time,  # 同上
    'SM': allPredict_flatten  # 保存預測值
})
result_pred.sort_values(['time'], inplace=True)

# 按網格保存預測結果
grouped_pred = result_pred.groupby(['lon', 'lat'])
for name, group in tqdm(grouped_pred):
    directory = "E:/降尺度重採樣數據/0.25度/feature/模型預測結果/網格/CCI_all_4/"
    # 檢查資料夾是否存在，如果不存在則創建
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # 使用資料夾路徑和檔名格式化完整的檔案路徑
    filename = f"{directory}grid_{name[0]}_{name[1]}.csv"
    group.sort_values('time', inplace=True)  # 按時間排序
    group.to_csv(filename, index=False)  # 保存到CSV檔案
    
    
    
    
    
#全時段相關係數
corrcoefs_all = np.zeros(num_grids)
for i in range(num_grids):
    observed = all_Y[i, :]
    predicted = allPredict[i, :].flatten()  
    corrcoef, _ = pearsonr(observed, predicted)
    corrcoefs_all[i] = corrcoef
avg_corrcoef_all = np.mean(corrcoefs_all)
meaian_cor = np.median(corrcoefs_all)
print(meaian_cor)
print(avg_corrcoef_all)

#全時段Bias
biases_all = np.zeros(num_grids)
for i in range(num_grids):
    observed = all_Y[i, :]
    predicted = allPredict[i, :].flatten() 
    bias = np.mean(observed - predicted)
    biases_all[i] = bias
avg_bias_all = np.mean(biases_all)
meaian_bias = np.median(biases_all)
print(meaian_bias)
print(avg_bias_all)

#全時段RMSE
rmse_all = np.zeros(num_grids)
for i in range(num_grids):
    observed = all_Y[i, :]
    predicted = allPredict[i, :].flatten() 
    mse = np.mean((observed - predicted)**2)
    rmse_all[i] = np.sqrt(mse)
avg_rmse_all = np.mean(rmse_all)
meaian_rmse = np.median(rmse_all)
print(meaian_rmse)
print(avg_rmse_all)

# 放入5KM因子預測SM
# testX = np.load('E:/1125_各因子重新升尺度/5K/1201_SMAP_因子_大檔案_保存檔/0229_testX_plus_Ks.npy', allow_pickle=True)

# 給npy各行賦予列名
# column_names = ['lon', 'lat', 'time','ks','manning','carbon','clay','ph','elevation','slope','landuse','band4', 'band7' ,'ndvi','api','tsk','tslb','temp','sr','hum']

# testX = pd.DataFrame(testX, columns=column_names)
# testX['time'] = pd.to_datetime(testX['time'])  # 將時間欄位值轉為pandas格式
# testX.sort_values(['lon', 'lat', 'time'], inplace=True)


lon = testX['lon'].values
lat = testX['lat'].values
time = testX['time'].values


test_num_grids = len(testX.groupby(['lon', 'lat']))
testX.drop(columns=['lon', 'lat', 'time','hum'], inplace=True)


testX_values = testX.values


column_1 = testX_values[:, 0]

min_value = np.min(column_1)
max_value = np.max(column_1)

print(f"第一列的最小值是：{min_value}")
print(f"第一列的最大值是：{max_value}")


testX_values = testX_values.reshape(test_num_grids, time_steps, 1, 1, feature_num)
testX_values = testX_values.astype(np.float64)


testy = model.predict(testX_values)

# 删除冗余维度
testy = np.squeeze(testy)


result = pd.DataFrame({
    'lon': lon,
    'lat': lat,
    'time': time,
    'SM': testy.flatten() 
})
result.sort_values(['time', 'lon', 'lat'], inplace=True)

from tqdm import tqdm

grouped = result.groupby(['lon', 'lat'])

# 使用 tqdm 包裝你的迴圈，以便於展示進度條
for name, group in tqdm(grouped):
    # 確保保存檔案的資料夾存在，如果不存在則創建
    directory = 'E:/1125_各因子重新升尺度/3K/3K_SM_CCI/'
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # 格式化檔案名稱，使用網格的 'lon' 和 'lat' 作為CSV檔案名
    filename = f"{directory}grid_{name[0]}_{name[1]}.csv"
    
    # 根據時間對每個網格的數據進行排序
    group.sort_values('time', inplace=True)
    
    # 將排序後的數據寫入到CSV檔案中，不保存索引
    group.to_csv(filename, index=False)
    
    
    

from tqdm import tqdm

# 根據 'time' 欄創建一個以日期為index的group
grouped = result.groupby(['time'])

# 迴圈將每一天資料數據分開
for name, group in tqdm(grouped):  # 加入 tqdm 顯示進度條
    date = pd.to_datetime(name)
    filename = f"E:/降尺度重採樣數據/0.05度/0919/test_y_1011_case3_daily/day_{date.year}_{date.month}_{date.day}.csv"
    group.sort_values(['lon', 'lat'], inplace=True)  # 根據經緯度進行排序
    group.to_csv(filename, index=False)  # 將資料拆成每日分開寫入 CSV