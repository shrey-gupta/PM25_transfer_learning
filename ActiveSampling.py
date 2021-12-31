from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

import pandas as pd
import sys
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, Flatten
from keras import optimizers, utils, initializers, regularizers
import keras.backend as K

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler #Importing the StandardScaler

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr
from math import sqrt

################# Geo plotting libraries
import geopandas as gdp
from matplotlib.colors import ListedColormap
import geoplot as glpt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor

from statistics import mean
from sklearn.cluster import KMeans
from scipy.spatial import distance


sns.set_style("darkgrid")

################################################################################################################
####################################################################US dataset #######################################3
#######################################################################################################################################

def remove_outlier(col):
    aqi_df[col] = aqi_df.groupby('Date')[col].transform(lambda x: x.fillna(x.mean()))




aqi_df = pd.read_csv('AQI_datasets/UCI_AQI/AirQualityUCI.csv', sep=',', delimiter=";",decimal=",")
aqi_df = aqi_df.drop(["Unnamed: 15","Unnamed: 16"], axis=1)
aqi_df.dropna(inplace=True)
aqi_df.set_index("Date", inplace=True)
aqi_df.index = pd.to_datetime(aqi_df.index)
type(aqi_df.index)
aqi_df['Time'] = pd.to_datetime(aqi_df['Time'],format= '%H.%M.%S').dt.hour
type(aqi_df['Time'][0])


# aqi_df['Date'] = pd.to_datetime(aqi_df['Date'])
# aqi_df_2004 = aqi_df[aqi_df['Date'].dt.year == 2004]
# aqi_df_2005 = aqi_df[aqi_df['Date'].dt.year == 2005]


aqi_df.drop('NMHC_GT', axis=1, inplace=True)
aqi_df.replace(to_replace= -200, value= np.NaN, inplace= True)
aqi_df.replace(to_replace= -200, value= np.NaN, inplace= True)

col_list = aqi_df.columns[1:]
for i in col_list:
    remove_outlier(i)

aqi_df.fillna(method='ffill', inplace= True)
aqi_df.dropna(axis = 0)
print(aqi_df.shape)

# col = aqi_df['PT08.S3(NOx)']
# AutoData_df = AutoData_df.sort_values(by=['PT08.S3(NOx)'])
# aqi_df = aqi_df.reset_index(drop=True, inplace=True)

aqi_df['YEAR'] = aqi_df.index.year

aqi_df_2004 = aqi_df[aqi_df['YEAR'] == 2004]
aqi_df_2005 = aqi_df[aqi_df['YEAR'] == 2005]

X_2004 = aqi_df_2004.drop(['NOx_GT','T','Time'], axis=1)
y_2004 = aqi_df_2004['NOx_GT']

X_2005 = aqi_df_2005.drop(['NOx_GT','T','Time'], axis=1)
y_2005 = aqi_df_2005['NOx_GT']

columns_uci = X_2004.columns
index_uci_2004 = X_2004.index
index_uci_2005 = X_2005.index

ss = StandardScaler()
X_2004_std = ss.fit_transform(X_2004)
X_2005_std = ss.fit_transform(X_2005)

X_2004 = pd.DataFrame(X_2004_std, columns = columns_uci)
X_2004 = X_2004.set_index(index_uci_2004)
X_2005 = pd.DataFrame(X_2005_std, columns = columns_uci)
X_2005 = X_2005.set_index(index_uci_2005)
# print(X_2004)
# print(X_2005)

X_2005_train, X_2005_test, y_2005_train, y_2005_test = train_test_split(X_2005, y_2005, test_size = 0.92, random_state=1)
# print("Train: ",X_2005_train.shape)
# print("Test: ", X_2005_test.shape)

X_source = X_2004
X_source['NOx_GT'] = y_2004

X_train = X_2005_train
X_train['NOx_GT'] = y_2005_train

X_test = X_2005_test
X_test['NOx_GT'] = y_2005_test

print(X_train.shape)
print(X_source.shape)
print(X_test.shape)
print(X_train.columns)

######################################## Seeding Technique ###################################################
kmeans = KMeans(n_clusters = 35, random_state=0).fit(X_source)

Alternate_df = X_train.copy()
idxlist = []
Alternate_df_np = Alternate_df.to_numpy()

new_df_list = []


for rowkm in kmeans.cluster_centers_:
    mindist = -99
    rowidx = 0
    idx = 0
    for row in Alternate_df_np:
        dst = distance.euclidean(row, rowkm)

        if(dst >= mindist):
            mindist = dst
            rowidx = idx
            rowval = row

        idx = idx + 1

    print("Row selected: ", rowidx) #Alternate_df.loc[rowidx,:]\
    print("Min. distance: ", mindist)
    print("Matrix shape: ", Alternate_df_np.shape)
    new_df_list.append(rowval)
    Alternate_df_np = np.delete(Alternate_df_np, rowidx, 0)
    idxlist.append(rowidx)


new_df = pd.DataFrame(np.vstack(new_df_list))
# print(new_df)
print(new_df.shape)
# print(new_df.dtypes)

##################################################### Phase 2: Seeding ################################################

alt_source_df = X_source.copy()
idxlist2 = []
alt_source_df_np = alt_source_df.to_numpy()

final_df_list = []

for row_nm in new_df_list:
    min_dist = -99
    row_idx = 0
    idx_val = 0
    for row_alt in alt_source_df_np:
        dst = distance.euclidean(row_alt, row_nm)
        if(dst >= mindist):
            min_dist = dst
            row_idx = idx_val
            row_val = row_alt

        idx_val = idx_val + 1

    print("Row selected: ", row_idx) #Alternate_df.loc[rowidx,:]\
    print("Min. distance: ", min_dist)
    print("Matrix shape: ", alt_source_df_np.shape)
    final_df_list.append(row_val)
    alt_source_df_np = np.delete(alt_source_df_np, row_idx, 0)
    idxlist2.append(row_idx)


final_df = pd.DataFrame(np.vstack(final_df_list), columns= X_source.columns)
print(final_df)
print(final_df.shape)

print("Shape before :",X_source.shape)
X_source = pd.concat([X_source, final_df])
X_source = X_source.drop_duplicates(keep=False)
print("Shape after :",X_source.shape)

#################################################################################################################################

X_train = pd.concat([X_train, final_df], ignore_index=True)

X_train.to_csv('ActiveSampling/temp_seeded_train.csv',index=False)
X_source.to_csv('ActiveSampling/temp_seeded_source.csv',index=False)

# X_train.to_csv('ActiveSampling/UCI_NO2_activesampling_train.csv',index=False)
# X_source.to_csv('ActiveSampling/UCI_NO2_activesampling_source.csv',index=False)
# X_test.to_csv('ActiveSampling/UCI_NO2_activesampling_test.csv',index=False)

############################## Finding best instances from the source dataset ##########################################################
X_train = pd.read_csv('ActiveSampling/temp_seeded_train.csv')
X_source = pd.read_csv('ActiveSampling/temp_seeded_source.csv')

X_source['ManDis'] = ""

train_df_mean = []
prow = X_train.mean()
# train_df_mean = [prow.aod_value, prow.fire, prow.narr_dpt, prow.narr_vis, prow.narr_pres2m, prow.narr_pres30m, prow.narr_tmp30m, prow.narr_hpbl, prow.nldas_pevapsfc, prow.nldas_dlwrfsfc_1,
# prow.nldas_dswrfsfc_2, prow.nldas_cape, prow.nldas_pressfc, prow.nldas_tmp2m, prow.nldas_rh2m, prow.nldas_ugrd10m, prow.nldas_vgrd10m, prow.forest_cover, prow.elev, prow.emissi11_pm25, prow.emissi11_pm10, prow.high,
# prow.limi, prow.local, prow.iss, prow.pd, prow.narr_pres10m, prow.narr_tmp1815mb]
train_df_mean = [prow.CO_GT, prow.PT08_S1_CO, prow.C6H6_GT, prow.PT08_S2_NMHC, prow.PT08_S3_NOx, prow.NO2_GT, prow.PT08_S4_NO2, prow.PT08_S5_O3, prow.RH, prow.AH, prow.NOx_GT]

rowidx = 0
for row in X_source.itertuples():
    # row_list =[row.aod_value, row.fire, row.narr_dpt, row.narr_vis, row.narr_pres2m, row.narr_pres30m, row.narr_tmp30m, row.narr_hpbl, row.nldas_pevapsfc, row.nldas_dlwrfsfc_1,
    # row.nldas_dswrfsfc_2, row.nldas_cape, row.nldas_pressfc, row.nldas_tmp2m, row.nldas_rh2m, row.nldas_ugrd10m, row.nldas_vgrd10m, row.forest_cover, row.elev, row.emissi11_pm25, row.emissi11_pm10, row.high,
    # row.limi, row.local, row.iss, row.pd, row.narr_pres10m, row.narr_tmp1815mb]

    row_list =[row.CO_GT, row.PT08_S1_CO, row.C6H6_GT, row.PT08_S2_NMHC, row.PT08_S3_NOx, row.NO2_GT, row.PT08_S4_NO2, row.PT08_S5_O3, row.RH, row.AH, row.NOx_GT]

    man_dis = 0
    for i in range(0, len(row_list)):
        tempval = train_df_mean[i] - row_list[i]
        man_dis = man_dis + abs(tempval)

    X_source.loc[rowidx,'ManDis'] = man_dis
    # print(Source_df.loc[rowidx,"ManDis"])
    rowidx = rowidx + 1


X_source_rest = X_source.sort_values('ManDis')
X_source = X_source.head(6500)
X_source_rest = X_source.iloc[6500:]
X_source_rest = X_source_rest.drop(['ManDis'], axis =1)
X_source = X_source.drop(['ManDis'], axis =1)

X_train.to_csv('ActiveSampling/UCI_NOx_activesampling_train.csv',index=False)
X_source.to_csv('ActiveSampling/UCI_NOx_activesampling_source.csv',index=False)
X_test.to_csv('ActiveSampling/UCI_NOx_activesampling_test.csv',index=False)

print("Done !!!!!")
