################################################################################
###### This code is used for UCI beijing dataset: STrAdaBoost.R2 ###############

################################# Header files #################################
from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2 ## For two-stage TrAdaBoost.R2

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

#Geo plotting libraries
#import geopandas as gdp
#from matplotlib.colors import ListedColormap
#import geoplot as glpt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold
import matplotlib.lines as mlines
import folium
import glob

from statistics import mean

print("Done uploading repositories")

################# UCI Beijing dataset (2013 - 2017) #####################################################################
############### Spacio-temporal dataset. (multi-year and multi-terrain) #################################################

path_beijing = r'AQI_datasets/Beijing_AQI/PRSA_Data_20130301-20170228/' ## Path for all the files
allFiles = glob.glob(path_beijing + "/*.csv")
beijing_aqi_df = pd.DataFrame()
list_beijing = []
for file_ in allFiles:
    temp_df = pd.read_csv(file_, index_col = None, header=0)
    list_beijing.append(temp_df)
beijing_aqi_df = pd.concat(list_beijing)

cols = ['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station']

beijing_aqi_df = beijing_aqi_df[cols]

drop_index = ['No']
beijing_aqi_df = beijing_aqi_df.drop(drop_index, axis=1)
beijing_aqi_df = beijing_aqi_df.sort_values(['station', 'year'])
# beijing_aqi_df.head(20)
# beijing_aqi_df.year.value_counts()
# beijing_aqi_df.wd.value_counts()
# beijing_aqi_df.shape
# beijing_aqi_df['station'].nunique()

######################## Seperate the dataset into predictors and target variable. #########################################
predictors = ['year', 'month', 'SO2', 'NO2', 'CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station', 'O3']
beijing_predictors_df = beijing_aqi_df[predictors]
wd_codes = {'N':1, 'E': 2, 'W': 3, 'S': 4, 'NE': 5, 'NW': 6, 'SE': 7, 'SW': 8, 'NNE': 9, 'NNW': 10, 'SSE': 11,
            'SSW': 12, 'WNW': 13, 'WSW': 14, 'ENE': 15, 'ESE': 16 }
beijing_predictors_df.replace(wd_codes, inplace=True)
print(beijing_predictors_df.wd.value_counts())

############# Replacing missing values ####################################################################################
beijing_predictors_df = beijing_predictors_df.dropna(axis=0) # , subset=['O3'])

###### percentage of missing values in each column ########################################################################
round(beijing_predictors_df.isna().sum()/len(beijing_predictors_df.index), 2)*100
corrMatrix_beijing = beijing_predictors_df.corr()
corrMatrix_beijing["O3"].sort_values(ascending=False)

### Drop stations (dropping stations at this stage so more combination of splits can be made later) #######################
drop_stations = ['station']
beijing_predictors_df = beijing_predictors_df.drop(drop_stations, axis=1)

######################################## Splitting the dataset by the year #################################################
beijing_predictors_df.year.value_counts()
beijing_predictors_df_target = beijing_predictors_df[beijing_predictors_df['year'].isin([2014, 2015])]
beijing_predictors_df_source = beijing_predictors_df[beijing_predictors_df['year'].isin([2016])]

print(beijing_predictors_df_source.shape)
print(beijing_predictors_df_target.shape)

#################################### Standardize the dataset. ###############################################################

cols_to_norm = ['SO2', 'NO2', 'CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

ss = StandardScaler()
beijing_predictors_df_target[cols_to_norm] = ss.fit_transform(beijing_predictors_df_target[cols_to_norm])
beijing_predictors_df_source[cols_to_norm] = ss.fit_transform(beijing_predictors_df_source[cols_to_norm])

################################# Splitting the dataset into train and test set ###############################################
target_beijing_col = ['O3']
beijing_predictors_df_target_Y = beijing_predictors_df_target[target_beijing_col]
beijing_predictors_df_target_X = beijing_predictors_df_target.drop(target_beijing_col, axis =1)

beijing_predictors_df_source_Y = beijing_predictors_df_source[target_beijing_col]
beijing_predictors_df_source_X = beijing_predictors_df_source.drop(target_beijing_col, axis =1)


X_train, X_test, y_train, y_test = train_test_split(beijing_predictors_df_target_X, beijing_predictors_df_target_Y, test_size = 0.999, random_state = 1)

X_source = beijing_predictors_df_source_X
y_source = beijing_predictors_df_source_Y

################################### Prediction for TrAdaBoost.R2 ##############################################################
predictionlist = []
r2scorelist = []
rmselist = []

print("TrAdaBoost.R2")

for x in range(0, 10):

    TF_train_X = pd.concat([X_source, X_train], sort= False)
    TF_train_y = pd.concat([y_source, y_train], sort= False)

    np_TF_train_X = TF_train_X.to_numpy()
    np_TF_train_y = TF_train_y.to_numpy()

    np_test_X = X_test.to_numpy()
    np_test_y = y_test.to_numpy()

    np_TF_train_y_list = np_TF_train_y.ravel()
    np_test_y_list = np_test_y.ravel()

    sample_size = [len(X_source), len(X_train)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    ################################################TrAdaBoost.R2################
    regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                          n_estimators = n_estimators, sample_size = sample_size,
                          steps = steps, fold = fold,
                          random_state = random_state)

    regr_1.fit(np_TF_train_X, np_TF_train_y_list)
    y_pred1 = regr_1.predict(np_test_X)
    predictionlist.append(y_pred1)

    mse_twostageboost = np.sqrt(mean_squared_error(np_test_y_list, y_pred1))
    print("RMSE of TrAdaboostR2:", mse_twostageboost)
    rmselist.append(mse_twostageboost)

    r2_score_twostageboost_values = pearsonr(np_test_y_list, y_pred1)
    r2_score_twostageboost = (r2_score_twostageboost_values[0])**2
    print("R^2 of TrAdaboostR2:", r2_score_twostageboost)
    r2scorelist.append(r2_score_twostageboost)

predict_tradaboost = np.mean(predictionlist, axis=0)
list_orginal_tradaboost = np_test_y_list

print("mean RMSE of TrAdaboostR2:", mean(rmselist))
print("mean R^2 of TrAdaboostR2:", mean(r2scorelist))

# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('TrAdaBoost.R2 Predicted vs Actual')
# # plt.savefig("AQI_datasets/UCI_AQI_Results/NOx/AdaBoostR2_Transfer.png")
# plt.show()

with open('AQI_datasets/Beijing_AQI_Results/O3/TrAdaBoostOut_R2_std_999.txt', 'w') as filehandle1:
    for listitem in r2scorelist:
        filehandle1.write('%s\n' % listitem)

with open('AQI_datasets/Beijing_AQI_Results/O3/TrAdaBoostOut_RMSE_std_999.txt', 'w') as filehandle2:
    for listitem in rmselist:
        filehandle2.write('%s\n' % listitem)

with open('AQI_datasets/Beijing_AQI_Results/O3/TrAdaBoostOut_prediction_std_999.txt', 'w') as filehandle3:
    for listitem in predict_tradaboost:
        filehandle3.write('%s\n' % listitem)

###########################################################################################################################################
