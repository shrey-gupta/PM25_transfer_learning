########################## Header files #########################################################
# from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2 ##For STrAdaBoost.R2
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 ## For two-stage TrAdaBoost.R2

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

############################## UCI Italian dataset #######################################################
###################### 2 years and single terrain dataset is read #########################
aqi_df = pd.read_csv('AQI_datasets/UCI_AQI/AirQualityUCI.csv', sep=',', delimiter=";", decimal=",", index_col = None, header=0)
aqi_df.head(20)

def remove_outlier(col):
    aqi_df[col] = aqi_df.groupby('Date')[col].transform(lambda x: x.fillna(x.mean()))


####################### drop end rows with NaN values #####################################
aqi_df.dropna(how = 'all', inplace = True)

drop_unamed = ['Unnamed: 15', 'Unnamed: 16']
aqi_df = aqi_df.drop(drop_unamed, axis = 1)
aqi_df.columns

#################### Observing data statistics ############################################
aqi_df.describe()

################### Drop unwanted columns #################################################
drop_uw = ['Time', 'PT08_S1_CO', 'PT08_S2_NMHC', 'PT08_S3_NOx', 'NO2_GT', 'PT08_S4_NO2']
aqi_df = aqi_df.drop(drop_uw, axis = 1)

################# Replace the -200 values seen in the dataset with ########################
aqi_df.replace(to_replace = -200, value = np.NaN, inplace = True)

################ Replace the NaN values with the column mean ##############################
col_list = aqi_df.columns[1:]
for i in col_list:
    remove_outlier(i)

aqi_df.fillna(method ='ffill', inplace= True)
aqi_df.dropna(axis = 0)

############### Convert 'Date' column to datetime and then seperate out year and month into different columns ######
aqi_df.Date = pd.to_datetime(aqi_df.Date)
aqi_df['Year'] = aqi_df['Date'].dt.year
aqi_df['Month'] = aqi_df['Date'].dt.month
drop_date = ['Date']
aqi_df = aqi_df.drop(drop_date, axis = 1)
print(aqi_df)

################################################################################################################

###################### Split the dataset according to the year. #################################################
aqi_df_2004 = aqi_df[aqi_df['Year'] == 2004]
aqi_df_2005 = aqi_df[aqi_df['Year'] == 2005]

aqi_df_2004 = aqi_df_2004.reset_index(drop=True)
aqi_df_2005 = aqi_df_2005.reset_index(drop=True)


##################### Divide the dataframe into target and the predictors. ######################################
target_uci_col = ['PT08_S5_O3']
aqi_df_2004_target = aqi_df_2004[target_uci_col]
aqi_df_2004_target.columns = ['O3']

aqi_df_2005_target = aqi_df_2005[target_uci_col]
aqi_df_2005_target.columns = ['O3']

aqi_df_2004_predictors = aqi_df_2004.drop(target_uci_col, axis = 1)
aqi_df_2004_predictors.columns = ['CO', 'NMHC', 'C6H6', 'NOx', 'Temp', 'RH', 'AH', 'Year', 'Month']

aqi_df_2005_predictors = aqi_df_2005.drop(target_uci_col, axis = 1)
aqi_df_2005_predictors.columns = ['CO', 'NMHC', 'C6H6', 'NOx', 'Temp', 'RH', 'AH', 'Year', 'Month']

print(aqi_df_2004_predictors.shape)
print(aqi_df_2005_predictors.shape)

#################### Standardize the dataset #####################################################################
columns_uci = ['CO', 'NMHC', 'C6H6', 'NOx', 'Temp', 'RH', 'AH', 'Year', 'Month']
cols_to_norm = ['CO', 'NMHC', 'C6H6', 'NOx', 'Temp', 'RH', 'AH']

ss = StandardScaler()
aqi_df_2004_predictors[cols_to_norm] = ss.fit_transform(aqi_df_2004_predictors[cols_to_norm])
aqi_df_2005_predictors[cols_to_norm] = ss.fit_transform(aqi_df_2005_predictors[cols_to_norm])

# aqi_df_2004_predictors
# aqi_df_2005_predictors

#################### Splitting dataset into train, test and source. ###############################################
X_2005_train, X_2005_test, y_2005_train, y_2005_test = train_test_split(aqi_df_2005_predictors, aqi_df_2005_target, test_size = 0.96, random_state = 1)

X_2004 = aqi_df_2004_predictors
y_2004 = aqi_df_2004_target

# X_2005_train.shape

####################################### Prediction for TrAdaBoost.R2 ###############################################
predictionlist = []
r2scorelist = []
rmselist = []

print("TrAdaBoost.R2")

for x in range(0, 10):

    TF_train_X = pd.concat([X_2004, X_2005_train], sort= False)
    TF_train_y = pd.concat([y_2004, y_2005_train], sort= False)

    np_TF_train_X = TF_train_X.to_numpy()
    np_TF_train_y = TF_train_y.to_numpy()

    np_test_X = X_2005_test.to_numpy()
    np_test_y = y_2005_test.to_numpy()

    np_TF_train_y_list = np_TF_train_y.ravel()
    np_test_y_list = np_test_y.ravel()

    sample_size = [len(X_2004), len(X_2005_train)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    ################################################TrAdaBoost.R2############################################################################
    regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6), #xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.0001,max_depth = 6,n_estimators = 2000),
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

with open('AQI_datasets/UCI_AQI_Results/O3/TrAdaBoostOut_R2_std.txt', 'w') as filehandle1:
    for listitem in r2scorelist:
        filehandle1.write('%s\n' % listitem)

with open('AQI_datasets/UCI_AQI_Results/O3/TrAdaBoostOut_RMSE_std.txt', 'w') as filehandle2:
    for listitem in rmselist:
        filehandle2.write('%s\n' % listitem)

with open('AQI_datasets/UCI_AQI_Results/O3/TrAdaBoostOut_prediction_std.txt', 'w') as filehandle3:
    for listitem in predict_tradaboost:
        filehandle3.write('%s\n' % listitem)
