########################## Header files ########################################
################################################################################
###### This code is used for UCI italian dataset: STrAdaBoost.R2 ################
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

##################################################### Incorporating active sampling results #####################################
target_column = ['O3']

X_train = pd.read_csv('ActiveSampling/UCI_O3_activesampling_train.csv')
X_test = pd.read_csv('ActiveSampling/UCI_O3_activesampling_test.csv')
X_source = pd.read_csv('ActiveSampling/UCI_O3_activesampling_source.csv')

print(X_train.shape)
print(X_test.shape)
print(X_source.shape)

y_2004 = X_source[target_column]
X_2004 = X_source.drop(target_column, axis = 1)

y_2005_train = X_train[target_column]
X_2005_train = X_train.drop(target_column, axis = 1)

y_2005_test = X_test[target_column]
X_2005_test = X_test.drop(target_column, axis = 1)

################################# Prediction for STrAdaBoost ##################################################################
predictionlist = []
r2scorelist = []
rmselist = []

print("STrAdaBoost.R2")

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

    ################################################ STrAdaBoost.R2 ############################################################################
    regr_1 = STrAdaBoostR2(DecisionTreeRegressor(max_depth=6), #xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.0001,max_depth = 6,n_estimators = 2000),
                          n_estimators = n_estimators, sample_size = sample_size,
                          steps = steps, fold = fold,
                          random_state = random_state)

    regr_1.fit(np_TF_train_X, np_TF_train_y_list)
    y_pred1 = regr_1.predict(np_test_X)
    predictionlist.append(y_pred1)

    mse_twostageboost = np.sqrt(mean_squared_error(np_test_y_list, y_pred1))
    print("RMSE of STrAdaboostR2:", mse_twostageboost)
    rmselist.append(mse_twostageboost)

    r2_score_twostageboost_values = pearsonr(np_test_y_list, y_pred1)
    r2_score_twostageboost = (r2_score_twostageboost_values[0])**2
    print("R^2 of STrAdaboostR2:", r2_score_twostageboost)
    r2scorelist.append(r2_score_twostageboost)

predict_stradaboost = np.mean(predictionlist, axis=0)
list_orginal_stradaboost = np_test_y_list

print("mean RMSE of STrAdaboostR2:", mean(rmselist))
print("mean R^2 of STrAdaboostR2:", mean(r2scorelist))


with open('AQI_datasets/UCI_AQI_Results/O3/STrAdaBoostOut_R2_2.txt', 'w') as filehandle1:
    for listitem in r2scorelist:
        filehandle1.write('%s\n' % listitem)

with open('AQI_datasets/UCI_AQI_Results/O3/STrAdaBoostOut_RMSE_2.txt', 'w') as filehandle2:
    for listitem in rmselist:
        filehandle2.write('%s\n' % listitem)

with open('AQI_datasets/UCI_AQI_Results/O3/STrAdaBoostOut_prediction_2.txt', 'w') as filehandle3:
    for listitem in predict_stradaboost:
        filehandle3.write('%s\n' % listitem)



plt.scatter(predict_adaboost, list_orginal_adaboost,  c ="red", alpha=0.6)
plt.scatter(predict_tradaboost, list_orginal_tradaboost,  c ="blue", alpha=0.4)
plt.scatter(predict_stradaboost, list_orginal_stradaboost,  c ="green", alpha=0.2)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Predicted vs Actual')
plt.show()
