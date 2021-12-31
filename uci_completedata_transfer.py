################################################################################
###### This code is used for UCI regression datasets ###########################

######################Library import ###########################################
from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2 ###Import this for STrAdaBoost.R2
# from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 ###Import this for two-stage TrAdaBoost.R2

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
import geopandas as gdp
from matplotlib.colors import ListedColormap
# import geoplot as glpt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold
import matplotlib.lines as mlines

####################Read datasets from the local directory #########################
####################Provide the file path on your sytem to read the csv files ######
##### Reading concrete dataset. Analysis ==> 'Cement' attribute found to be correlated at 0.4 ==> 100 instances chosen for variance sampling.
# ConcreteData_df = pd.read_excel('UCI_regression/Concrete_Data.xls')

##### Reading housing dataset. Analysis ==> 'nox' attribute found to be correlated at 0.4 ==> 70 instances chosen for variance sampling.
# HousingData_df = pd.read_csv('UCI_regression/BostonHousing/BostonHousing.csv')

##### Reading auto dataset. Analysis ==> 'horsepower' attribute found to be correlated at 0.4 ==> 40 instances chosen for variance sampling.
dropcol_initial = ['name']
AutoData_df = pd.read_csv('UCI_regression/MPG/Auto.csv')
AutoData_df = AutoData_df.drop(dropcol_initial, axis = 1)
# print("The shape of the Input data is: ", AutoData_df.shape)

###################Preprocessing Data ##########################################
## Finding the correlation first
# print("The correlation matrix is: ")
# print(ConcreteData_df.corr())
# col = ConcreteData_df['Cement']
# ConcreteData_df = ConcreteData_df.sort_values(by=['Cement'])

# print(AutoData_df[(col <= 80)].shape)
# print(AutoData_df[(col > 80) & (col <= 110)].shape)
# print(AutoData_df[(col > 110)].shape)

# col = HousingData_df['nox']
# print(HousingData_df[(col < 0.450)].shape)
# print(HousingData_df[(col > 0.450) & (col < 0.600)].shape)

#####################Splitting dataset into subsets ############################
# drop_col = ['Cement']
drop_col = ['horsepower']
# drop_col = ['nox']

# Train_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] > 225) & (ConcreteData_df['Cement'] <= 350)]
# Train_df = HousingData_df.loc[(HousingData_df['nox'] > 0.475) & (HousingData_df['nox'] <= 0.600)]
Train_df = AutoData_df.loc[(AutoData_df['horsepower'] > 80) & (AutoData_df['horsepower'] <= 110)]
Train_df = Train_df.drop(drop_col, axis = 1)
Train_df = Train_df.reset_index(drop=True)
# print(Train_df.shape)

# Source_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] <= 225)]
# Source_df = HousingData_df.loc[(HousingData_df['nox'] <= 0.475)]
Source_df = AutoData_df.loc[(AutoData_df['horsepower'] <= 80)]
Source_df = Source_df.drop(drop_col, axis = 1)
Source_df = Source_df.reset_index(drop=True)
# print(Source_df.shape)

# Test_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] > 350)]
# Test_df = HousingData_df.loc[(HousingData_df['nox'] > 0.600)]
Test_df = AutoData_df.loc[(AutoData_df['horsepower'] > 110)]
Test_df = Test_df.drop(drop_col, axis = 1)
Test_df = Test_df.reset_index(drop=True)
# print(Test_df.shape)

######################### Finding best instances from the source dataset #######
#####################This block of code is removed for TrAdaBoost.R2 ###########
Source_df["ManDis"] = ""

train_df_mean = []
prow = Train_df.mean()
# train_df_mean = [prow.BlastFurnaceSlag, prow.FlyAsh, prow.Water, prow.Superplasticizer, prow.CoarseAggregate, prow.FineAggregate, prow.Age, prow.ConcreteCompressiveStrength]
# train_df_mean = [prow.crim, prow.zn, prow.indus, prow.chas, prow.rm, prow.age, prow.dis, prow.rad, prow.tax, prow.ptratio, prow.b, prow.lstat, prow.medv]
train_df_mean = [prow.mpg, prow.cylinders, prow.displacement, prow.weight, prow.acceleration, prow.year, prow.origin]

rowidx = 0
for row in Source_df.itertuples():
    # row_list =[row.BlastFurnaceSlag, row.FlyAsh, row.Water, row.Superplasticizer, row.CoarseAggregate, row.FineAggregate, row.Age, row.ConcreteCompressiveStrength]
    # row_list =[row.crim, row.zn, row.indus, row.chas, row.rm, row.age, row.dis, row.rad, row.tax, row.ptratio, row.b, row.lstat, row.medv]
    row_list =[row.mpg, row.cylinders, row.displacement, row.weight, row.acceleration, row.year, row.origin]

    man_dis = 0
    for i in range(0, len(row_list)):
        tempval = train_df_mean[i] - row_list[i]
        man_dis = man_dis + abs(tempval)

    Source_df.loc[rowidx,"ManDis"] = man_dis
    # print(Source_df.loc[rowidx,"ManDis"])
    rowidx = rowidx + 1

# print(Source_df)
Source_df = Source_df.sort_values('ManDis')
Train_source_df = Source_df.head(50) ## For housing 70 was taken, For auto 40 was taken, For concrete 100 was taken
Source_df = Source_df.iloc[50:]
Source_df = Source_df.drop(['ManDis'], axis =1)
Train_source_df = Train_source_df.drop(['ManDis'], axis =1)

print(Train_source_df.shape)
print(Source_df.shape)
################################################################################

# target_column = ['ConcreteCompressiveStrength']
# target_column = ['medv']
target_column = ['mpg']

Train_df = pd.concat([Train_df, Train_source_df], ignore_index=True) ### This line is used only for STrAdaBoost.R2 and not for TrAdaBoost.R2

Train_df_y = Train_df[target_column]
Train_df_X = Train_df.drop(target_column, axis = 1)

Test_df_y = Test_df[target_column]

Test_df_X = Test_df.drop(target_column, axis = 1)
Source_df_y = Source_df[target_column]
Source_df_X = Source_df.drop(target_column, axis = 1)

print(Train_df_X.shape)
print(Test_df_X.shape)
print(Source_df_X.shape)

## Merging the datasets
X_df = pd.concat([Train_df_X, Test_df_X], ignore_index=True)
y_df = pd.concat([Train_df_y, Test_df_y], ignore_index=True)

np_train_X = X_df.to_numpy()
np_train_y = y_df.to_numpy()

np_source_X = Source_df_X.to_numpy()
np_source_y = Source_df_y.to_numpy()

np_train_y_list = np_train_y.ravel()

# sample_size = [len(Target_df_X), len(Source_df_X)]
n_estimators = 100
steps = 30
fold = 10
random_state = np.random.RandomState(1)

################################################STrAdaBoost.R2, TwoStageAdaBoostR2#####

kf = KFold(n_splits = 10) ## Create no. of CV Folds
error = []

for train_idx, test_idx in kf.split(X_df):
    X_train, X_test = np_train_X[train_idx], np_train_X[test_idx]
    y_train, y_test = np_train_y[train_idx], np_train_y[test_idx]

    sample_size = [len(Source_df_X), len(X_train)]

    X_train_new = np.concatenate((np_source_X, X_train))
    y_train_new = np.concatenate((np_source_y, y_train))

    np_y_train_new_list = y_train_new.ravel()

    regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                      n_estimators = n_estimators, sample_size = sample_size,
                      steps = steps, fold = fold,
                      random_state = random_state)
    regr_1.fit(X_train_new, np_y_train_new_list)
    y_pred2 = regr_1.predict(X_test)
    mse_adaboost = sqrt(mean_squared_error(y_test, y_pred2))
    print("RMSE of updated TrAdaboost.R2:", mse_adaboost)
    error.append(mse_adaboost)

print("Mean RMSE: ", sum(error)/len(error))

################## This is to plot the RMS Error and R2 Errors for the regression datasets###############
############################## Concrete #################################################################
# Stradaboostr2_vals_c1 = [13.4827, 11.4268, 9.8128, 9.4807, 9.3008]
# Stradaboostr2_vals_c2 = [11.0523, 9.7095, 9.0558, 8.2823, 8.4017]
# Stradaboostr2_vals_c3 = [9.8161, 9.2209, 8.5170, 8.2861, 8.2940]
#
# tradaboostr2_vals_c1 = [13.1574, 10.9064, 10.2261, 9.6316, 9.9406]
# tradaboostr2_vals_c2 = [12.5787, 11.9236, 10.2411, 9.1687, 8.3575]
# tradaboostr2_vals_c3 = [11.3698, 10.9281, 10.0875, 9.7463, 10.0507]

######################## Housing #########################################################################
Stradaboostr2_vals_c1 = [5.1471, 4.6071, 3.9612, 4.0192, 3.5896]
Stradaboostr2_vals_c2 = [3.9085, 3.5184, 3.4642, 3.3926, 3.1932]
Stradaboostr2_vals_c3 = [5.4579, 5.4087, 4.4772, 4.1595, 4.3258]

tradaboostr2_vals_c1 = [6.2413, 4.3046, 4.1392, 3.7370, 3.6625]
tradaboostr2_vals_c2 = [3.5426, 3.3584, 3.4409, 3.2106, 3.3889]
tradaboostr2_vals_c3 = [5.8373, 5.1851, 4.9193, 4.8286, 4.5810]

######################################### Auto MPG #########################################################
# Stradaboostr2_vals_c1 = [5.0310, 3.2700, 3.1405, 2.9630, 2.8862]
# Stradaboostr2_vals_c2 = [3.6326, 3.2661, 3.1774, 3.2179, 3.1813]
# Stradaboostr2_vals_c3 = [3.9503, 2.9288, 2.8186, 2.7323, 2.7659]
#
# tradaboostr2_vals_c1 = [4.2658, 3.3920, 2.9866, 2.9890, 2.7403]
# tradaboostr2_vals_c2 = [3.6795, 3.6379, 3.1619, 3.1428, 3.0614]
# tradaboostr2_vals_c3 = [3.9136, 3.0207, 2.7356, 2.6930, 2.6273]

#############################################################################################################

xlim_list = [2, 4, 6, 8, 10]
box = mlines.Line2D([], [], color = 'tomato', marker='s', linestyle='None',
                          markersize = 20, label='STrAdaBoost.R2')
circle = mlines.Line2D([], [], color = 'navy', marker='o', linestyle='None',
                          markersize = 20, label='TrAdaBoost.R2*')


line1 = mlines.Line2D([], [], color = 'black', marker='None',linestyle='-',
                          markersize = 30, label='Case 1', linewidth = 6.0)
line2 = mlines.Line2D([], [], color = 'black', marker='None', linestyle='--',
                          markersize = 30, label='Case 2', linewidth = 6.0)
line3 = mlines.Line2D([], [], color = 'black', marker='None', linestyle=':',
                          markersize = 30, label='Case 3', linewidth = 6.0)

#'None'

plt.plot(xlim_list, Stradaboostr2_vals_c1, '-sk', color = 'tomato', linewidth = 4.0, markersize = 20)
plt.plot(xlim_list, Stradaboostr2_vals_c2, '--sk', color = 'tomato', linewidth = 4.0, markersize = 20)
plt.plot(xlim_list, Stradaboostr2_vals_c3, ':sk', color = 'tomato', linewidth = 4.0, markersize = 20)

plt.plot(xlim_list, tradaboostr2_vals_c1, '-ok', color = 'navy', linewidth = 4.0, markersize = 20)
plt.plot(xlim_list, tradaboostr2_vals_c2, '--ok', color = 'navy', linewidth = 4.0, markersize = 20)
plt.plot(xlim_list, tradaboostr2_vals_c3, ':ok', color = 'navy', linewidth = 4.0, markersize = 20)

# plt.plot(Adaboost_r2value, color = 'm')
# plt.xlim()
plt.xticks(np.arange(2, 12, 2.0), fontsize = 40)
# plt.yticks(np.arange(7, 15, 1.0), fontsize = 40)
plt.yticks(np.arange(3, 7, 1.0), fontsize = 40)
# plt.yticks(np.arange(2, 6, 1.0), fontsize = 40)
plt.ylabel('RMS Error', fontsize = 40)
plt.xlabel('% Target', fontsize = 40)

plt.legend(handles= [box, circle, line1, line2, line3], fontsize = 30)

# plt.title('UCI Auto MPG dataset', fontsize = 28)
plt.show()

# ###############################################Trying AdaBoost for regression####################################################################
#
# # kf = KFold(n_splits = 2) ## Create no. of CV Folds
# # error = []
# #
# # for train_idx, test_idx in kf.split(X_df):
# #     X_train, X_test = np_train_X[train_idx], np_train_X[test_idx]
# #     y_train, y_test = np_train_y[train_idx], np_train_y[test_idx]
# #
# #     regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators = n_estimators)
# #     regr_2.fit(X_train, y_train)
# #     y_pred2 = regr_2.predict(X_test)
# #
# #     mse_adaboost = sqrt(mean_squared_error(y_test, y_pred2))
# #     print("RMSE of regular AdaboostR2:", mse_adaboost)
# #     error.append(mse_adaboost)
# #
# # print("Mean RMSE: ", sum(error)/len(error))
# # r2_score_adaboost = r2_score(test_peru_df_y, y_pred2)
# # r2_score_adaboost_values = pearsonr(np_TF_test_y_list, y_pred2)
# # r2_score_adaboost = (r2_score_adaboost_values[0])**2
# # print("R^2 of AdaboostR2:", r2_score_adaboost)
