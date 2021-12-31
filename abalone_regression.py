# from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2 ##STrAdaBoost.R2
# from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 ##two-stage TrAdaBoost.R2

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

import statistics

from scipy.stats import rv_continuous
from scipy.stats import *

from statistics import mean
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.model_selection import KFold


######### Instance Transfer repositories ####################
from adapt.instance_based import TwoStageTrAdaBoostR2

print("Repositories uploaded!!")

from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2
from sklearn.model_selection import GridSearchCV
from adapt.instance_based import KMM

print("Second Upload Completed!!")

################################### Abalone ################################################################
#### range: [0.0 - 1.130]
#### Mid of correlation variable: Whole_weight
#### [0, 0.12] [0.12, 0.15], [0.15, 1.130]
#### Target variable: Rings
#################################################################################################################################
# target_var_abalone = ['Rings']

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame) #"df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

colnames_abalone = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
abaloneData_df = pd.read_csv('UCI_regression/Abalone/abalone.data', header = None, names = colnames_abalone)

gender = {'M': 1,'F': 2, 'I': 3}
abaloneData_df.Sex = [gender[item] for item in abaloneData_df.Sex]

print("Abalone Data")
print(abaloneData_df.shape)

ss = StandardScaler()

drop_col_abalone = ['Whole_weight']

abalone_train_df = abaloneData_df.loc[(abaloneData_df['Whole_weight'] > 0.500) & (abaloneData_df['Whole_weight'] <= 1.000)]
abalone_train_df = abalone_train_df.drop(drop_col_abalone, axis = 1)
abalone_train_df = abalone_train_df.reset_index(drop=True)
abalone_train_df = clean_dataset(abalone_train_df)
abalone_train_df = abalone_train_df.reset_index(drop=True)
print("Target Set: ",abalone_train_df.shape)


abalone_source1_df = abaloneData_df.loc[(abaloneData_df['Whole_weight'] > 1.000)]
abalone_source1_df = abalone_source1_df.drop(drop_col_abalone, axis = 1)
abalone_source1_df = abalone_source1_df.reset_index(drop=True)
abalone_source1_df = clean_dataset(abalone_source1_df)
abalone_source1_df = abalone_source1_df.reset_index(drop=True)
print("Source Set 1: ",abalone_source1_df.shape)

abalone_source2_df = abaloneData_df.loc[(abaloneData_df['Whole_weight'] <= 0.500)]
abalone_source2_df = abalone_source2_df.drop(drop_col_abalone, axis = 1)
abalone_source2_df = abalone_source2_df.reset_index(drop=True)
abalone_source2_df = clean_dataset(abalone_source2_df)
abalone_source2_df = abalone_source2_df.reset_index(drop=True)
print("Source Set 2: ",abalone_source2_df.shape)

abalone_source_df = pd.concat([abalone_source1_df, abalone_source2_df], ignore_index = True)
abalone_source_df = abalone_source_df.sample(frac = 0.5, replace = True, random_state = 1)

print("Final Source Set: ",abalone_source_df.shape)


#################### Splitting into features and target ####################
target_column_abalone = ['Rings']

abalone_train_df_y = abalone_train_df[target_column_abalone]
abalone_train_df_X = abalone_train_df.drop(target_column_abalone, axis = 1)
abalone_cols = abalone_train_df_X.columns
abalone_train_df_X[abalone_cols] = ss.fit_transform(abalone_train_df_X[abalone_cols])

abalone_source_df_y = abalone_source_df[target_column_abalone]
abalone_source_df_X = abalone_source_df.drop(target_column_abalone, axis = 1)
abalone_cols = abalone_source_df_X.columns
abalone_source_df_X[abalone_cols] = ss.fit_transform(abalone_source_df_X[abalone_cols])

print("---------------------------")

########################### Transfer Learning abalone #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}


print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_abalone = []
rmselist_AdaTL_abalone = []

r2scorelist_Ada_abalone = []
rmselist_Ada_abalone = []

r2scorelist_KMM_abalone = []
rmselist_KMM_abalone = []

r2scorelist_GBRTL_abalone = []
rmselist_GBRTL_abalone = []

r2scorelist_GBR_abalone = []
rmselist_GBR_abalone = []

r2scorelist_TwoTrAda_abalone = []
rmselist_TwoTrAda_abalone = []

r2scorelist_stradaboost_abalone = []
rmselist_stradaboost_abalone = []


kfold = KFold(n_splits = 20, random_state = 42, shuffle=False)

for train_ix, test_ix in kfold.split(abalone_train_df_X):
    ############### get data ###############
    abalone_test_df_X, abalone_tgt_df_X  = abalone_train_df_X.iloc[train_ix], abalone_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    abalone_test_df_y, abalone_tgt_df_y  = abalone_train_df_y.iloc[train_ix], abalone_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(abalone_tgt_df_X.shape, abalone_test_df_X.shape)

    ############### Merging the datasets ##########################################
    abalone_X_df = pd.concat([abalone_tgt_df_X, abalone_source_df_X], ignore_index=True)
    abalone_y_df = pd.concat([abalone_tgt_df_y, abalone_source_df_y], ignore_index=True)

    abalone_np_train_X = abalone_X_df.to_numpy()
    abalone_np_train_y = abalone_y_df.to_numpy()

    abalone_np_test_X = abalone_test_df_X.to_numpy()
    abalone_np_test_y = abalone_test_df_y.to_numpy()

    abalone_np_train_y_list = abalone_np_train_y.ravel()
    abalone_np_test_y_list = abalone_np_test_y.ravel()

    src_size_abalone = len(abalone_source_df_y)
    tgt_size_abalone = len(abalone_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_abalone - 1), step=1)
    tgt_idx = np.arange(start=src_size_abalone, stop=((src_size_abalone + tgt_size_abalone)-1), step=1)



    ################### AdaBoost Tl ###################
    model_AdaTL_abalone = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_AdaTL_abalone.fit(abalone_np_train_X, abalone_np_train_y_list)

    y_pred_AdaTL_abalone = model_AdaTL_abalone.predict(abalone_np_test_X)

    mse_AdaTL_abalone = sqrt(mean_squared_error(abalone_np_test_y, y_pred_AdaTL_abalone))
    rmselist_AdaTL_abalone.append(mse_AdaTL_abalone)

    r2_score_AdaTL_abalone = pearsonr(abalone_np_test_y_list, y_pred_AdaTL_abalone)
    r2_score_AdaTL_abalone = (r2_score_AdaTL_abalone[0])**2
    r2scorelist_AdaTL_abalone.append(r2_score_AdaTL_abalone)


    ################### AdaBoost ###################
    model_Ada_abalone = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_Ada_abalone.fit(abalone_tgt_df_X, abalone_tgt_df_y)

    y_pred_Ada_abalone = model_Ada_abalone.predict(abalone_np_test_X)

    mse_Ada_abalone = sqrt(mean_squared_error(abalone_np_test_y, y_pred_Ada_abalone))
    rmselist_Ada_abalone.append(mse_Ada_abalone)

    r2_score_Ada_abalone = pearsonr(abalone_np_test_y_list, y_pred_Ada_abalone)
    r2_score_Ada_abalone = (r2_score_Ada_abalone[0])**2
    r2scorelist_Ada_abalone.append(r2_score_Ada_abalone)


   ################### KMM ###################
    model_KMM_abalone = KMM(get_estimator = get_estimator)
    model_KMM_abalone.fit(abalone_np_train_X, abalone_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_abalone = model_KMM_abalone.predict(abalone_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_abalone = sqrt(mean_squared_error(abalone_np_test_y, y_pred_KMM_abalone))
    rmselist_KMM_abalone.append(mse_KMM_abalone)

    r2_score_KMM_abalone = pearsonr(abalone_np_test_y_list, y_pred_KMM_abalone)
    r2_score_KMM_abalone = (r2_score_KMM_abalone[0])**2
    r2scorelist_KMM_abalone.append(r2_score_KMM_abalone)


    ################### GBRTL ###################
    model_GBRTL_abalone = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBRTL_abalone.fit(abalone_np_train_X, abalone_np_train_y_list)

    y_pred_GBRTL_abalone = model_GBRTL_abalone.predict(abalone_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_abalone = sqrt(mean_squared_error(abalone_np_test_y, y_pred_GBRTL_abalone))
    rmselist_GBRTL_abalone.append(mse_GBRTL_abalone)

    r2_score_GBRTL_abalone = pearsonr(abalone_np_test_y_list, y_pred_GBRTL_abalone)
    r2_score_GBRTL_abalone = (r2_score_GBRTL_abalone[0])**2
    r2scorelist_GBRTL_abalone.append(r2_score_GBRTL_abalone)


    ################### GBR ###################
    model_GBR_abalone = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBR_abalone.fit(abalone_tgt_df_X, abalone_tgt_df_y)

    y_pred_GBR_abalone = model_GBR_abalone.predict(abalone_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_abalone = sqrt(mean_squared_error(abalone_np_test_y, y_pred_GBR_abalone))
    rmselist_GBR_abalone.append(mse_GBR_abalone)

    r2_score_GBR_abalone = pearsonr(abalone_np_test_y_list, y_pred_GBR_abalone)
    r2_score_GBR_abalone = (r2_score_GBR_abalone[0])**2
    r2scorelist_GBR_abalone.append(r2_score_GBR_abalone)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_abalone = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 100, cv = 10) #, kwargs_TwoTrAda)
    model_TwoTrAda_abalone.fit(abalone_np_train_X, abalone_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_abalone = model_TwoTrAda_abalone.predict(abalone_np_test_X)

    mse_TwoTrAda_abalone = sqrt(mean_squared_error(abalone_np_test_y, y_pred_TwoTrAda_abalone))
    rmselist_TwoTrAda_abalone.append(mse_TwoTrAda_abalone)

    r2_score_TwoTrAda_abalone = pearsonr(abalone_np_test_y_list, y_pred_TwoTrAda_abalone)
    r2_score_TwoTrAda_abalone = (r2_score_TwoTrAda_abalone[0])**2
    r2scorelist_TwoTrAda_abalone.append(r2_score_TwoTrAda_abalone)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(abalone_tgt_df_X), len(abalone_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    model_stradaboost_abalone = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_abalone.fit(abalone_np_train_X, abalone_np_train_y_list)
    y_pred_stradaboost_abalone = model_stradaboost_abalone.predict(abalone_np_test_X)


    mse_stradaboost_abalone = sqrt(mean_squared_error(abalone_np_test_y, y_pred_stradaboost_abalone))
    rmselist_stradaboost_abalone.append(mse_stradaboost_abalone)

    r2_score_stradaboost_abalone = pearsonr(abalone_np_test_y_list, y_pred_stradaboost_abalone)
    r2_score_stradaboost_abalone = (r2_score_stradaboost_abalone[0])**2
    r2scorelist_stradaboost_abalone.append(r2_score_stradaboost_abalone)


with open('abalone_rmse.txt', 'w') as abalone_handle_rmse:
    abalone_handle_rmse.write("AdaBoost TL:\n ")
    abalone_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_abalone)

    abalone_handle_rmse.write("\n\nAdaBoost:\n ")
    abalone_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_abalone)

    abalone_handle_rmse.write("\n\nKMM:\n ")
    abalone_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_abalone)

    abalone_handle_rmse.write("\n\nGBRT:\n ")
    abalone_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_abalone)

    abalone_handle_rmse.write("\n\nGBR:\n ")
    abalone_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_abalone)

    abalone_handle_rmse.write("\n\nTrAdaBoost:\n ")
    abalone_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_abalone)

    abalone_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    abalone_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_abalone)


with open('abalone_r2.txt', 'w') as abalone_handle_r2:
    abalone_handle_r2.write("AdaBoost TL:\n ")
    abalone_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_abalone)

    abalone_handle_r2.write("\n\nAdaBoost:\n ")
    abalone_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_abalone)

    abalone_handle_r2.write("\n\nKMM:\n ")
    abalone_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_abalone)

    abalone_handle_r2.write("\n\nGBRT:\n ")
    abalone_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_abalone)

    abalone_handle_r2.write("\n\nGBR:\n ")
    abalone_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_abalone)

    abalone_handle_r2.write("\n\nTrAdaBoost:\n ")
    abalone_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_abalone)

    abalone_handle_r2.write("\n\nSTrAdaBoost:\n ")
    abalone_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_abalone)




######################################################################################

print("-------------------------------------------")
