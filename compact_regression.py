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

################################### compact ################################################################
## 'nox' found to be correlated at 0.4 :: [0.385 - 0.871] :: 50
#################################################################################################################################

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame) #"df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


colnames_compact = ['lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'pgout', 'ppgout',
                    'pgfree', 'pgscan', 'atch', 'pgin', 'ppgin', 'pflt', 'vflt', 'runqsz', 'freemem', 'freeswap', 'usr' ]
compactData_df = pd.read_csv('UCI_regression/ComputerActivity/cpu_act.data', header = None, names = colnames_compact)
print("Computer Activity Data")
print(compactData_df.shape)
compactData_df

# print(compactData_df['pgin'].min())
# print(compactData_df['pgin'].max())


ss = StandardScaler()

drop_col_compact = ['pgin']

compact_train_df = compactData_df.loc[(compactData_df['pgin'] > 0.80) & (compactData_df['pgin'] <= 5.00)]
compact_train_df = compact_train_df.drop(drop_col_compact, axis = 1)
compact_train_df = compact_train_df.reset_index(drop=True)
compact_train_df = clean_dataset(compact_train_df)
compact_train_df = compact_train_df.reset_index(drop=True)
print("Target Set: ",compact_train_df.shape)


compact_source1_df = compactData_df.loc[(compactData_df['pgin'] > 5.00)]
compact_source1_df = compact_source1_df.drop(drop_col_compact, axis = 1)
compact_source1_df = compact_source1_df.reset_index(drop=True)
compact_source1_df = clean_dataset(compact_source1_df)
compact_source1_df = compact_source1_df.reset_index(drop=True)
print("Source Set 1: ",compact_source1_df.shape)


compact_source2_df = compactData_df.loc[(compactData_df['pgin'] <= 0.80)]
compact_source2_df = compact_source2_df.drop(drop_col_compact, axis = 1)
compact_source2_df = compact_source2_df.reset_index(drop=True)
compact_source2_df = clean_dataset(compact_source2_df)
compact_source2_df = compact_source2_df.reset_index(drop=True)
print("Source Set 2: ",compact_source2_df.shape)


compact_source_df = pd.concat([compact_source1_df, compact_source2_df], ignore_index=True)
compact_source_df = compact_source_df.sample(frac = 0.5, replace = True, random_state = 1)

print("Final Source Set: ",compact_source_df.shape)


#################### Splitting into features and target ####################
target_column_compact = ['usr']

compact_train_df_y = compact_train_df[target_column_compact]
compact_train_df_X = compact_train_df.drop(target_column_compact, axis = 1)
compact_cols = compact_train_df_X.columns
compact_train_df_X[compact_cols] = ss.fit_transform(compact_train_df_X[compact_cols])


compact_source_df_y = compact_source_df[target_column_compact]
compact_source_df_X = compact_source_df.drop(target_column_compact, axis = 1)
compact_cols = compact_source_df_X.columns
compact_source_df_X[compact_cols] = ss.fit_transform(compact_source_df_X[compact_cols])
compact_source_df_X


print("---------------------------")

########################### Transfer Learning compact #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}


print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_compact = []
rmselist_AdaTL_compact = []

r2scorelist_Ada_compact = []
rmselist_Ada_compact = []

r2scorelist_KMM_compact = []
rmselist_KMM_compact = []

r2scorelist_GBRTL_compact = []
rmselist_GBRTL_compact = []

r2scorelist_GBR_compact = []
rmselist_GBR_compact = []

r2scorelist_TwoTrAda_compact = []
rmselist_TwoTrAda_compact = []

r2scorelist_stradaboost_compact = []
rmselist_stradaboost_compact = []



kfold = KFold(n_splits = 10, random_state = 42, shuffle=False)

for train_ix, test_ix in kfold.split(compact_train_df_X):
    ############### get data ###############
    compact_test_df_X, compact_tgt_df_X  = compact_train_df_X.iloc[train_ix], compact_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    compact_test_df_y, compact_tgt_df_y  = compact_train_df_y.iloc[train_ix], compact_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(compact_tgt_df_X.shape, compact_test_df_X.shape)

    ############### Merging the datasets ##########################################
    compact_X_df = pd.concat([compact_tgt_df_X, compact_source_df_X], ignore_index=True)
    compact_y_df = pd.concat([compact_tgt_df_y, compact_source_df_y], ignore_index=True)

    compact_np_train_X = compact_X_df.to_numpy()
    compact_np_train_y = compact_y_df.to_numpy()

    compact_np_test_X = compact_test_df_X.to_numpy()
    compact_np_test_y = compact_test_df_y.to_numpy()

    compact_np_train_y_list = compact_np_train_y.ravel()
    compact_np_test_y_list = compact_np_test_y.ravel()

    src_size_compact = len(compact_source_df_y)
    tgt_size_compact = len(compact_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_compact - 1), step=1)
    tgt_idx = np.arange(start=src_size_compact, stop=((src_size_compact + tgt_size_compact)-1), step=1)



    ################### AdaBoost Tl ###################
    model_AdaTL_compact = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_AdaTL_compact.fit(compact_np_train_X, compact_np_train_y_list)

    y_pred_AdaTL_compact = model_AdaTL_compact.predict(compact_np_test_X)

    mse_AdaTL_compact = sqrt(mean_squared_error(compact_np_test_y, y_pred_AdaTL_compact))
    rmselist_AdaTL_compact.append(mse_AdaTL_compact)

    r2_score_AdaTL_compact = pearsonr(compact_np_test_y_list, y_pred_AdaTL_compact)
    r2_score_AdaTL_compact = (r2_score_AdaTL_compact[0])**2
    r2scorelist_AdaTL_compact.append(r2_score_AdaTL_compact)


    ################### AdaBoost ###################
    model_Ada_compact = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_Ada_compact.fit(compact_tgt_df_X, compact_tgt_df_y)

    y_pred_Ada_compact = model_Ada_compact.predict(compact_np_test_X)

    mse_Ada_compact = sqrt(mean_squared_error(compact_np_test_y, y_pred_Ada_compact))
    rmselist_Ada_compact.append(mse_Ada_compact)

    r2_score_Ada_compact = pearsonr(compact_np_test_y_list, y_pred_Ada_compact)
    r2_score_Ada_compact = (r2_score_Ada_compact[0])**2
    r2scorelist_Ada_compact.append(r2_score_Ada_compact)


   ################### KMM ###################
    model_KMM_compact = KMM(get_estimator = get_estimator)
    model_KMM_compact.fit(compact_np_train_X, compact_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_compact = model_KMM_compact.predict(compact_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_compact = sqrt(mean_squared_error(compact_np_test_y, y_pred_KMM_compact))
    rmselist_KMM_compact.append(mse_KMM_compact)

    r2_score_KMM_compact = pearsonr(compact_np_test_y_list, y_pred_KMM_compact)
    r2_score_KMM_compact = (r2_score_KMM_compact[0])**2
    r2scorelist_KMM_compact.append(r2_score_KMM_compact)


    ################### GBRTL ###################
    model_GBRTL_compact = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)
    model_GBRTL_compact.fit(compact_np_train_X, compact_np_train_y_list)

    y_pred_GBRTL_compact = model_GBRTL_compact.predict(compact_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_compact = sqrt(mean_squared_error(compact_np_test_y, y_pred_GBRTL_compact))
    rmselist_GBRTL_compact.append(mse_GBRTL_compact)

    r2_score_GBRTL_compact = pearsonr(compact_np_test_y_list, y_pred_GBRTL_compact)
    r2_score_GBRTL_compact = (r2_score_GBRTL_compact[0])**2
    r2scorelist_GBRTL_compact.append(r2_score_GBRTL_compact)


    ################### GBR ###################
    model_GBR_compact = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)
    model_GBR_compact.fit(compact_tgt_df_X, compact_tgt_df_y)

    y_pred_GBR_compact = model_GBR_compact.predict(compact_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_compact = sqrt(mean_squared_error(compact_np_test_y, y_pred_GBR_compact))
    rmselist_GBR_compact.append(mse_GBR_compact)

    r2_score_GBR_compact = pearsonr(compact_np_test_y_list, y_pred_GBR_compact)
    r2_score_GBR_compact = (r2_score_GBR_compact[0])**2
    r2scorelist_GBR_compact.append(r2_score_GBR_compact)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_compact = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 100, cv = 10) #, kwargs_TwoTrAda)
    model_TwoTrAda_compact.fit(compact_np_train_X, compact_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_compact = model_TwoTrAda_compact.predict(compact_np_test_X)

    mse_TwoTrAda_compact = sqrt(mean_squared_error(compact_np_test_y, y_pred_TwoTrAda_compact))
    rmselist_TwoTrAda_compact.append(mse_TwoTrAda_compact)

    r2_score_TwoTrAda_compact = pearsonr(compact_np_test_y_list, y_pred_TwoTrAda_compact)
    r2_score_TwoTrAda_compact = (r2_score_TwoTrAda_compact[0])**2
    r2scorelist_TwoTrAda_compact.append(r2_score_TwoTrAda_compact)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(compact_tgt_df_X), len(compact_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    model_stradaboost_compact = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_compact.fit(compact_np_train_X, compact_np_train_y_list)
    y_pred_stradaboost_compact = model_stradaboost_compact.predict(compact_np_test_X)


    mse_stradaboost_compact = sqrt(mean_squared_error(compact_np_test_y, y_pred_stradaboost_compact))
    rmselist_stradaboost_compact.append(mse_stradaboost_compact)

    r2_score_stradaboost_compact = pearsonr(compact_np_test_y_list, y_pred_stradaboost_compact)
    r2_score_stradaboost_compact = (r2_score_stradaboost_compact[0])**2
    r2scorelist_stradaboost_compact.append(r2_score_stradaboost_compact)


with open('compact_rmse.txt', 'w') as compact_handle_rmse:
    compact_handle_rmse.write("AdaBoost TL:\n ")
    compact_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_compact)

    compact_handle_rmse.write("\n\nAdaBoost:\n ")
    compact_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_compact)

    compact_handle_rmse.write("\n\nKMM:\n ")
    compact_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_compact)

    compact_handle_rmse.write("\n\nGBRT:\n ")
    compact_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_compact)

    compact_handle_rmse.write("\n\nGBR:\n ")
    compact_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_compact)

    compact_handle_rmse.write("\n\nTrAdaBoost:\n ")
    compact_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_compact)

    compact_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    compact_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_compact)


with open('compact_r2.txt', 'w') as compact_handle_r2:
    compact_handle_r2.write("AdaBoost TL:\n ")
    compact_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_compact)

    compact_handle_r2.write("\n\nAdaBoost:\n ")
    compact_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_compact)

    compact_handle_r2.write("\n\nKMM:\n ")
    compact_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_compact)

    compact_handle_r2.write("\n\nGBRT:\n ")
    compact_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_compact)

    compact_handle_r2.write("\n\nGBR:\n ")
    compact_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_compact)

    compact_handle_r2.write("\n\nTrAdaBoost:\n ")
    compact_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_compact)

    compact_handle_r2.write("\n\nSTrAdaBoost:\n ")
    compact_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_compact)



######################################################################################

print("-------------------------------------------")
