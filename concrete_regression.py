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

################################### Concrete ###########################################################################################################
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame) #"df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


ConcreteData_df = pd.read_excel('UCI_regression/Concrete/Concrete_Data.xls') ## 'Cement' found to be correlated at 0.4 :: 100
print("Concrete Data")
print("-------------------------------------------")
print(ConcreteData_df.shape)

# concrete_cols = ConcreteData_df.columns
# ss = StandardScaler()
# ConcreteData_df[concrete_cols] = ss.fit_transform(ConcreteData_df[concrete_cols])
# print(ConcreteData_df)


ss = StandardScaler()

drop_col_concrete = ['Cement']

concrete_train_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] > 225) & (ConcreteData_df['Cement'] <= 350)]
concrete_train_df = concrete_train_df.drop(drop_col_concrete, axis = 1)
concrete_train_df = concrete_train_df.reset_index(drop=True)
concrete_train_df = clean_dataset(concrete_train_df)
concrete_train_df = concrete_train_df.reset_index(drop=True)
print("Training Set: ",concrete_train_df.shape)
concrete_train_df

concrete_source1_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] > 350)]
concrete_source1_df = concrete_source1_df.drop(drop_col_concrete, axis = 1)
concrete_source1_df = concrete_source1_df.reset_index(drop=True)
concrete_source1_df = clean_dataset(concrete_source1_df)
concrete_source1_df = concrete_source1_df.reset_index(drop=True)
print("Source Set 1: ",concrete_source1_df.shape)

concrete_source2_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] <= 225)]
concrete_source2_df = concrete_source2_df.drop(drop_col_concrete, axis = 1)
concrete_source2_df = concrete_source2_df.reset_index(drop=True)
concrete_source2_df = clean_dataset(concrete_source2_df)
concrete_source2_df = concrete_source2_df.reset_index(drop=True)
print("Source Set 2: ",concrete_source2_df.shape)

concrete_source_df = pd.concat([concrete_source1_df, concrete_source2_df], ignore_index=True)
print("Final Source Set: ",concrete_source_df.shape)


#################### Splitting into features and target ####################
target_column_concrete = ['ConcreteCompressiveStrength']

concrete_train_df_y = concrete_train_df[target_column_concrete]
concrete_train_df_X = concrete_train_df.drop(target_column_concrete, axis = 1)
concrete_cols = concrete_train_df_X.columns
concrete_train_df_X[concrete_cols] = ss.fit_transform(concrete_train_df_X[concrete_cols])


concrete_source_df_y = concrete_source_df[target_column_concrete]
concrete_source_df_X = concrete_source_df.drop(target_column_concrete, axis = 1)
concrete_cols = concrete_source_df_X.columns
concrete_source_df_X[concrete_cols] = ss.fit_transform(concrete_source_df_X[concrete_cols])


########################### Transfer Learning Concrete #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}



print("Adaboost.R2 Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_concrete = []
rmselist_AdaTL_concrete = []

r2scorelist_Ada_concrete = []
rmselist_Ada_concrete = []

r2scorelist_KMM_concrete = []
rmselist_KMM_concrete = []

r2scorelist_GBRTL_concrete = []
rmselist_GBRTL_concrete = []

r2scorelist_GBR_concrete = []
rmselist_GBR_concrete = []

r2scorelist_TwoTrAda_concrete = []
rmselist_TwoTrAda_concrete = []

r2scorelist_stradaboost_concrete = []
rmselist_stradaboost_concrete = []



kfold = KFold(n_splits = 10, random_state = 42, shuffle=False)

for train_ix, test_ix in kfold.split(concrete_train_df_X):
    ############### get data ###############
    concrete_test_df_X, concrete_tgt_df_X  = concrete_train_df_X.iloc[train_ix], concrete_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    concrete_test_df_y, concrete_tgt_df_y  = concrete_train_df_y.iloc[train_ix], concrete_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(concrete_tgt_df_X.shape, concrete_test_df_X.shape)

    ############### Merging the datasets ##########################################
    concrete_X_df = pd.concat([concrete_tgt_df_X, concrete_source_df_X], ignore_index=True)
    concrete_y_df = pd.concat([concrete_tgt_df_y, concrete_source_df_y], ignore_index=True)

    concrete_np_train_X = concrete_X_df.to_numpy()
    concrete_np_train_y = concrete_y_df.to_numpy()

    concrete_np_test_X = concrete_test_df_X.to_numpy()
    concrete_np_test_y = concrete_test_df_y.to_numpy()

    concrete_np_train_y_list = concrete_np_train_y.ravel()
    concrete_np_test_y_list = concrete_np_test_y.ravel()

    src_size_concrete = len(concrete_source_df_y)
    tgt_size_concrete = len(concrete_tgt_df_y)

    src_idx = np.arange(start = 0, stop = (src_size_concrete - 1), step = 1)
    tgt_idx = np.arange(start = src_size_concrete, stop = ((src_size_concrete + tgt_size_concrete) - 1), step=1)


    ################### AdaBoost Tl ###################
    model_AdaTL_concrete = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_AdaTL_concrete.fit(concrete_np_train_X, concrete_np_train_y_list)

    y_pred_AdaTL_concrete = model_AdaTL_concrete.predict(concrete_np_test_X)

    mse_AdaTL_concrete = sqrt(mean_squared_error(concrete_np_test_y, y_pred_AdaTL_concrete))
    rmselist_AdaTL_concrete.append(mse_AdaTL_concrete)

    r2_score_AdaTL_concrete = pearsonr(concrete_np_test_y_list, y_pred_AdaTL_concrete)
    r2_score_AdaTL_concrete = (r2_score_AdaTL_concrete[0])**2
    r2scorelist_AdaTL_concrete.append(r2_score_AdaTL_concrete)


    ################### AdaBoost ###################
    model_Ada_concrete = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_Ada_concrete.fit(concrete_tgt_df_X, concrete_tgt_df_y)

    y_pred_ada_concrete = model_Ada_concrete.predict(concrete_np_test_X)

    mse_Ada_concrete = sqrt(mean_squared_error(concrete_np_test_y, y_pred_ada_concrete))
    rmselist_Ada_concrete.append(mse_Ada_concrete)

    r2_score_Ada_concrete = pearsonr(concrete_np_test_y_list, y_pred_ada_concrete)
    r2_score_Ada_concrete = (r2_score_Ada_concrete[0])**2
    r2scorelist_Ada_concrete.append(r2_score_Ada_concrete)


   ################### KMM ###################
    model_KMM_concrete = KMM(get_estimator = get_estimator)
    model_KMM_concrete.fit(concrete_np_train_X, concrete_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_concrete = model_KMM_concrete.predict(concrete_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_concrete = sqrt(mean_squared_error(concrete_np_test_y, y_pred_KMM_concrete))
    rmselist_KMM_concrete.append(mse_KMM_concrete)

    r2_score_KMM_concrete = pearsonr(concrete_np_test_y_list, y_pred_KMM_concrete)
    r2_score_KMM_concrete = (r2_score_KMM_concrete[0])**2
    r2scorelist_KMM_concrete.append(r2_score_KMM_concrete)


    ################### GBRTL ###################
    model_GBRTL_concrete = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBRTL_concrete.fit(concrete_np_train_X, concrete_np_train_y_list)

    y_pred_GBRTL_concrete = model_GBRTL_concrete.predict(concrete_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_concrete = sqrt(mean_squared_error(concrete_np_test_y, y_pred_GBRTL_concrete))
    rmselist_GBRTL_concrete.append(mse_GBRTL_concrete)

    r2_score_GBRTL_concrete = pearsonr(concrete_np_test_y_list, y_pred_GBRTL_concrete)
    r2_score_GBRTL_concrete = (r2_score_GBRTL_concrete[0])**2
    r2scorelist_GBRTL_concrete.append(r2_score_GBRTL_concrete)


    ################### GBR ###################
    model_GBR_concrete = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)
    model_GBR_concrete.fit(concrete_tgt_df_X, concrete_tgt_df_y)

    y_pred_GBR_concrete = model_GBR_concrete.predict(concrete_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_concrete = sqrt(mean_squared_error(concrete_np_test_y, y_pred_GBR_concrete))
    rmselist_GBR_concrete.append(mse_GBR_concrete)

    r2_score_GBR_concrete = pearsonr(concrete_np_test_y_list, y_pred_GBR_concrete)
    r2_score_GBR_concrete = (r2_score_GBR_concrete[0])**2
    r2scorelist_GBR_concrete.append(r2_score_GBR_concrete)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_concrete = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 100, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_concrete.fit(concrete_np_train_X, concrete_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_concrete = model_TwoTrAda_concrete.predict(concrete_np_test_X)

    mse_TwoTrAda_concrete = sqrt(mean_squared_error(concrete_np_test_y, y_pred_TwoTrAda_concrete))
    rmselist_TwoTrAda_concrete.append(mse_TwoTrAda_concrete)

    r2_score_TwoTrAda_concrete = pearsonr(concrete_np_test_y_list, y_pred_TwoTrAda_concrete)
    r2_score_TwoTrAda_concrete = (r2_score_TwoTrAda_concrete[0])**2
    r2scorelist_TwoTrAda_concrete.append(r2_score_TwoTrAda_concrete)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(concrete_tgt_df_X), len(concrete_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)


    model_stradaboost_concrete = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_concrete.fit(concrete_np_train_X, concrete_np_train_y_list)
    y_pred_stradaboost_concrete = model_stradaboost_concrete.predict(concrete_np_test_X)


    mse_stradaboost_concrete = sqrt(mean_squared_error(concrete_np_test_y, y_pred_stradaboost_concrete))
    rmselist_stradaboost_concrete.append(mse_stradaboost_concrete)

    r2_score_stradaboost_concrete = pearsonr(concrete_np_test_y_list, y_pred_stradaboost_concrete)
    r2_score_stradaboost_concrete = (r2_score_stradaboost_concrete[0])**2
    r2scorelist_stradaboost_concrete.append(r2_score_stradaboost_concrete)



with open('concrete_rmse.txt', 'w') as concrete_handle_rmse:
    concrete_handle_rmse.write("AdaBoost TL:\n ")
    concrete_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_concrete)

    concrete_handle_rmse.write("\n\nAdaBoost:\n ")
    concrete_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_concrete)

    concrete_handle_rmse.write("\n\nKMM:\n ")
    concrete_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_concrete)

    concrete_handle_rmse.write("\n\nGBRT:\n ")
    concrete_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_concrete)

    concrete_handle_rmse.write("\n\nGBR:\n ")
    concrete_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_concrete)

    concrete_handle_rmse.write("\n\nTrAdaBoost:\n ")
    concrete_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_concrete)

    concrete_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    concrete_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_concrete)


with open('concrete_r2.txt', 'w') as concrete_handle_r2:
    concrete_handle_r2.write("AdaBoost TL:\n ")
    concrete_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_concrete)

    concrete_handle_r2.write("\n\nAdaBoost:\n ")
    concrete_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_concrete)

    concrete_handle_r2.write("\n\nKMM:\n ")
    concrete_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_concrete)

    concrete_handle_r2.write("\n\nGBRT:\n ")
    concrete_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_concrete)

    concrete_handle_r2.write("\n\nGBR:\n ")
    concrete_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_concrete)

    concrete_handle_r2.write("\n\nTrAdaBoost:\n ")
    concrete_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_concrete)

    concrete_handle_r2.write("\n\nSTrAdaBoost:\n ")
    concrete_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_concrete)


######################################################################################


# print("RMSE of Adaboost.R2(TL):", statistics.mean(rmselist_AdaTL_concrete))
# print("R^2 of AdaboostR2(TL):", statistics.mean(r2scorelist_AdaTL_concrete))
# print("\n")
# print("RMSE of Adaboost.R2(TL):", rmselist_AdaTL_concrete)
# print("R^2 of AdaboostR2(TL):", r2scorelist_AdaTL_concrete)


print("-------------------------------------------")
