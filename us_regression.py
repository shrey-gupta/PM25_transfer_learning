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


######### Instance Transfer repositories ####################
from adapt.instance_based import TwoStageTrAdaBoostR2

print("Repositories uploaded!!")

from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2
from sklearn.model_selection import GridSearchCV
from adapt.instance_based import KMM

print("Second Upload Completed!!")

########################################## Shorter Individual Datasets (US) ###############################################################
from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

### Removing the error: Input contains NaN, infinity or a value too large for dtype('float64')
### Link used: https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
### Also do reset_index for the dataframe

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame) #"df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


for i in range(1,101):
    US_df_train = pd.read_csv('US_data/US_multi_slice/Target/Target'+str(i)+'.csv')
    US_train_droplist = ['cmaq_id', 'cmaq_x', 'cmaq_y', 'Latitude', 'Longitude', 'year', 'day', 'rid']
    US_df_train = US_df_train.drop(US_train_droplist, axis = 1)
    US_df_train = US_df_train.rename(columns={"is": "ISS"})
    US_df_train = clean_dataset(US_df_train)
    US_df_train = US_df_train.reset_index(drop=True)
    print(US_df_train.shape)

    US_df_transfer = pd.read_csv('US_data/US_multi_slice/US_source.csv')
    US_transfer_droplist = ['cmaq_id', 'cmaq_x', 'cmaq_y', 'Latitude', 'Longitude', 'year', 'day', 'rid']
    US_df_transfer = US_df_transfer.drop(US_transfer_droplist, axis = 1)
    US_df_transfer = US_df_transfer.rename(columns={"is": "ISS"})
    US_df_transfer = clean_dataset(US_df_transfer)
    US_df_transfer = US_df_transfer.reset_index(drop=True)
    print(US_df_transfer.columns)

    US_df_test = pd.read_csv('US_data/US_multi_slice/Test/Test'+str(i)+'.csv')
    US_test_droplist = ['cmaq_id', 'cmaq_x', 'cmaq_y', 'Latitude', 'Longitude', 'year', 'day', 'rid']
    US_df_test = US_df_test.drop(US_test_droplist, axis = 1)
    US_df_test = US_df_test.rename(columns={"is": "ISS"})
    US_df_test = clean_dataset(US_df_test)
    US_df_test = US_df_test.reset_index(drop=True)
    print(US_df_test.shape)

    target_column = ['pm25_value']
    US_df_train_y = US_df_train[target_column]
    US_df_train_X = US_df_train.drop(target_column, axis = 1)

    US_df_transfer_y = US_df_transfer[target_column]
    US_df_transfer_X = US_df_transfer.drop(target_column, axis = 1)

    US_df_test_y = US_df_test[target_column]
    US_df_test_X = US_df_test.drop(target_column, axis = 1)

    TF_train_X = pd.concat([US_df_transfer_X, US_df_train_X], sort= False)
    TF_train_y = pd.concat([US_df_transfer_y, US_df_train_y], sort= False)

    np_TF_train_X = TF_train_X.to_numpy()
    np_TF_train_y = TF_train_y.to_numpy()

    # np_TF_train_X = US_df_train_X.to_numpy()
    # np_TF_train_y = US_df_train_y.to_numpy()

    np_US_df_test_X = US_df_test_X.to_numpy()
    np_US_df_test_y = US_df_test_y.to_numpy()

    np_TF_train_y_list = np_TF_train_y.ravel()
    np_US_df_test_y_list = np_US_df_test_y.ravel()

    ######################################## Phase 1: Seeding Technique (US) ###################################################

    kmeans = KMeans(n_clusters = 150, random_state=0).fit(US_df_transfer)

    US_alternate_df = US_df_transfer.copy()
    US_alternate_df_np = US_df_transfer.to_numpy()

    idxlist = []
    US_new_df_list = []

    for rowkm in kmeans.cluster_centers_:
        mindist = -99
        rowidx = 0
        idx = 0
        for row in US_alternate_df_np:
            dst = distance.euclidean(row, rowkm)

            if(dst >= mindist):
                mindist = dst
                rowidx = idx
                rowval = row

            idx = idx + 1

    #     print("Row selected: ", rowidx) #Alternate_df.loc[rowidx,:]\
    #     print("Min. distance: ", mindist)
    #     print("Matrix shape: ", kinematics_alternate_df_np.shape)
        US_new_df_list.append(rowval)
        US_alternate_df = np.delete(US_alternate_df_np, rowidx, 0)
        idxlist.append(rowidx)


    US_new_df = pd.DataFrame(np.vstack(US_new_df_list))

    print("Shape of dataset extracted: ")
    print(US_new_df.shape)
    print("----------------------------------------------")

    ##################################################### Phase 2: Seeding (US) ################################################

    US_alternate_transfer_df = US_df_transfer[1:].copy()
    US_alternate_transfer_df_np = US_alternate_transfer_df.to_numpy()

    idxlist2 = []
    US_final_df_list = []

    for row_nm in US_new_df_list:
        min_dist = -99
        row_idx = 0
        idx_val = 0
        for row_alt in US_alternate_transfer_df_np:
            dst = distance.euclidean(row_alt, row_nm)
            if(dst >= mindist):
                min_dist = dst
                row_idx = idx_val
                row_val = row_alt

            idx_val = idx_val + 1

    #     print("Row selected: ", row_idx) #Alternate_df.loc[rowidx,:]\
    #     print("Min. distance: ", min_dist)
    #     print("Matrix shape: ", Elevators_alternate_source_df_np.shape)
        US_final_df_list.append(row_val)
        US_alternate_transfer_df_np = np.delete(US_alternate_transfer_df_np, row_idx, 0)
        idxlist2.append(row_idx)


    US_final_df = pd.DataFrame(np.vstack(US_final_df_list), columns = US_df_transfer.columns)

    print("----------------------------------------------")
    print("Shape of source before :",US_df_transfer.shape)
    US_df_transfer = pd.DataFrame(np.vstack(US_alternate_transfer_df_np), columns= US_df_transfer.columns)
    print("Shape of source after :", US_df_transfer.shape)

    print("----------------------------------------------")
    print("Shape of target before :", US_df_train.shape)
    US_df_train = pd.concat([US_df_train, US_final_df], ignore_index=True)
    print("Shape of target after :", US_df_train.shape)

    print("----------------------------------------------")


    ################################## Finding best instances from the source dataset (US) #############################

    US_df_transfer["ManDis"] = ""

    US_df_train_mean = []
    prow = US_df_train.mean()
    US_df_train_mean = [prow.elev, prow.emissi11, prow.forest_cover, prow.high, prow.limi, prow.local, prow.ISS,
           prow.pd, prow.nldas_pevapsfc, prow.nldas_dlwrfsfc, prow.nldas_dswrfsfc, prow.nldas_cape, prow.nldas_fpcsfc,
           prow.nldas_pcpsfc, prow.nldas_rh2m, prow.nldas_tmp2m, prow.nldas_vgrd10m, prow.nldas_ugrd10m, prow.nldas_pressfc,
           prow.narr_dpt, prow.narr_vis, prow.narr_hpbl, prow.narr_rh2m, prow.narr_tmp2m, prow.narr_ugrd10m, prow.narr_vgrd10m,
           prow.narr_rh30mb, prow.narr_rh63mb, prow.narr_rh96mb, prow.narr_rh129mb, prow.narr_rh1512mb, prow.narr_rh1815mb,
           prow.narr_tmp30mb, prow.narr_tmp63mb, prow.narr_tmp96mb, prow.narr_tmp129mb, prow.narr_tmp1512mb, prow.narr_tmp1815mb,
           prow.narr_ugrd30m, prow.narr_ugrd30mb, prow.narr_ugrd63mb, prow.narr_ugrd96mb, prow.narr_ugrd129mb, prow.narr_ugrd1512mb,
           prow.narr_ugrd1815mb, prow.narr_vgrd30m, prow.narr_vgrd30mb, prow.narr_vgrd63mb, prow.narr_vgrd96mb, prow.narr_vgrd129mb,
           prow.narr_vgrd1512mb, prow.narr_vgrd1815mb, prow.aod_value, prow.pm25_value, prow.gc_aod]

    rowidx = 0

    for row in US_df_transfer.itertuples():
        row_list =[row.elev, row.emissi11, row.forest_cover, row.high, row.limi, row.local, row.ISS,
               row.pd, row.nldas_pevapsfc, row.nldas_dlwrfsfc, row.nldas_dswrfsfc, row.nldas_cape, row.nldas_fpcsfc,
               row.nldas_pcpsfc, row.nldas_rh2m, row.nldas_tmp2m, row.nldas_vgrd10m, row.nldas_ugrd10m, row.nldas_pressfc,
               row.narr_dpt, row.narr_vis, row.narr_hpbl, row.narr_rh2m, row.narr_tmp2m, row.narr_ugrd10m, row.narr_vgrd10m,
               row.narr_rh30mb, row.narr_rh63mb, row.narr_rh96mb, row.narr_rh129mb, row.narr_rh1512mb, row.narr_rh1815mb,
               row.narr_tmp30mb, row.narr_tmp63mb, row.narr_tmp96mb, row.narr_tmp129mb, row.narr_tmp1512mb, row.narr_tmp1815mb,
               row.narr_ugrd30m, row.narr_ugrd30mb, row.narr_ugrd63mb, row.narr_ugrd96mb, row.narr_ugrd129mb, row.narr_ugrd1512mb,
               row.narr_ugrd1815mb, row.narr_vgrd30m, row.narr_vgrd30mb, row.narr_vgrd63mb, row.narr_vgrd96mb, row.narr_vgrd129mb,
               row.narr_vgrd1512mb, row.narr_vgrd1815mb, row.aod_value, row.pm25_value, row.gc_aod]

        man_dis = 0
        for i in range(0, len(row_list)):
            tempval = US_df_train_mean[i] - row_list[i]
            man_dis = man_dis + abs(tempval)

        US_df_transfer.loc[rowidx,"ManDis"] = man_dis
        rowidx = rowidx + 1

    US_df_transfer = US_df_transfer.sort_values(by =['ManDis'])
    US_df_transfer = US_df_transfer.head(6000)
    US_df_transfer = US_df_transfer.drop(['ManDis'], axis =1)
    US_df_transfer = US_df_transfer.reset_index(drop=True)

    #################### Splitting with small target set and large source and test set #############
    print("Target Set: ", US_df_train.shape)
    print("Source Set: ", US_df_transfer.shape)
    print("Test Set: ", US_df_test.shape)


    target_column = ['pm25_value']
    US_df_train_y = US_df_train[target_column]
    US_df_train_X = US_df_train.drop(target_column, axis = 1)

    US_df_transfer_y = US_df_transfer[target_column]
    US_df_transfer_X = US_df_transfer.drop(target_column, axis = 1)

    US_df_test_y = US_df_test[target_column]
    US_df_test_X = US_df_test.drop(target_column, axis = 1)

    TF_train_X = pd.concat([US_df_transfer_X, US_df_train_X], sort= False)
    TF_train_y = pd.concat([US_df_transfer_y, US_df_train_y], sort= False)

    np_TF_train_X = TF_train_X.to_numpy()
    np_TF_train_y = TF_train_y.to_numpy()

    # np_TF_train_X = US_df_train_X.to_numpy()
    # np_TF_train_y = US_df_train_y.to_numpy()

    np_US_df_test_X = US_df_test_X.to_numpy()
    np_US_df_test_y = US_df_test_y.to_numpy()

    np_TF_train_y_list = np_TF_train_y.ravel()
    np_US_df_test_y_list = np_US_df_test_y.ravel()

    print("---------------------------")

    ##################################################################################################################

    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2


    sample_size = [len(US_df_train_X), len(US_df_transfer_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    print("TL Round:", i)
    print("-------------------------------------------")

    r2scorelist_stradaboost_us = []
    rmselist_stradaboost_us = []

    r2scorelist_GBRTL_us = []
    rmselist_GBRTL_us = []

    r2scorelist_AdaTL_us = []
    rmselist_AdaTL_us = []


    for x in range(0, 20):

        ###################################### STrAdaBoost.R2 ##########################################################
        model_stradaboost_us = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                              n_estimators = n_estimators, sample_size = sample_size,
                              steps = steps, fold = fold,
                              random_state = random_state)

        model_stradaboost_us.fit(np_TF_train_X, np_TF_train_y_list)
        y_pred_us = model_stradaboost_us.predict(np_US_df_test_X)

        mse_stradaboost_us = sqrt(mean_squared_error(np_US_df_test_y_list, y_pred_us))
        rmselist_stradaboost_us.append(mse_stradaboost_us)

        r2_score_stradaboost_us = pearsonr(np_US_df_test_y_list, y_pred_us)
        r2_score_stradaboost_us = (r2_score_stradaboost_us[0])**2
        r2scorelist_stradaboost_us.append(r2_score_stradaboost_us)



    with open('US_data/US_multi_slice/Results/us_rmse'+str(i)+'.txt', 'a') as us_data_rmse:
        us_data_rmse.write("STrAdaBoost TL:\n ")
        us_data_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_us)


    with open('US_data/US_multi_slice/Results/us_r2'+str(i)+'.txt', 'a') as us_data_r2:
        us_data_r2.write("STrAdaBoost TL:\n ")
        us_data_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_us)
