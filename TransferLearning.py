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

########## Geo plotting libraries ##############
import geopandas as gdp
from matplotlib.colors import ListedColormap
import geoplot as glpt

sns.set_style("darkgrid")

########################################################################################################################################################

CA_df = pd.read_csv('CA.csv')
CA_droplist = ['month', 'doy']
CA_df = CA_df.drop(CA_droplist, axis =1)
print(CA_df.shape)
print(CA_df.columns)
# CA_df["ManDis"] = ""
# CA_df = CA_df.fillna(0)

Peru_df_train = pd.read_csv('lima_train.csv')
# Peru_train_droplist = ['ID','Lat', 'Lon', 'month', 'doy']
Peru_train_droplist = ['ID','Lat', 'Lon', 'month', 'doy', 'short_radi_surf', 'year', 'rid']
Peru_df_train = Peru_df_train.drop(Peru_train_droplist, axis =1)
# Peru_df = Peru_df.fillna(0)
print(Peru_df_train.shape)
print(Peru_df_train.columns)

Peru_df_test = pd.read_csv('Lima_mean.csv')
Peru_test_droplist = ['ID','Lat', 'Lon', 'month', 'doy', 'year', 'short_radi_surf']
Peru_df_test = Peru_df_test.drop(Peru_test_droplist, axis =1)

print(Peru_df_test.shape)
print(Peru_df_test.columns)

############# Finding no. of unique columns in a dataframe ################
# Peru_df = Peru_df.sort_values(by = ['ID'])
# print(Peru_df['ID'].nunique())
# print(Peru_df['ID'].value_counts())

######################### Finding best CA instances based on mean of Peru dataset values ################################################################

# Peru_df_mean = []
# prow = Peru_df_train.mean()
# Peru_df_mean = [prow.AOD550, prow.temp_2m, prow.rhum, prow.zonal_wind_10m, prow.merid_wind_10m, prow.surf_pres, prow.hpbl, prow.conv_prec, prow.NDVI, prow.DEM, prow.Population, prow.distance, prow.NLCD_Developed]
#
# rowidx = 0
# for row in CA_df.itertuples():
#     row_list =[row.AOD550, row.temp_2m, row.rhum, row.zonal_wind_10m, row.merid_wind_10m, row.surf_pres, row.hpbl, row.conv_prec, row.NDVI, row.DEM, row.Population, row.distance, row.NLCD_Developed]
#     man_dis = 0
#     for i in range(0, len(row_list)):
#         tempval = Peru_df_mean[i] - row_list[i]
#         man_dis = man_dis + abs(tempval)
#
#     CA_df.loc[rowidx,"ManDis"] = man_dis
#     print(CA_df.loc[rowidx,"ManDis"])
#     rowidx = rowidx + 1
#
# CA_df = CA_df.sort_values('ManDis')
# CA_df = CA_df.head(4*len(Peru_df_train))
# CA_df = CA_df.drop(['ManDis'], axis =1)

################################################################################################################################################################

########################## Splitting CA and Peru datasets using train_test_split() method ######################################################################
CA_df, other_CA_df = train_test_split(CA_df, test_size=0.20)
# Peru_df, other_Peru_df = train_test_split(Peru_df, test_size=0.4)

################################################################################################################################################################

############################################# Finding combination of sensors for Peru dataset ####################################################################
# list_sensors = ['2940', '2577','4334', '3282', '3431','3654', '495', '4454', '3381', '3425']
# combination_list = list(combinations(list_sensors, 9))
# print(len(combination_list))

###################################################################################################################################################################

# trAdaboost_r2value = []
# Adaboost_r2value = []
# trAdaboost_r2average = []
# Adaboost_r2average = []
#
# iter = 0
#
# ########################################## for loop for the iterations over combination begins here ##############################################################
# # for templist in combination_list:
#     # iter = iter +1
#     # print("Iteration number: ",iter)
#     # print("The list of sensors are: ",templist)
#
# train_peru_df = Peru_df.loc[Peru_df['ID'].isin(templist)]
# other_templist = np.setdiff1d(list_sensors, templist)
# test_peru_df = Peru_df.loc[Peru_df['ID'].isin(other_templist)]
#
# # train_peru_df, test_peru_df = train_test_split(Peru_df, test_size=0.7)
#
# Peru_drop_ID = ['ID']
# train_peru_df = train_peru_df.drop(Peru_drop_ID, axis =1)
# test_peru_df = test_peru_df.drop(Peru_drop_ID, axis =1)


target_column = ['PM25']
CA_df_y = CA_df[target_column]
CA_df_X = CA_df.drop(target_column, axis = 1)

target_column2 = ['pm25_value']
Peru_df_train_y = Peru_df_train[target_column2]
Peru_df_train_X = Peru_df_train.drop(target_column2, axis = 1)

Peru_df_test_y = Peru_df_test[target_column]
Peru_df_test_X = Peru_df_test.drop(target_column, axis =1)

TF_train_X = pd.concat([CA_df_X, Peru_df_train_X], sort= False)
TF_train_y = pd.concat([CA_df_y, Peru_df_train_y], sort= False)

np_TF_train_X = TF_train_X.to_numpy()
np_TF_train_y = TF_train_y.to_numpy()

np_test_peru_df_X = Peru_df_test_X.to_numpy()
np_test_peru_df_y = Peru_df_test_y.to_numpy()

np_TF_train_y_list = np_TF_train_y.ravel()
np_test_peru_df_y_list = np_test_peru_df_y.ravel()

# print(np_TF_train_y_list)


# scaler = StandardScaler()
# scaled_peru_X = scaler.fit_transform(test_peru_df_X)
# test_peru_df_X = pd.DataFrame(scaled_peru_X, columns = test_peru_df_X.columns.tolist())
# test_peru_df_X['Population'] = test_peru_df['Population']
# test_peru_df_X['doy'] = test_peru_df['doy']
# test_peru_df_X['month'] = test_peru_df['month']
# test_peru_df_X = test_peru_df_X.fillna(0)

# print("Shape of California dataset: ",CA_df_X.shape)
# print("Shape of Peru training dataset",train_peru_df_X.shape)
# print("Shape of Peru testing dataset", Peru_df_test.shape)

sample_size = [len(CA_df_X), len(Peru_df_train_X)]
n_estimators = 100
steps = 10
fold = 30
random_state = np.random.RandomState(1)

# ################################################TwoStageAdaBoostR2############################################################################
regr_1 = TwoStageTrAdaBoostR2( DecisionTreeRegressor(max_depth=4),
                      n_estimators = n_estimators, sample_size = sample_size,
                      steps = steps, fold = fold,
                      random_state = random_state)

regr_1.fit(np_TF_train_X, np_TF_train_y_list)
y_pred1 = regr_1.predict(np_test_peru_df_X)

# with open('trAdaBoostOutput6.txt', 'w') as filehandle1:
#     for listitem in y_pred1:
#         filehandle1.write('%s\n' % listitem)
#
# mse_twostageboost = sqrt(mean_squared_error(np_US_df_test_y_list, y_pred1))
# print("RMSE of TwoStageTrAdaboostR2:", mse_twostageboost)
#
# # r2_score_twostageboost = r2_score(np_test_peru_df_y_list, y_pred1)
# r2_score_twostageboost_values = pearsonr(np_US_df_test_y_list, y_pred1)
# r2_score_twostageboost = (r2_score_twostageboost_values[0])**2
# print("R^2 of TwoStageTrAdaboostR2:", r2_score_twostageboost)
#
# # trAdaboost_r2value.append(r2_score_twostageboost)
# # val1 = sum(trAdaboost_r2value)/len(trAdaboost_r2value)
# # trAdaboost_r2average.append(val1)
# #
# # print("AVERAGE R^2 of TrAdaBoost R^2: ",val1)
# #
# # #Plotting the data.
# # peru_map = gdp.read_file('Peru_shape2/PER_adm1.shp') #Upload the Peru map
# # fig, axs = plt.subplots(figsize = (30, 30))
# # peru_map.plot(ax = axs) #Create a plot for the peru map
# #
# # Peru_plot_df['PM25'] = y_pred1
# #
# # rainbow_colors = ListedColormap(['#0000ff', '#0054ff', '#00abff', '#00ffff', '#54ffab', '#abff53', '#ffff00', '#ffaa00', '#ff5400', '#ff0000']) #Using VIBGYOR for colors
# # axs.axis('off') #remove the axis
# # gdf_peru = gdp.GeoDataFrame(Peru_plot_df, geometry=gdp.points_from_xy(Peru_plot_df.Lon, Peru_plot_df.Lat)) #Create a geodataframe
# # gdf_peru.plot(column = 'PM25', ax=axs, cmap=rainbow_colors, legend = True) #Plot the geodatframe
# # plt.show()
# #
# #
# ###############################################Trying AdaBoost for regression####################################################################
#
# regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),
#                           n_estimators = n_estimators)
# regr_2.fit(np_TF_train_X, np_TF_train_y_list)
# y_pred2 = regr_2.predict(np_US_df_test_X)
#
# with open('AdaBoostOutput6.txt', 'w') as filehandle2:
#     for listitem in y_pred2:
#         filehandle2.write('%s\n' % listitem)
#
# mse_adaboost = sqrt(mean_squared_error(np_US_df_test_y_list, y_pred2))
# print("RMSE of regular AdaboostR2:", mse_adaboost)
#
# # r2_score_adaboost = r2_score(test_peru_df_y, y_pred2)
# r2_score_adaboost_values = pearsonr(np_US_df_test_y_list, y_pred2)
# r2_score_adaboost = (r2_score_adaboost_values[0])**2
# print("R^2 of AdaboostR2:", r2_score_adaboost)
#
# # Adaboost_r2value.append(r2_score_adaboost)
# # val2 = sum(Adaboost_r2value)/len(Adaboost_r2value)
# # Adaboost_r2average.append(val2)
# # print("AVERAGE R^2 of AdaBoost R^2: ",val2)
# #
# # #Plotting the data.
# # peru_map = gdp.read_file('Peru_shape2/PER_adm1.shp') #Upload the Peru map
# # fig, axs = plt.subplots(figsize = (30, 30))
# # peru_map.plot(ax = axs) #Create a plot for the peru map
# #
# # Peru_plot_df = Peru_plot_df.drop(['PM25'], axis =1)
# # Peru_plot_df['PM25'] = y_pred2
# #
# # rainbow_colors = ListedColormap(['#0000ff', '#0054ff', '#00abff', '#00ffff', '#54ffab', '#abff53', '#ffff00', '#ffaa00', '#ff5400', '#ff0000']) #Using VIBGYOR for colors
# # axs.axis('off') #remove the axis
# # gdf_peru = gdp.GeoDataFrame(Peru_plot_df, geometry=gdp.points_from_xy(Peru_plot_df.Lon, Peru_plot_df.Lat)) #Create a geodataframe
# # gdf_peru.plot(column = 'PM25', ax=axs, cmap=rainbow_colors, legend = True) #Plot the geodatframe
# # plt.show()
# #
# # # #################################### for loop ends here #################################################################
# # #
# # # #############################R^2 plot
# # # plt.plot(trAdaboost_r2value, color = 'r')
# # # plt.plot(Adaboost_r2value, color = 'm')
# # #
# # # plt.xlim(-1, len(trAdaboost_r2value))
# # # plt.ylim(0,1)
# # # plt.xlabel('Iterations', fontsize=16)
# # # plt.ylabel('R^2 values', fontsize=16)
# # # plt.show()
# # #
# # # ###################################Error plot in R^2 values
# # # plt.errorbar(trAdaboost_r2value, Adaboost_r2value, yerr=None, fmt='none', color='black',
# # #              ecolor='lightgray', elinewidth=3, capsize=0)
# # #
# # # plt.xlim(-1, len(trAdaboost_r2average))
# # # plt.ylim(0,1)
# # # plt.xlabel('Iterations', fontsize=16)
# # # plt.ylabel('R^2 values', fontsize=16)
# # # plt.show()
# # #
# # # max_trAdaBoost = max(trAdaboost_r2value)
# # # max_AdaBoost = max(Adaboost_r2value)
# # #
# # # print("Max TrAdaBoost: ", max_trAdaBoost)
# # # print("Max AdaBoost: ", max_AdaBoost)
