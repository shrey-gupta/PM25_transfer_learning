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
from sklearn.preprocessing import StandardScaler
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr
from math import sqrt

#Geo plotting libraries
import geopandas as gdp
from matplotlib.colors import ListedColormap
import geoplot as glpt

#Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

sns.set_style("darkgrid")

########################################################################################################################################################
US_df_train = pd.read_csv('lima_train.csv')
US_train_droplist = ['ID', 'Lat', 'Lon','year', 'month', 'doy', 'rid']
US_df_train = US_df_train.drop(US_train_droplist, axis = 1)

print(US_df_train.columns)
print(US_df_train.shape)


US_df_transfer = pd.read_csv('ca_nocenter.csv')
# US_transfer_droplist = ['month', 'doy']
US_transfer_droplist = ['cmaq_id','Lat', 'Lon','year', 'month', 'doy', 'rid']
US_df_transfer = US_df_transfer.drop(US_transfer_droplist, axis = 1)
US_df_transfer = US_df_transfer.fillna(0)

print(US_df_transfer.columns)
print(US_df_transfer.shape)

US_df_transfer2 = pd.read_csv('ca_remaining.csv')
US_transfer2_droplist = ['cmaq_id','Lat', 'Lon','year', 'month', 'doy', 'rid']
US_df_transfer2 = US_df_transfer2.drop(US_transfer2_droplist, axis = 1)
US_df_transfer2 = US_df_transfer2.fillna(0)

# print(US_df_transfer.notnull().values.all())
# print(US_df_transfer.isnull().values.any())

print(US_df_transfer2.columns)
print(US_df_transfer2.shape)
# US_df_transfer["ManDis"] = ""


US_df_test = pd.read_csv('Lima_mean.csv')
US_test_droplist = ['ID', 'Lat', 'Lon', 'year', 'month', 'doy']
US_df_test = US_df_test.drop(US_test_droplist, axis = 1)

print(US_df_test.columns)
print(US_df_test.shape)

######################### Finding best CA instances based on mean of Peru dataset values ################################################################

# Peru_df_mean = []
# prow = US_df_train.mean()
# Peru_df_mean = [prow.AOD550, prow.temp_2m, prow.rhum, prow.zonal_wind_10m, prow.merid_wind_10m, prow.surf_pres, prow.hpbl, prow.conv_prec, prow.NDVI, prow.DEM, prow.Population, prow.distance, prow.NLCD_Developed]
#
# rowidx = 0
# for row in US_df_transfer.itertuples():
#     row_list =[row.AOD550, row.temp_2m, row.rhum, row.zonal_wind_10m, row.merid_wind_10m, row.surf_pres, row.hpbl, row.conv_prec, row.NDVI, row.DEM, row.Population, row.distance, row.NLCD_Developed]
#     man_dis = 0
#     for i in range(0, len(row_list)):
#         tempval = Peru_df_mean[i] - row_list[i]
#         man_dis = man_dis + abs(tempval)
#
#     US_df_transfer.loc[rowidx,"ManDis"] = man_dis
#     print(US_df_transfer.loc[rowidx,"ManDis"])
#     rowidx = rowidx + 1
#
# US_df_transfer = US_df_transfer.sort_values('ManDis')
# US_df_transfer = US_df_transfer.head(4*len(US_df_train))
# US_df_transfer = US_df_transfer.drop(['ManDis'], axis =1)

# print(US_df_transfer)

target_column = ['pm25_value']
US_df_train_y = US_df_train[target_column]
US_df_train_X = US_df_train.drop(target_column, axis = 1)

target_column2 = ['PM25']
US_df_transfer_y = US_df_transfer[target_column]
US_df_transfer_X = US_df_transfer.drop(target_column, axis = 1)

US_df_transfer2_y = US_df_transfer2[target_column]
US_df_transfer2_X = US_df_transfer2.drop(target_column, axis = 1)

# scaler = StandardScaler()
# scaled_df_X = scaler.fit_transform(US_df_transfer_X)
# US_df_transfer_X = pd.DataFrame(scaled_df_X, columns = US_df_transfer_X.columns.tolist())

US_df_test_y = US_df_test[target_column2]
US_df_test_X = US_df_test.drop(target_column2, axis = 1)

TF_train_X = pd.concat([US_df_transfer_X, US_df_train_X], sort= False)
TF_train_X = pd.concat([US_df_transfer2_X, TF_train_X], sort= False)
TF_train_y = pd.concat([US_df_transfer_y, US_df_train_y], sort= False)
TF_train_y = pd.concat([US_df_transfer2_y, TF_train_y], sort= False)

##### You can comment this out ###########
TF_train_X.index = range(len(TF_train_X))
TF_train_y.index = range(len(TF_train_y))
###########################################

# TF_train_X.to_csv('CA_2.csv',index=False)

np_TF_train_X = TF_train_X.to_numpy()
np_TF_train_y = TF_train_y.to_numpy()

np_US_df_test_X = US_df_test_X.to_numpy()
np_US_df_test_y = US_df_test_y.to_numpy()

np_TF_train_y_list = np_TF_train_y.ravel()
np_US_df_test_y_list = np_US_df_test_y.ravel()

Peru_plot_df = pd.read_csv('Lima_mean.csv')

sample_size = [len(US_df_transfer2_X), len(US_df_transfer_X) + len(US_df_train_X)]
n_estimators = 100
steps = 30
fold = 10
random_state = np.random.RandomState(1)

# tuned_parameters = [{'n_estimators': [50, 100, 200], 'steps': [10, 20, 30, 40], 'fold': [10, 20, 30, 40]}]
# scores = ['precision', 'recall']

################################################TwoStageAdaBoostR2############################################################################
regr_1 = TwoStageTrAdaBoostR2( DecisionTreeRegressor(max_depth=6),
                      n_estimators = n_estimators, sample_size = sample_size,
                      steps = steps, fold = fold,
                      random_state = random_state)
# regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6))
#
# clf = GridSearchCV(estimator=regr_1, tuned_parameters)

regr_1.fit(np_TF_train_X, np_TF_train_y_list)
# print(clf.best_params_)


y_pred1 = regr_1.predict(np_US_df_test_X)

with open('trAdaBoostOutput_LIMA_2.txt', 'w') as filehandle1:
    for listitem in y_pred1:
        filehandle1.write('%s\n' % listitem)

# mse_twostageboost = sqrt(mean_squared_error(np_US_df_test_y_list, y_pred1))
# print("RMSE of TwoStageTrAdaboostR2:", mse_twostageboost)
#
# r2_score_twostageboost_values = pearsonr(np_US_df_test_y_list, y_pred1)
# r2_score_twostageboost = (r2_score_twostageboost_values[0])**2
# print("R^2 of TwoStageTrAdaboostR2:", r2_score_twostageboost)

################################################### Plotting the data. ######################################################################################

peru_map = gdp.read_file('Peru_shape2/PER_adm1.shp') #Upload the Peru map
fig, axs = plt.subplots(figsize = (30, 30))
peru_map.plot(ax = axs) #Create a plot for the peru map

Peru_plot_df['PM25_trAdaBoost'] = y_pred1


rainbow_colors = ListedColormap(['#0000ff', '#0054ff', '#00abff', '#00ffff', '#54ffab', '#abff53', '#ffff00', '#ffaa00', '#ff5400', '#ff0000']) #Using VIBGYOR for colors
axs.axis('off') #remove the axis
gdf_peru = gdp.GeoDataFrame(Peru_plot_df, geometry=gdp.points_from_xy(Peru_plot_df.Lon, Peru_plot_df.Lat)) #Create a geodataframe
gdf_peru.plot(column = 'PM25_trAdaBoost', ax=axs, cmap=rainbow_colors, legend = True) #Plot the geodatframe
plt.show()


############################################### Trying AdaBoost for regression ####################################################################

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),
                          n_estimators = n_estimators)
regr_2.fit(np_TF_train_X, np_TF_train_y_list)
y_pred2 = regr_2.predict(np_US_df_test_X)

with open('AdaBoostOutput_LIMA_2.txt', 'w') as filehandle2:
    for listitem in y_pred2:
        filehandle2.write('%s\n' % listitem)

# mse_adaboost = sqrt(mean_squared_error(np_US_df_test_y_list, y_pred2))
# print("RMSE of regular AdaboostR2:", mse_adaboost)
#
# r2_score_adaboost_values = pearsonr(np_US_df_test_y_list, y_pred2)
# r2_score_adaboost = (r2_score_adaboost_values[0])**2
# print("R^2 of AdaboostR2:", r2_score_adaboost)


############################################### Plotting the data.#####################################################################

peru_map = gdp.read_file('Peru_shape2/PER_adm1.shp') #Upload the Peru map
fig, axs = plt.subplots(figsize = (30, 30))
peru_map.plot(ax = axs) #Create a plot for the peru map

Peru_plot_df = Peru_plot_df.drop(['PM25'], axis =1)
Peru_plot_df['PM25_AdaBoost'] = y_pred2

rainbow_colors = ListedColormap(['#0000ff', '#0054ff', '#00abff', '#00ffff', '#54ffab', '#abff53', '#ffff00', '#ffaa00', '#ff5400', '#ff0000']) #Using VIBGYOR for colors
axs.axis('off') #remove the axis
gdf_peru = gdp.GeoDataFrame(Peru_plot_df, geometry=gdp.points_from_xy(Peru_plot_df.Lon, Peru_plot_df.Lat)) #Create a geodataframe
gdf_peru.plot(column = 'PM25_AdaBoost', ax=axs, cmap=rainbow_colors, legend = True) #Plot the geodatframe
plt.show()
