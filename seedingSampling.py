from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2
# from newtwoStage_TrAdaBoostR2 import TradaboostRegressor

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

################# Geo plotting libraries
import geopandas as gdp
from matplotlib.colors import ListedColormap
import geoplot as glpt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor

from statistics import mean
from sklearn.cluster import KMeans
from scipy.spatial import distance


sns.set_style("darkgrid")

us_df = pd.read_csv('us.csv')
print(us_df.shape)
# print(us_df.columns)
us_df = us_df.sort_values(by=['rid'])

droplist = ['Latitude', 'Longitude','cmaq_id', 'lon', 'lat', 'day', 'year', 'month']
us_df = us_df.drop(droplist, axis = 1)
print(us_df.columns)

Target_df  = us_df.loc[us_df['rid'] == 9]
Source_df = us_df.loc[us_df['rid'] == 5]
print(Target_df.shape)
print(Source_df.shape)

drop_rid = ['rid']
Target_df  = Target_df.drop(drop_rid, axis = 1)
Source_df = Source_df.drop(drop_rid, axis = 1)

target_column = ['pm25_value']
Target_df_y = Target_df[target_column]
Target_df_X = Target_df.drop(target_column, axis = 1)

y_source = Source_df[target_column]
X_source = Source_df.drop(target_column, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(Target_df_X, Target_df_y, test_size=0.7, random_state=1)


#################################################################################
kmeans = KMeans(n_clusters = 1500, random_state=0).fit(X_source)
# print(kmeans.cluster_centers_)

Alternate_df = X_train.copy()
idxlist = []
Alternate_df_np = Alternate_df.to_numpy()

new_df_list = []


for rowkm in kmeans.cluster_centers_:
    mindist = -99
    rowidx = 0
    idx = 0
    for row in Alternate_df_np:
        dst = distance.euclidean(row, rowkm)

        if(dst >= mindist):
            mindist = dst
            rowidx = idx
            rowval = row

        idx = idx + 1

    print("Row selected: ", rowidx) #Alternate_df.loc[rowidx,:]\
    print("Min. distance: ", mindist)
    print("Matrix shape: ", Alternate_df_np.shape)
    new_df_list.append(rowval)
    Alternate_df_np = np.delete(Alternate_df_np, rowidx, 0)
    idxlist.append(rowidx)


new_df = pd.DataFrame(np.vstack(new_df_list))
# print(new_df)
print(new_df.shape)
# print(new_df.dtypes)

##################################################################################

alt_source_df = X_source.copy()
idxlist2 = []
alt_source_df_np = alt_source_df.to_numpy()

final_df_list = []

for row_nm in new_df_list:
    min_dist = -99
    row_idx = 0
    idx_val = 0
    for row_alt in alt_source_df_np:
        dst = distance.euclidean(row_alt, row_nm)
        if(dst >= mindist):
            min_dist = dst
            row_idx = idx_val
            row_val = row_alt

        idx_val = idx_val + 1

    print("Row selected: ", row_idx) #Alternate_df.loc[rowidx,:]\
    print("Min. distance: ", min_dist)
    print("Matrix shape: ", alt_source_df_np.shape)
    final_df_list.append(row_val)
    alt_source_df_np = np.delete(alt_source_df_np, row_idx, 0)
    idxlist2.append(row_idx)


final_df = pd.DataFrame(np.vstack(final_df_list), columns= X_source.columns)
print(final_df)
print(final_df.shape)

print("Shape before :",X_source.shape)
X_source = df = pd.concat([X_source, final_df])
X_source = X_source.drop_duplicates(keep=False)
print("Shape after :",X_source.shape)

#############################################################################################
