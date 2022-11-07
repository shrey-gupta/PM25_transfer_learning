import pandas as pd
import sys
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from scipy.stats.stats import pearsonr
from math import sqrt
import statistics
from scipy.stats import *
from scipy.spatial import distance

############################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib.colors import ListedColormap
from matplotlib import cm
import geopandas as gpd
from pyproj import Proj, transform

print("Repositories uploaded!!")

#### import small training dataset
df_pm25small = pd.read_csv("US_data/BigUS/pm25small_lut.csv", index_col=None)
df_pm25small = df_pm25small.drop(['Unnamed: 0'], axis = 1)
df_pm25small['neighbor_location'] = ''

df_small_1 = df_pm25small[df_pm25small['day'] == 1]
df_small_1 = df_small_1.reset_index(drop = True)

inProj = Proj(init='epsg:3857')
outProj = Proj(init='epsg:4326')
lat, lon = transform(inProj, outProj, df_pm25small['pm_x'].to_numpy(), df_pm25small['pm_y'].to_numpy())
df_pm25small['lat_pm_x'] = lat
df_pm25small['lon_pm_y'] = lon

df_pm25small_cpy = df_small_1[0:0] ## create an empty dataframe

from sklearn.metrics.pairwise import haversine_distances
from math import radians


PI = 3.14

for dd in range(1,366):
    print(dd)

    df_small_dd = df_pm25small[df_pm25small['day'] == dd]
    df_small_dd = df_small_dd.reset_index(drop = True)
    print(df_small_dd.shape)

    for idx, row in df_small_dd.iterrows():
        df_small_drop = df_small_dd.drop([idx], axis=0)
        df_small_drop = df_small_drop.reset_index(drop = True)

        point_rd = [row['lat_pm_x'], row['lon_pm_y']]
        neighbors_list = []
        print("The points are :", [row['lat_pm_x'], row['lon_pm_y']])

        for ix, rval in df_small_drop.iterrows():
            neighbor_rad = [rval['lat_pm_x'], rval['lon_pm_y']]

            test_points = [point_rd, neighbor_rad]
            test_points_rad = np.array([[radians(x[0]), radians(x[1])] for x in test_points])

            hsine_res = haversine_distances([point_rd, neighbor_rad])
            hsine_res = hsine_res[0,1]* 6371000/1000

            if hsine_res <= 1500:
                neighbors_list.append([rval['lat_pm_x'], rval['lon_pm_y']])

        print("The list of neighbors are: ",neighbors_list)
        print("-----------------")
        df_small_dd.at[idx, 'neighbor_location'] = neighbors_list

    df_pm25small_cpy = pd.concat([df_pm25small_cpy, df_small_dd], axis=0)
    print("Added!")
    print("\n")

df_pm25small_cpy.to_csv('US_data/BigUS/pm25small_nn.csv', index=True) ##nearest neighbor file.
