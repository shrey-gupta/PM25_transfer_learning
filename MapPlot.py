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

from os import listdir
from os.path import isfile, join
import glob

#Geo plotting libraries
import geopandas as gdp
from matplotlib.colors import ListedColormap
import geoplot as glpt


sns.set_style("darkgrid")

# Read DOYs 002, 003, and 004 (each file should have 5959 pixels)
Lima_filenames = glob.glob("/home/shrey/Shrey/CausalityIP_Shrey_Navya/PM_25/2016/*.csv")
# Lima_2 = pd.read_csv('2016/Lima_Predictions_2016_002.csv')
# Lima_3 = pd.read_csv('2016/Lima_Predictions_2016_003.csv')
# Lima_4 = pd.read_csv('2016/Lima_Predictions_2016_004.csv')
# Lima = pd.read_csv('Lima_ID.csv')


collist = ['ID', 'Lat', 'Lon', 'AOD550']
Lima_files = []
i = 1
for f in Lima_filenames:
    df = pd.read_csv(f)
    Lima_files.append(df)
    print(i)
    i = i+1

# doy2 = Lima_2[collist]
# doy3 = Lima_3[collist]
# doy4 = Lima_4[collist]
# doy = Lima[collist]

# Combining into average
Lima_mean = pd.concat(Lima_files).groupby(level=0).mean()
Lima_mean.to_csv('Lima_mean.csv',index=False)

#Plotting the data.
# peru_map = gdp.read_file('Peru_shape2/PER_adm1.shp') #Upload the Peru map
# fig, axs = plt.subplots(figsize = (30, 30))
# peru_map.plot(ax = axs) #Create a plot for the peru map
#
# rainbow_colors = ListedColormap(['#0000ff', '#0054ff', '#00abff', '#00ffff', '#54ffab', '#abff53', '#ffff00', '#ffaa00', '#ff5400', '#ff0000']) #Using VIBGYOR for colors
# axs.axis('off') #remove the axis
# gdf_peru = gdp.GeoDataFrame(Lima_mean, geometry=gdp.points_from_xy(Lima_mean.Lon, Lima_mean.Lat)) #Create a geodataframe
# gdf_peru.plot(column = 'AOD550', ax=axs, cmap=rainbow_colors, legend = True) #Plot the geodatframe
# plt.show()
