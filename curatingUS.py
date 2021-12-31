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

#Geo plotting libraries
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


sns.set_style("darkgrid")

########################################################################################################################################################
# US_df_train = pd.read_csv('US_data/Train/Train_3.csv')
# print(US_df_train.columns)
# US_df_transfer = pd.read_csv('US_data/Transfer/Transfer_3.csv')
#
# US_df_test = pd.read_csv('US_data/Test/Test_3.csv')
#
# US_complete = pd.read_csv('us.csv')
# print(US_complete.columns)
#
# US_complete_2011 = pd.read_csv('US_Monthly_2011.csv')
# print(US_complete_2011.columns)

dropcol_initial = ['name']

AutoData_df = pd.read_csv('UCI_regression/MPG/Auto.csv')
AutoData_df = AutoData_df.drop(dropcol_initial, axis = 1)

print("The shape of the Input data is: ", AutoData_df.shape) ## horsepower column has correlation 0.4 :: [46 - 230]
## Preprocessing Data
## Finding the correlation first
print("The correlation matrix is: ")
print(AutoData_df.corr())

col = AutoData_df['horsepower']
AutoData_df = AutoData_df.sort_values(by=['horsepower'])
print(AutoData_df[(col <= 80)].shape)
print(AutoData_df[(col > 80) & (col <= 110)].shape)
print(AutoData_df[(col > 110)].shape)
