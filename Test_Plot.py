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
# import geoplot as glpt

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
import statistics

from matplotlib.transforms import Affine2D
import folium
import matplotlib.lines as mlines


# sns.set_style("darkgrid")
#######################################################################################################

methods = ['S-TrAdaBoost.R2', 'two-stage TrAdaBoost.R2', 'AdaBoost.R2']

# STrAdaBoost_R2_val = (max(STrAdaBoost_R2) - min(STrAdaBoost_R2))/2
# AdaBoost_R2_val = (max(AdaBoost_R2) - min(AdaBoost_R2))/2
# TrAdaBoost_R2_val = (max(TrAdaBoost_R2) - min(TrAdaBoost_R2))/2

STrAdaBoost_std1 = statistics.pstdev(STrAdaBoost_RMSE_NOx)
TrAdaBoost_std1 = statistics.pstdev(TrAdaBoost_RMSE_NOx)
AdaBoost_std1 = statistics.pstdev(AdaBoost_RMSE_NOx)

STrAdaBoost_std2 = statistics.pstdev(STrAdaBoost_RMSE_NO2)
TrAdaBoost_std2 = statistics.pstdev(TrAdaBoost_RMSE_NO2)
AdaBoost_std2 = statistics.pstdev(AdaBoost_RMSE_NO2)

STrAdaBoost_std3 = statistics.pstdev(STrAdaBoost_RMSE_O3)
TrAdaBoost_std3 = statistics.pstdev(TrAdaBoost_RMSE_O3)
AdaBoost_std3 = statistics.pstdev(AdaBoost_RMSE_O3)

# STrAdaBoost_std3 = statistics.pstdev(STrAdaBoost_R2_NOx_Norm)
# TrAdaBoost_std3 = statistics.pstdev(TrAdaBoost_R2_NOx_Norm)
# AdaBoost_std3 = statistics.pstdev(AdaBoost_R2_NOx_Norm)
#
# STrAdaBoost_std4 = statistics.pstdev(STrAdaBoost_R2_NOx_Norm)
# TrAdaBoost_std4 = statistics.pstdev(TrAdaBoost_R2_NOx_Norm)
# AdaBoost_std4 = statistics.pstdev(AdaBoost_R2_NOx_Norm)


y1 = [mean(STrAdaBoost_RMSE_NOx), mean(TrAdaBoost_RMSE_NOx), mean(AdaBoost_RMSE_NOx)]
dy1 = [STrAdaBoost_std1, TrAdaBoost_std1, AdaBoost_std1]


y2 = [mean(STrAdaBoost_RMSE_NO2), mean(TrAdaBoost_RMSE_NO2), mean(AdaBoost_RMSE_NO2)]
dy2 = [STrAdaBoost_std2, TrAdaBoost_std2, AdaBoost_std2]

y3 = [mean(STrAdaBoost_RMSE_O3), mean(TrAdaBoost_RMSE_O3), mean(AdaBoost_RMSE_O3)]
dy3 = [STrAdaBoost_std3, TrAdaBoost_std3, AdaBoost_std3]

# y3 = [mean(STrAdaBoost_R2_NOx_Norm), mean(TrAdaBoost_R2_NOx_Norm), mean(AdaBoost_R2_NOx_Norm)]
# dy3 = [STrAdaBoost_std3, TrAdaBoost_std3, AdaBoost_std3]
#
# y4 = [mean(STrAdaBoost_R2_NO2_Norm), mean(TrAdaBoost_R2_NO2_Norm), mean(AdaBoost_R2_NO2_Norm)]
# dy4 = [STrAdaBoost_std4, TrAdaBoost_std4, AdaBoost_std4]

plus = mlines.Line2D([], [], marker='P', color='maroon', linestyle='None',
                          markersize = 20, label='NO2')
circle = mlines.Line2D([], [], marker='o', color='black', linestyle='None',
                          markersize = 20, label='NOx')
triangle = mlines.Line2D([], [], marker='v', color='blue', linestyle='None',
                          markersize = 20, label='O3')


x = np.arange(len(methods))

print(dy1)
print(dy2)
print(dy3)

fig, ax = plt.subplots()

# trans1 = Affine2D().translate(-0.025, 0.0) + ax.transData
# trans2 = Affine2D().translate(+0.025, 0.0) + ax.transData
# trans3 = Affine2D().translate(+0.05, 0.0) + ax.transData
# trans4 = Affine2D().translate(-1.5, 0.0) + ax.transData

# er1 = ax.errorbar(x, y1, yerr = dy1, marker = "o", linestyle = "None", transform = trans1, elinewidth = 10, markersize = 20)
# er2 = ax.errorbar(x, y2, yerr = dy2, marker = "s", linestyle = "None", transform = trans2, elinewidth = 6, markersize = 10)
# er3 = ax.errorbar(x, y3, yerr = dy3, marker = "o", linestyle = "None", transform = trans3)
# er4 = ax.errorbar(x, y4, yerr = dy4, marker = "o", linestyle = ":", transform = trans4)


plt.errorbar(x, y1, yerr = dy1, fmt='o', color='black', ecolor='black', elinewidth = 10, markersize = 20) #capsize=0)
plt.errorbar(x, y2, yerr = dy2, fmt='P', color='maroon', ecolor='maroon', elinewidth = 10, markersize = 20)
plt.errorbar(x, y3, yerr = dy3, fmt='v', color='blue', ecolor='blue', elinewidth = 10, markersize = 20) #capsize=0)
# plt.xticks(x)
plt.xticks(np.arange(3), ('S-TrAdaBoost.R2', 'two-stage TrAdaBoost.R2', 'AdaBoost.R2'), fontsize = 32)
plt.yticks(fontsize = 30)
plt.ylabel("RMS Error", fontsize = 36)
# plt.title('Mean RMSE for methodologies with Standard deviation error bars', fontsize=12)
plt.legend(handles= [plus, circle, triangle], fontsize = 20, loc= 'center left')

plt.show()

# fig, ax = plt.subplots()
# ax.bar(x, y, yerr=dy, align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.yticks(np.arange(0.25, 0.40, 0.02))
# ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
# ax.set_xticks(x)
# ax.set_xticklabels(methods)
# ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()
