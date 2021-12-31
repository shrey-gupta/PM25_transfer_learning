from __future__ import division
import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from pandas import DataFrame
import codecs,json
import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import random
from sklearn.metrics import mutual_info_score
import entropy_estimators as ee
from sklearn import preprocessing
from sklearn.feature_selection import f_regression, mutual_info_regression, RFE
from sklearn import preprocessing
from sklearn.decomposition import PCA #Used for PCA
from sklearn.preprocessing import StandardScaler #Importing the StandardScaler

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm

import pymrmr #This is for MRMR feature selection technique
from mlxtend.feature_selection import SequentialFeatureSelector as SFS #This is for wrapper methods for feature selection.
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


CA_df = pd.read_csv('us.csv')

#Drop a list of columns from the dataframe
drop_list = ["cmaq_id", "day", "month", "year", "rid", "lon", "lat", "Longitude", "Latitude"]
CA_df = CA_df.drop(drop_list, axis=1)
#print(CA_df.shape)


#Find all the columns in the dataframe and split them into features and target.
CA_columnlist = CA_df.columns.tolist()
CA_target_column = [CA_columnlist[0]] #Target Column is the Column 0
# CA_feature_column = CA_columnlist.drop(CA_target_column, axis=1)
CA_df_target = CA_df[CA_target_column]
CA_df_features = CA_df.drop(CA_target_column, axis=1)


#Mutual Inforamtion for regression dataset.
# MI = mutual_info_regression(CA_df_features, CA_df_target)
# print(MI)
#print(CA_df_features.columns.tolist())

##########################################Standardize or Normalize the features######################################################
#Standardizing the features
#Drop the 'fire' column first since it is categorical.
CA_df_features = CA_df_features.drop(['fire'], axis = 1)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(CA_df_features)
scaled_X = pd.DataFrame(scaled_X, columns = CA_df_features.columns.tolist())
scaled_X['fire'] = CA_df['fire']
#print(scaled_X.std(axis=0))
normalized_CA_df_features = scaled_X
# normalized_CA_df = normalized_CA_df_features


#Normalizing the features.
# normalized_X = preprocessing.normalize(CA_df_features)
# normalized_CA_df = pd.DataFrame(normalized_X, columns = CA_df_features.columns.tolist())
# normalized_CA_df_features = normalized_CA_df

#Include this when you want to include the target feature into the normalized feature set. You can do the same for standardized feature set.
# normalized_CA_df['PM25'] = CA_df_target

#Dropping all the features with correlation  > 0.8
# #Create correlation matrix
# corr_matrix = normalized_CA_df_features.corr().abs()
# #Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# #Find features with correlation greater than 0.80
# to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
# #Drop features
# # normalized_CA_df_features.drop(to_drop, axis=1, inplace=True)
# # print(normalized_CA_df_features.columns.tolist())
# print(to_drop)

#Normalizing the target variable as well
# normalized_X_target = preprocessing.normalize(CA_df_target)
# normalized_CA_df['PM25'] = normalized_X_target

#print(normalized_CA_df)
#print(CA_df.corr(method='pearson'))

#Mutual Information for the features with the Target Variable. The more the MI, the more relevant is the feature.
# MI = mutual_info_regression(normalized_CA_df_features, CA_df_target)
# data_tuples = list(zip(normalized_CA_df_features.columns.tolist(),MI))
# MI_df = pd.DataFrame(data_tuples, columns=['Features', 'Mutual Information'])
# MI_df = MI_df.sort_values(by=['Mutual Information'])
# print(MI_df)


# sel = SelectFromModel(LogisticRegression(C = 1, penalty = 'l1'))
# sel.fit(CA_df_features, CA_df_target)
# print(CA_df_features, CA_df_target)


#This code plots the correlation between the target and the remaining features.
# corr_pearson = normalized_CA_df[normalized_CA_df.columns[1:]].corr(method='pearson')['PM25'][:]
# print(corr_pearson.abs().sort_values())

##################################This code gives the correlation between each feature and sorts them in descending order.#############################################
# def top_correlation (df,n):
#     corr_matrix = df.corr(method='pearson')
#     correlation = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#                  .stack()
#                  .sort_values(ascending=False))
#     correlation = pd.DataFrame(correlation).reset_index()
#     correlation.columns=["Variable_1","Variable_2","Correlacion"]
#     correlation = correlation.reindex(correlation.Correlacion.abs().sort_values(ascending=False).index).reset_index().drop(["index"],axis=1)
#     return correlation.head(n)
#
# corrrelation_top = top_correlation(normalized_CA_df_features, 50) #Get top 50 rows for correaltion values.
# corrrelation_top = corrrelation_top.loc[corrrelation_top['Correlacion'] >= 0.9] #Get correlation values greater than 0.9.
# print(corrrelation_top)

##################################LASSO regression feature selection.############################################################
# pearson_drop_list = ['narr_pres2m', 'narr_pres30m', 'narr_pres10m', 'emissi11_pm10', 'emissi11_pm25', 'nldas_tmp2m', 'narr_tmp30m'] #, 'PM25']
# normalized_CA_df_features_pearson = normalized_CA_df_features.drop(pearson_drop_list, axis=1)

# reg = LassoCV()
# reg.fit(normalized_CA_df_features_pearson, CA_df_target)
# print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
# print("Best score using built-in LassoCV: %f" %reg.score(normalized_CA_df_features_pearson, CA_df_target))
# coef = pd.Series(reg.coef_, index = normalized_CA_df_features_pearson.columns)
# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# imp_coef = coef.sort_values()
# #import matplotlib
# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
# imp_coef.plot(kind = "barh")
# plt.title("Feature importance using Lasso Model")
# plt.show()

# print(normalized_CA_df_features.shape)

####################################Implementing MRMR feature selection technique#####################################################################
features_MRMR = pymrmr.mRMR(normalized_CA_df_features, 'MIQ', 20)
print(features_MRMR)

#####################################Wrapper method for Feature Selection#############################################################################
CA_df_features_wrapper = normalized_CA_df_features[features_MRMR]

lr = LinearRegression()
sfs = SFS(lr,
          k_features=10,
          forward=True,
          floating=False,
          scoring='neg_mean_squared_error',
          cv=10)
sfs = sfs.fit(CA_df_features_wrapper, CA_df_target)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

print(sfs.k_feature_names_)
print(sfs.k_score_)

#####################PCA for Feature Selection where importance of feature is plotted on a heat map.#################################
# pca = PCA()
# df_PCA_features = pca.fit_transform(normalized_CA_df_features)
# pca_expvr = pca.explained_variance_ratio_
# print(pca_expvr)
# cum_pca_expvr = np.cumsum(pca_expvr)
# print(cum_pca_expvr)
#
# plt.matshow(pca.components_,cmap='viridis')
# plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
# plt.colorbar()
# plt.xticks(range(len(CA_columnlist[1:])),CA_columnlist[1:],rotation=65,ha='left')
# plt.tight_layout()
# plt.show()

################PCA for Feature Selection where numerical values of feature importance is provided#########################(n_components=2)
# model = PCA(n_components=4).fit(normalized_CA_df_features)
# X_pc = model.transform(normalized_CA_df_features)
# #print(model.explained_variance_ratio_)
# print(abs(model.components_))


# n_pcs= model.components_.shape[0] #No. of components
# CA_feature_names = model.components_[0]
# print(model.components_[0])
# most_important_names  = []
#
# for i in range(10):
#     most_important_names.insert(i, CA_feature_names[i])
#
# print(most_important_names)
# most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
# CA_feature_names = CA_columnlist[1:]
# most_important_names = [CA_feature_names[most_important[i]] for i in range(n_pcs)]
# dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
# df_PC = pd.DataFrame(dic.items())
# print(df_PC)
