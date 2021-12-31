from __future__ import division
import pandas as pd
from pandas import DataFrame
import codecs,json
import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import numpy as np
import random
from sklearn.metrics import mutual_info_score
import pandas as pd
import entropy_estimators as ee
from sklearn import preprocessing
from sklearn.feature_selection import f_regression, mutual_info_regression
import continous as con


CA_df = pd.read_csv('CA.csv')

#Find all the columns in the dataframe and split them into features and target.
CA_columnlist = CA_df.columns.tolist()
CA_target_column = [CA_columnlist[0]]
CA_feature_column = CA_columnlist[1:14]

#Split the dataframe into target and features.
CA_df_target = CA_df[CA_target_column]
CA_df_features = CA_df[CA_feature_column]

print(CA_feature_column)
print(CA_target_column)
#Mutual Inforamtion for regression dataset.
# MI = mutual_info_regression(CA_df_features, CA_df_target)
# print(MI)

#print(con.get_h(CA_df_features['AOD550'], k=5))
print(CA_df_features['AOD550'])

########################################################################################################
#
# V = set(CA_feature_column) ## Constant set
# # print(V)
# T = set(CA_target_column) ## Constant set
# # print(T)
#
# ## New sets. CMB represent conditional Markov Blanket.
# CMB = set() ## Constant set
# Temp = set() ## Temporary set
# CMBTemp = set() ## Temporary set
# MB = set()
#
#
# ############################## GROWING PHASE ############################################################
#
# oldLen = len(CMB)
# newLen = len(CMB)
#
# Threshold = 0.01
#
# while True:
#     Temp.update(V)
#     # max = 0
#     max_MI = 0
#
#     for alpha in range(len(Temp)):
#         element = Temp.pop()
#         l1 = CA_df_features[element].tolist()
#         l2 = CA_df_target['PM25'].tolist()
#
#         m=[]
#         for i in l1:
#             m.append([i])
#         n=[]
#         for j in l2:
#             n.append([j])
#
#         #MI_temp = mutual_info_regression(l1, l2)
#         MI_temp = ee.mi(m,n)
#         print("The element is: ",element)
#         print("It's mututal information is: ", MI_temp)
#         if(MI_temp > max_MI):
#             max_MI_element = element
#             max_MI = MI_temp
#
#     #Now get the list for the element with Maximum MI with the Target. Find how they perform on CMI.
#     x1 = CA_df_features[max_MI_element].tolist()
#     x2 = CA_df_target['PM25'].tolist()
#     x=[]
#     for i in x1:
#         x.append([i])
#     y=[]
#     for j in x2:
#         y.append([j])
#
#     #CMBTemp.update(CMB)
#     CMScore = 1
#     hmax = 0
#     old_MB_length = len(MB) #Old MB length
#     print("Markov Blanket right now is: ", MB)
#
#     if len(MB)==0:
#         #Since there is no z (given), we calculate mutual information instead of the conditional mutual information.
#         #Add the element to th CMB if it is greater than threshold.
#         if max_MI > Threshold:
#             MB.add(max_MI_element)
#             Temp.discard(max_MI_element)
#
#     else:
#         CMB.update(MB)
#         for q in range(len(MB)):
#             cond_element = CMB.pop()
#             print(cond_element)
#             x3 = CA_df_features[cond_element].tolist()
#             z=[]
#             z.append(x3)
#
#         #Now that we have z, we can calculate conditional mututal inforamation.
#         #CMScore = ee.cmi(x,y,z)
#         # ent = con.get_h(z, k=5)
#         # MI1 = con.get_mi(x,z)
#         # MI2 = con.get_mi(y,z)
#         Half_Score = (MI1/ent)*MI2
#         CMScore = (ee.mi(y,x) - Half_Score)
#
#         print("CMS Score (CMI): ", CMScore)
#         if CMScore > Threshold:
#             MB.add(max_MI_element)
#             Temp.discard(max_MI_element)
#             # print("The max feature is :",maxInd)
#             # print(" The score is :",max)
#             newLen = newLen + 1
#             #print("New Length: ",newLen)
#             print(MB)
#
#     print("Markov Blanket after addition is: ", MB)
#
#     new_MB_length = len(MB)
#     if (new_MB_length == old_MB_length):
#         print("NO MORE FEATURES ADDED !!")
#         break
#
# print("The Current Markov Blanket after Growing phase is :",MB)
#
#
# ####################################### SHRINKING PHASE ######################################################
# #
# # CMBTemp = set()
# # min = 0
# # CMScore = 1
# # tempDict = {}
# # list4 = []
# # CMBTemp.update(CMB)
# #
# #
# # for alpha in range(len(CMBTemp)):
# # 	NCMBTemp = set()
# # 	CMScore = 1
# # 	ele = CMBTemp.pop()
# #
# # 	l1 = df1[ele].tolist()
# # 	l2 = df2['Z'].tolist()
# # 	x=[]
# # 	for i in l1:
# # 		x.append(i)
# # 	y=[]
# # 	for i in l2:
# # 		y.append(i)
# #
# # 	CMB.discard(ele)
# # 	NCMBTemp.update(CMB)
# #
# # 	hmax = 0
# #
# # 	if len(NCMBTemp)==0:
# # 		CMScore = ee.midd(x,y)
# #
# # 	else :
# # 		for q in range(len(NCMBTemp)):
# # 			condele = NCMBTemp.pop()
# # 			l3 = df1[condele].tolist()
# # 			z=[]
# # 			for i in l3:
# # 				z.append(i)
# # 			ent = ee.entropyd(z)
# # 			MI1 = ee.midd(x,z)
# # 			MI2 = ee.midd(y,z)
# # 			#print Score
# # 			Half_Score = (MI1/ent)*MI2
# #
# # 			if Half_Score > hmax :
# # 				hmax = Half_Score
# #
# #
# # 	CMScore = (ee.midd(y,x) - hmax)
# #
# # 	print("CMI for feature ",ele,"is :",CMScore)
# # 	CMB.add(ele)
# # 	tempDict[ele] = CMScore
# #
# #
# # c =1
# #
# # for item in tempDict.keys():
# # 	minVal = tempDict.get(item)
# # 	if minVal < Threshold :
# # 		CMB.discard(item)
# #
# # print("The Final Markov Blanket for the dataset is :",CMB)
