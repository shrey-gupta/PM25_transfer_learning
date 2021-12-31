import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

CA_df = pd.read_csv('us.csv')

#Drop a list of columns from the dataframe
drop_list = ["cmaq_id", "day", "month", "year", "rid", "lon", "lat", "Longitude", "Latitude"]
CA_df = CA_df.drop(drop_list, axis=1)
# #print(CA_df.shape)
#
#
# #Find all the columns in the dataframe and split them into features and target.
# CA_columnlist = CA_df.columns.tolist()
# CA_target_column = [CA_columnlist[0]] #Target Column is the Column 0
# # CA_feature_column = CA_columnlist.drop(CA_target_column, axis=1)
# CA_df_target = CA_df[CA_target_column]
# CA_df_features = CA_df.drop(CA_target_column, axis=1)
#
# sns.pairplot(CA_df)
# plt.show()
