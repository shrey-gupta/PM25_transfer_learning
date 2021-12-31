################################### poleTelecomm ################################################################
## 'nox' found to be correlated at 0.4 :: [0.385 - 0.871] :: 50
#################################################################################################################################
poleTelecommData_df = pd.read_csv('UCI_regression/BostonpoleTelecomm/BostonpoleTelecomm.csv')
print("poleTelecomm Data")
print(poleTelecommData_df.shape)

drop_col_poleTelecomm = ['nox']
poleTelecomm_tgt_df = poleTelecommData_df.loc[(poleTelecommData_df['nox'] > 0.475) & (poleTelecommData_df['nox'] <= 0.600)]
poleTelecomm_tgt_df = poleTelecomm_tgt_df.drop(drop_col_poleTelecomm, axis = 1)
poleTelecomm_tgt_df = poleTelecomm_tgt_df.reset_index(drop=True)
print("Target Set: ",poleTelecomm_tgt_df.shape)


poleTelecomm_source1_df = poleTelecommData_df.loc[(poleTelecommData_df['nox'] > 0.600)]
poleTelecomm_source1_df = poleTelecomm_source1_df.drop(drop_col_poleTelecomm, axis = 1)
poleTelecomm_source1_df = poleTelecomm_source1_df.reset_index(drop=True)
print("Source Set 1: ",poleTelecomm_source1_df.shape)


poleTelecomm_source2_df = poleTelecommData_df.loc[(poleTelecommData_df['nox'] <= 0.475)]
poleTelecomm_source2_df = poleTelecomm_source2_df.drop(drop_col_poleTelecomm, axis = 1)
poleTelecomm_source2_df = poleTelecomm_source2_df.reset_index(drop=True)
print("Source Set 2: ",poleTelecomm_source2_df.shape)


poleTelecomm_source_df = pd.concat([poleTelecomm_source1_df, poleTelecomm_source2_df], ignore_index=True)
print("Final Source Set: ",poleTelecomm_source_df.shape)


#################### Splitting into features and target ####################
target_column_poleTelecomm = ['medv']

poleTelecomm_tgt_df_y = poleTelecomm_tgt_df[target_column_poleTelecomm]
poleTelecomm_tgt_df_X = poleTelecomm_tgt_df.drop(target_column_poleTelecomm, axis = 1)

poleTelecomm_source_df_y = poleTelecomm_source_df[target_column_poleTelecomm]
poleTelecomm_source_df_X = poleTelecomm_source_df.drop(target_column_poleTelecomm, axis = 1)

print("---------------------------")

########################### Transfer Learning poleTelecomm #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}


print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_poleTelecomm = []
rmselist_AdaTL_poleTelecomm = []

r2scorelist_Ada_poleTelecomm = []
rmselist_Ada_poleTelecomm = []

r2scorelist_KMM_poleTelecomm = []
rmselist_KMM_poleTelecomm = []

r2scorelist_GBRTL_poleTelecomm = []
rmselist_GBRTL_poleTelecomm = []

r2scorelist_GBR_poleTelecomm = []
rmselist_GBR_poleTelecomm = []

r2scorelist_TwoTrAda_poleTelecomm = []
rmselist_TwoTrAda_poleTelecomm = []

r2scorelist_stradaboost_poleTelecomm = []
rmselist_stradaboost_poleTelecomm = []



kfold = KFold(n_splits = 10, random_state=42, shuffle=False)

for train_ix, test_ix in kfold.split(poleTelecomm_train_df_X):
    ############### get data ###############
    poleTelecomm_test_df_X, poleTelecomm_tgt_df_X  = poleTelecomm_train_df_X.iloc[train_ix], poleTelecomm_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    poleTelecomm_test_df_y, poleTelecomm_tgt_df_y  = poleTelecomm_train_df_y.iloc[train_ix], poleTelecomm_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(poleTelecomm_tgt_df_X.shape, poleTelecomm_test_df_X.shape)

    ############### Merging the datasets ##########################################
    poleTelecomm_X_df = pd.concat([poleTelecomm_tgt_df_X, poleTelecomm_source_df_X], ignore_index=True)
    poleTelecomm_y_df = pd.concat([poleTelecomm_tgt_df_y, poleTelecomm_source_df_y], ignore_index=True)

    poleTelecomm_np_train_X = poleTelecomm_X_df.to_numpy()
    poleTelecomm_np_train_y = poleTelecomm_y_df.to_numpy()

    poleTelecomm_np_test_X = poleTelecomm_test_df_X.to_numpy()
    poleTelecomm_np_test_y = poleTelecomm_test_df_y.to_numpy()

    poleTelecomm_np_train_y_list = poleTelecomm_np_train_y.ravel()
    poleTelecomm_np_test_y_list = poleTelecomm_np_test_y.ravel()

    src_size_poleTelecomm = len(poleTelecomm_source_df_y)
    tgt_size_poleTelecomm = len(poleTelecomm_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_poleTelecomm - 1), step=1)
    tgt_idx = np.arange(start=src_size_poleTelecomm, stop=((src_size_poleTelecomm + tgt_size_poleTelecomm)-1), step=1)



    ################### AdaBoost Tl ###################
    model_AdaTL_poleTelecomm = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_AdaTL_poleTelecomm.fit(poleTelecomm_np_train_X, poleTelecomm_np_train_y_list)

    y_pred_AdaTL_poleTelecomm = model_AdaTL_poleTelecomm.predict(poleTelecomm_np_test_X)

    mse_AdaTL_poleTelecomm = sqrt(mean_squared_error(poleTelecomm_np_test_y, y_pred_AdaTL_poleTelecomm))
    rmselist_AdaTL_poleTelecomm.append(mse_AdaTL_poleTelecomm)

    r2_score_AdaTL_poleTelecomm = pearsonr(poleTelecomm_np_test_y_list, y_pred_AdaTL_poleTelecomm)
    r2_score_AdaTL_poleTelecomm = (r2_score_AdaTL_poleTelecomm[0])**2
    r2scorelist_AdaTL_poleTelecomm.append(r2_score_AdaTL_poleTelecomm)


    ################### AdaBoost ###################
    model_Ada_poleTelecomm = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_Ada_poleTelecomm.fit(poleTelecomm_tgt_df_X, poleTelecomm_tgt_df_y)

    y_pred_Ada_poleTelecomm = model_Ada_poleTelecomm.predict(poleTelecomm_np_test_X)

    mse_Ada_poleTelecomm = sqrt(mean_squared_error(poleTelecomm_np_test_y, y_pred_Ada_poleTelecomm))
    rmselist_Ada_poleTelecomm.append(mse_Ada_poleTelecomm)

    r2_score_Ada_poleTelecomm = pearsonr(poleTelecomm_np_test_y_list, y_pred_Ada_poleTelecomm)
    r2_score_Ada_poleTelecomm = (r2_score_Ada_poleTelecomm[0])**2
    r2scorelist_Ada_poleTelecomm.append(r2_score_Ada_poleTelecomm)


   ################### KMM ###################
    model_KMM_poleTelecomm = KMM(get_estimator = get_estimator)
    model_KMM_poleTelecomm.fit(poleTelecomm_np_train_X, poleTelecomm_np_train_y_list, src_idx_poleTelecomm, tgt_idx_poleTelecomm)

    y_pred_KMM_poleTelecomm = model_KMM_poleTelecomm.predict(poleTelecomm_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_poleTelecomm = sqrt(mean_squared_error(poleTelecomm_np_test_y, y_pred_KMM_poleTelecomm))
    rmselist_KMM_poleTelecomm.append(mse_KMM_poleTelecomm)

    r2_score_KMM_poleTelecomm = pearsonr(poleTelecomm_np_test_y_list, y_pred_KMM_poleTelecomm)
    r2_score_KMM_poleTelecomm = (r2_score_KMM_poleTelecomm[0])**2
    r2scorelist_KMM_poleTelecomm.append(r2_score_KMM_poleTelecomm)


    ################### GBRTL ###################
    model_GBRTL_poleTelecomm = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBRTL_poleTelecomm.fit(poleTelecomm_np_train_X, poleTelecomm_np_train_y_list)

    y_pred_GBRTL_poleTelecomm = model_GBRTL_poleTelecomm.predict(poleTelecomm_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_poleTelecomm = sqrt(mean_squared_error(poleTelecomm_np_test_y, y_pred_GBRTL_poleTelecomm))
    rmselist_GBRTL_poleTelecomm.append(mse_GBRTL_poleTelecomm)

    r2_score_GBRTL_poleTelecomm = pearsonr(poleTelecomm_np_test_y_list, y_pred_GBRTL_poleTelecomm)
    r2_score_GBRTL_poleTelecomm = (r2_score_GBRTL_poleTelecomm[0])**2
    r2scorelist_GBRTL_poleTelecomm.append(r2_score_GBRTL_poleTelecomm)


    ################### GBR ###################
    model_GBR_poleTelecomm = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBR_poleTelecomm.fit(poleTelecomm_tgt_df_X, poleTelecomm_tgt_df_y)

    y_pred_GBR_poleTelecomm = model_GBR_poleTelecomm.predict(poleTelecomm_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_poleTelecomm = sqrt(mean_squared_error(poleTelecomm_np_test_y, y_pred_GBR_poleTelecomm))
    rmselist_GBR_poleTelecomm.append(mse_GBR_poleTelecomm)

    r2_score_GBR_poleTelecomm = pearsonr(poleTelecomm_np_test_y_list, y_pred_GBR_poleTelecomm)
    r2_score_GBR_poleTelecomm = (r2_score_GBR_poleTelecomm[0])**2
    r2scorelist_GBR_poleTelecomm.append(r2_score_GBR_poleTelecomm)


    ################### Two-TrAdaBoost ###################
    model_TwoTrAda_poleTelecomm = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 1000, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_poleTelecomm.fit(poleTelecomm_np_train_X, poleTelecomm_np_train_y_list, src_idx_poleTelecomm, tgt_idx_poleTelecomm)

    y_pred_TwoTrAda_poleTelecomm = model_TwoTrAda_poleTelecomm.predict(poleTelecomm_np_test_X)

    mse_TwoTrAda_poleTelecomm = sqrt(mean_squared_error(poleTelecomm_np_test_y, y_pred_TwoTrAda_poleTelecomm))
    rmselist_TwoTrAda_poleTelecomm.append(mse_TwoTrAda_poleTelecomm)

    r2_score_TwoTrAda_poleTelecomm = pearsonr(poleTelecomm_np_test_y_list, y_pred_TwoTrAda_poleTelecomm)
    r2_score_TwoTrAda_poleTelecomm = (r2_score_TwoTrAda_poleTelecomm[0])**2
    r2scorelist_TwoTrAda_poleTelecomm.append(r2_score_TwoTrAda_poleTelecomm)


    ################### STrAdaBoost ###################
    sample_size = [len(poleTelecomm_tgt_df_X), len(poleTelecomm_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    model_stradaboost_poleTelecomm = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_poleTelecomm.fit(poleTelecomm_np_train_X, poleTelecomm_np_train_y_list)
    y_pred_stradaboost_poleTelecomm = model_stradaboost_poleTelecomm.predict(poleTelecomm_np_test_X)


    mse_stradaboost_poleTelecomm = sqrt(mean_squared_error(poleTelecomm_np_test_y, y_pred_stradaboost_poleTelecomm))
    rmselist_stradaboost_poleTelecomm.append(mse_stradaboost_poleTelecomm)

    r2_score_stradaboost_poleTelecomm = pearsonr(poleTelecomm_np_test_y_list, y_pred_stradaboost_poleTelecomm)
    r2_score_stradaboost_poleTelecomm = (r2_score_stradaboost_poleTelecomm[0])**2
    r2scorelist_stradaboost_poleTelecomm.append(r2_score_stradaboost_poleTelecomm)


with open('poleTelecomm_rmse.txt', 'w') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in rmselist_AdaTL_poleTelecomm)

with open('poleTelecomm_r2.txt', 'w') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_poleTelecomm)

######################################################################################

with open('poleTelecomm_rmse.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in rmselist_Ada_poleTelecomm)

with open('poleTelecomm_r2.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in r2scorelist_Ada_poleTelecomm)

######################################################################################

with open('poleTelecomm_rmse.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in rmselist_KMM_poleTelecomm)

with open('poleTelecomm_r2.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in r2scorelist_KMM_poleTelecomm)


######################################################################################

with open('poleTelecomm_rmse.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in rmselist_GBRTL_poleTelecomm)

with open('poleTelecomm_r2.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_poleTelecomm)


######################################################################################

with open('poleTelecomm_rmse.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in rmselist_GBR_poleTelecomm)

with open('poleTelecomm_r2.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in r2scorelist_GBR_poleTelecomm)


######################################################################################

with open('poleTelecomm_rmse.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_poleTelecomm)

with open('poleTelecomm_r2.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_poleTelecomm)

######################################################################################

with open('poleTelecomm_rmse.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in rmselist_stradaboost_poleTelecomm)

with open('poleTelecomm_r2.txt', 'a') as poleTelecomm_handle:
    poleTelecomm_handle.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_poleTelecomm)


######################################################################################

print("-------------------------------------------")
