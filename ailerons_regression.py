################################################################ Ailerons ###############################################
target_ailerons = ['goal']
colnames_ailerons = ['climbRate', 'Sgz', 'p', 'q', 'curPitch', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate', 'diffDiffClb', 'SeTime1',
            'SeTime2', 'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', 'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12',
            'SeTime13', 'SeTime14', 'diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4', 'diffSeTime5', 'diffSeTime6', 'diffSeTime7',
            'diffSeTime8', 'diffSeTime9', 'diffSeTime10', 'diffSeTime11', 'diffSeTime12', 'diffSeTime13', 'diffSeTime14', 'alpha', 'Se', 'goal']
AileronsData_train_df = pd.read_csv('UCI_regression/Ailerons/ailerons.data', header = None, names = colnames_ailerons)
print("Ailerons Data")
# print(AileronsData_df_train.shape)
print("-------------------------------------------")

AileronsData_test_df = pd.read_csv('UCI_regression/Ailerons/ailerons.test', header = None, names = colnames_ailerons)
# print(AileronsData_df_test.shape)

#################### Splitting with small target set and large source and test set #############
AileronsData_source_df, AileronsData_tgt_df = train_test_split(AileronsData_train_df, test_size = 0.05) ## test_size = tgt size
# print(AileronsData_df_tgt.shape, AileronsData_df_source.shape, AileronsData_df_test.shape)

AileronsData_train_df = AileronsData_train_df.reset_index(drop = True)
AileronsData_tgt_df = AileronsData_tgt_df.reset_index(drop = True)
AileronsData_source_df = AileronsData_source_df.reset_index(drop = True)
print("Target Set: ", AileronsData_tgt_df.shape)
print("Source Set: ", AileronsData_source_df.shape)
print("Test Set: ", AileronsData_test_df.shape)


AileronsData_df_test_y = AileronsData_test_df[target_ailerons]
AileronsData_df_test_X = AileronsData_test_df.drop(target_ailerons, axis = 1)

AileronsData_df_tgt_y = AileronsData_tgt_df[target_ailerons]
AileronsData_df_tgt_X = AileronsData_tgt_df.drop(target_ailerons, axis = 1)

AileronsData_df_source_y = AileronsData_source_df[target_ailerons]
AileronsData_df_source_X = AileronsData_source_df.drop(target_ailerons, axis = 1)

#################### Splitting into features and target ####################
target_column_ailerons = ['mpg']

ailerons_tgt_df_y = ailerons_tgt_df[target_column_ailerons]
ailerons_tgt_df_X = ailerons_tgt_df.drop(target_column_ailerons, axis = 1)

ailerons_source_df_y = ailerons_source_df[target_column_ailerons]
ailerons_source_df_X = ailerons_source_df.drop(target_column_ailerons, axis = 1)

ailerons_source_df = pd.concat([ailerons_source1_df, ailerons_source2_df], ignore_index=True)
print("Final Source Set: ",ailerons_source_df.shape)

print("---------------------------")

########################### Transfer Learning ailerons #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}


print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_ailerons = []
rmselist_AdaTL_ailerons = []

r2scorelist_Ada_ailerons = []
rmselist_Ada_ailerons = []

r2scorelist_KMM_ailerons = []
rmselist_KMM_ailerons = []

r2scorelist_GBRTL_ailerons = []
rmselist_GBRTL_ailerons = []

r2scorelist_GBR_ailerons = []
rmselist_GBR_ailerons = []

r2scorelist_TwoTrAda_ailerons = []
rmselist_TwoTrAda_ailerons = []

r2scorelist_stradaboost_ailerons = []
rmselist_stradaboost_ailerons = []



kfold = KFold(n_splits = 10, random_state=42, shuffle=False)

for train_ix, test_ix in kfold.split(ailerons_train_df_X):
    ############### get data ###############
    ailerons_test_df_X, ailerons_tgt_df_X  = ailerons_train_df_X.iloc[train_ix], ailerons_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    ailerons_test_df_y, ailerons_tgt_df_y  = ailerons_train_df_y.iloc[train_ix], ailerons_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(ailerons_tgt_df_X.shape, ailerons_test_df_X.shape)

    ############### Merging the datasets ##########################################
    ailerons_X_df = pd.concat([ailerons_tgt_df_X, ailerons_source_df_X], ignore_index=True)
    ailerons_y_df = pd.concat([ailerons_tgt_df_y, ailerons_source_df_y], ignore_index=True)

    ailerons_np_train_X = ailerons_X_df.to_numpy()
    ailerons_np_train_y = ailerons_y_df.to_numpy()

    ailerons_np_test_X = ailerons_test_df_X.to_numpy()
    ailerons_np_test_y = ailerons_test_df_y.to_numpy()

    ailerons_np_train_y_list = ailerons_np_train_y.ravel()
    ailerons_np_test_y_list = ailerons_np_test_y.ravel()

    src_size_ailerons = len(ailerons_source_df_y)
    tgt_size_ailerons = len(ailerons_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_ailerons - 1), step=1)
    tgt_idx = np.arange(start=src_size_ailerons, stop=((src_size_ailerons + tgt_size_ailerons)-1), step=1)



    ################### AdaBoost Tl ###################
    model_AdaTL_ailerons = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_AdaTL_ailerons.fit(ailerons_np_train_X, ailerons_np_train_y_list)

    y_pred_AdaTL_ailerons = model_AdaTL_ailerons.predict(ailerons_np_test_X)

    mse_AdaTL_ailerons = sqrt(mean_squared_error(ailerons_np_test_y, y_pred_AdaTL_ailerons))
    rmselist_AdaTL_ailerons.append(mse_AdaTL_ailerons)

    r2_score_AdaTL_ailerons = pearsonr(ailerons_np_test_y_list, y_pred_AdaTL_ailerons)
    r2_score_AdaTL_ailerons = (r2_score_AdaTL_ailerons[0])**2
    r2scorelist_AdaTL_ailerons.append(r2_score_AdaTL_ailerons)


    ################### AdaBoost ###################
    model_Ada_ailerons = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_Ada_ailerons.fit(ailerons_tgt_df_X, ailerons_tgt_df_y)

    y_pred_Ada_ailerons = model_Ada_ailerons.predict(ailerons_np_test_X)

    mse_Ada_ailerons = sqrt(mean_squared_error(ailerons_np_test_y, y_pred_Ada_ailerons))
    rmselist_Ada_ailerons.append(mse_Ada_ailerons)

    r2_score_Ada_ailerons = pearsonr(ailerons_np_test_y_list, y_pred_Ada_ailerons)
    r2_score_Ada_ailerons = (r2_score_Ada_ailerons[0])**2
    r2scorelist_Ada_ailerons.append(r2_score_Ada_ailerons)


   ################### KMM ###################
    model_KMM_ailerons = KMM(get_estimator = get_estimator)
    model_KMM_ailerons.fit(ailerons_np_train_X, ailerons_np_train_y_list, src_idx_ailerons, tgt_idx_ailerons)

    y_pred_KMM_ailerons = model_KMM_ailerons.predict(ailerons_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_ailerons = sqrt(mean_squared_error(ailerons_np_test_y, y_pred_KMM_ailerons))
    rmselist_KMM_ailerons.append(mse_KMM_ailerons)

    r2_score_KMM_ailerons = pearsonr(ailerons_np_test_y_list, y_pred_KMM_ailerons)
    r2_score_KMM_ailerons = (r2_score_KMM_ailerons[0])**2
    r2scorelist_KMM_ailerons.append(r2_score_KMM_ailerons)

    ################### GBRTL ###################
    model_GBRTL_ailerons = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBRTL_ailerons.fit(ailerons_np_train_X, ailerons_np_train_y_list)

    y_pred_GBRTL_ailerons = model_GBRTL_ailerons.predict(ailerons_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_ailerons = sqrt(mean_squared_error(ailerons_np_test_y, y_pred_GBRTL_ailerons))
    rmselist_GBRTL_ailerons.append(mse_GBRTL_ailerons)

    r2_score_GBRTL_ailerons = pearsonr(ailerons_np_test_y_list, y_pred_GBRTL_ailerons)
    r2_score_GBRTL_ailerons = (r2_score_GBRTL_ailerons[0])**2
    r2scorelist_GBRTL_ailerons.append(r2_score_GBRTL_ailerons)

    ################### GBR ###################
    model_GBR_ailerons = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBR_ailerons.fit(ailerons_tgt_df_X, ailerons_tgt_df_y)

    y_pred_GBR_ailerons = model_GBR_ailerons.predict(ailerons_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_ailerons = sqrt(mean_squared_error(ailerons_np_test_y, y_pred_GBR_ailerons))
    rmselist_GBR_ailerons.append(mse_GBR_ailerons)

    r2_score_GBR_ailerons = pearsonr(ailerons_np_test_y_list, y_pred_GBR_ailerons)
    r2_score_GBR_ailerons = (r2_score_GBR_ailerons[0])**2
    r2scorelist_GBR_ailerons.append(r2_score_GBR_ailerons)

    ################### Two-TrAdaBoost ###################
    model_TwoTrAda_ailerons = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 1000, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_ailerons.fit(ailerons_np_train_X, ailerons_np_train_y_list, src_idx_ailerons, tgt_idx_ailerons)

    y_pred_TwoTrAda_ailerons = model_TwoTrAda_ailerons.predict(ailerons_np_test_X)

    mse_TwoTrAda_ailerons = sqrt(mean_squared_error(ailerons_np_test_y, y_pred_TwoTrAda_ailerons))
    rmselist_TwoTrAda_ailerons.append(mse_TwoTrAda_ailerons)

    r2_score_TwoTrAda_ailerons = pearsonr(ailerons_np_test_y_list, y_pred_TwoTrAda_ailerons)
    r2_score_TwoTrAda_ailerons = (r2_score_TwoTrAda_ailerons[0])**2
    r2scorelist_TwoTrAda_ailerons.append(r2_score_TwoTrAda_ailerons)

    ################### STrAdaBoost ###################
    sample_size = [len(ailerons_tgt_df_X), len(ailerons_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    model_stradaboost_ailerons = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_ailerons.fit(ailerons_np_train_X, ailerons_np_train_y_list)
    y_pred_stradaboost_ailerons = model_stradaboost_ailerons.predict(ailerons_np_test_X)


    mse_stradaboost_ailerons = sqrt(mean_squared_error(ailerons_np_test_y, y_pred_stradaboost_ailerons))
    rmselist_stradaboost_ailerons.append(mse_stradaboost_ailerons)

    r2_score_stradaboost_ailerons = pearsonr(ailerons_np_test_y_list, y_pred_stradaboost_ailerons)
    r2_score_stradaboost_ailerons = (r2_score_stradaboost_ailerons[0])**2
    r2scorelist_stradaboost_ailerons.append(r2_score_stradaboost_ailerons)


with open('ailerons_rmse.txt', 'w') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in rmselist_AdaTL_ailerons)

with open('ailerons_r2.txt', 'w') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_ailerons)

######################################################################################

with open('ailerons_rmse.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in rmselist_Ada_ailerons)

with open('ailerons_r2.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in r2scorelist_Ada_ailerons)

######################################################################################

with open('ailerons_rmse.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in rmselist_KMM_ailerons)

with open('ailerons_r2.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in r2scorelist_KMM_ailerons)


######################################################################################

with open('ailerons_rmse.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in rmselist_GBRTL_ailerons)

with open('ailerons_r2.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_ailerons)


######################################################################################

with open('ailerons_rmse.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in rmselist_GBR_ailerons)

with open('ailerons_r2.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in r2scorelist_GBR_ailerons)


######################################################################################

with open('ailerons_rmse.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_ailerons)

with open('ailerons_r2.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_ailerons)

######################################################################################

with open('ailerons_rmse.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in rmselist_stradaboost_ailerons)

with open('ailerons_r2.txt', 'a') as ailerons_handle:
    ailerons_handle.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_ailerons)


######################################################################################

print("-------------------------------------------")
