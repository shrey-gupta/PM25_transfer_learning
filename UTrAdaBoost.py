"""
STrAdaBoost.R2 algorithm

based on algorithm 3 in paper "Boosting for Regression Transfer".

"""

import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb


################################################################################
## the second stage
################################################################################
class Stage2_TrAdaBoostR2:
    def __init__(self,
                 base_estimator = DecisionTreeRegressor(max_depth = 6),
                 sample_size = None,
                 n_estimators = 100,
                 learning_rate = 0.1,
                 loss = 'square', #'linear'
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state


    def fit(self, X, y, sample_weight=None):
        ## Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            ## Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64) ##???
            sample_weight[:] = 1. / X.shape[0] ## Note: 1. means 1.0 .... Hence the expression is 1.0/ X.shape[0]
        else:
            ## Normalize existing weights. Don't need to normalize before.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            ## Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

        ## Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        ## this for loop is sequential and does not support parallel(revison is needed for making parallel)
        for iboost in range(self.n_estimators):
            ## AdaBoostR2' step
            sample_weight, estimator_weight, estimator_error = self._stage2_adaboostR2(iboost, X, y, sample_weight)

            ## Early termination. Hence, if it is returned None by the previous step, we would break out of this.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            ## Stop if error is zero. You got the best estimaator.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            ## Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
            ## Normalize before moving towards the next step.
                sample_weight = sample_weight /sample_weight_sum
        return self


    def _stage2_adaboostR2(self, iboost, X, y, sample_weight):

        estimator = copy.deepcopy(self.base_estimator) ## some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)
        ## deepcopy() allows to make a copy and execute changes into the copy without altering the original.

        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        ## Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight)
        cdf /= cdf[-1]
        # print("The cdf is: ", cdf)
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)
        # print("the bootstrap_idx is: ", bootstrap_idx)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        ## add the fitted estimator to the list of estimators.
        self.estimators_.append(estimator)

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect = error_vect/error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        ## Calculate the average loss or the adjusted error is calculated.
        estimator_error = (sample_weight * error_vect).sum()

        if estimator_error <= 0:
            ## Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            ## Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        ## avoid overflow of np.log(1. / beta). This step is basically for overflow.
        if beta < 1e-308:
            beta = 1e-308
        estimator_weight = self.learning_rate * np.log(1. / beta)

        ## Boost weight using AdaBoost.R2 alg except the weight of the source data remains the same
        ## the weight of the source data are remained
        # source_weight_sum= np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
        # target_weight_sum = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)

        if not iboost == self.n_estimators - 1:
            sample_weight[-self.sample_size[-1]:] *= np.power(beta,(1. - error_vect[-self.sample_size[-1]:]) * self.learning_rate)

            ## make the sum weight of the source data not changing
            # source_weight_sum_new = np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
            # target_weight_sum_new = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
            #
            # if source_weight_sum_new != 0. and target_weight_sum_new != 0.:
            #     sample_weight[:-self.sample_size[-1]] = sample_weight[:-self.sample_size[-1]]*source_weight_sum/source_weight_sum_new
            #     sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[-1]:]*target_weight_sum/target_weight_sum_new

        return sample_weight, estimator_weight, estimator_error


    def predict(self, X):
        # Evaluate predictions of all estimators
        predictions = np.array([
                est.predict(X) for est in self.estimators_[:len(self.estimators_)]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]


################################################################################
## the whole two stages
################################################################################
class TwoStageTrAdaBoostR2:
    def __init__(self,
                 base_estimator = DecisionTreeRegressor(max_depth = 6),
                 sample_size = None,
                 n_estimators = 100,
                 steps = 30,
                 fold = 10,
                 learning_rate = 0.1,
                 loss = 'square', #'linear',
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.steps = steps
        self.fold = fold
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state


    def fit(self, X, y, sample_weight = None):
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")

        ## Checking if correct sample size has been input.
        ## If 'no' sample size is provided, raise this error.
        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        ## If sample size provided does not equal to the size of the input then raise this error.
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")


        ## Dissociates the source and target dataset from X and y provided using sample_size.
        X_source = X[:-self.sample_size[-1]]
        y_source = y[:-self.sample_size[-1]]
        X_target = X[-self.sample_size[-1]:]
        y_target = y[-self.sample_size[-1]:]


        self.models_ = []
        self.errors_ = []

        ## This loop accounts for the number of steps 'S' in TrAdaBoost algorithm.
        for istep in range(self.steps):
            ## The AdaBoostR2' is being called twice. Once for just fitting the training data and the other time for testing using cross validation.
            ## How can we make it into 1 model being called.
            ## Calling AdaBoostR2'
#             model = Stage2_TrAdaBoostR2(self.base_estimator,
#                                         sample_size = self.sample_size,
#                                         n_estimators = self.n_estimators,
#                                         learning_rate = self.learning_rate,
#                                         loss = self.loss,
#                                         random_state = self.random_state)

            model = AdaBoostRegressor(self.base_estimator, #DecisionTreeRegressor(max_depth=6),
                                      n_estimators = self.n_estimators)
                                      # learning_rate = self.learning_rate,
                                      # loss = self.loss,
                                      # random_state = self.random_state)

            # model = xgb.XGBRegressor(objective ='reg:squarederror',
            #                         learning_rate = self.learning_rate,
            #                         max_depth = 5,
            #                         n_estimators = self.n_estimators)


            ## The sample weights were calculate above. They are 1/n+m. Where n: no. of source instances, m: no. of target instances.
            ## Fitting the model means making it learn. This basically means for 'Boosting', to provide weights to the instances.
            model.fit(X, y, sample_weight = sample_weight)
            ## Add this models to the list of models
            self.models_.append(model)

            ## Cross Validation training
            kf = KFold(n_splits = self.fold) ## Create no. of CV Folds
            error = []
            ## Seperate the source and the target weights from the sample_weight
            target_weight = sample_weight[-self.sample_size[-1]:]
            source_weight = sample_weight[:-self.sample_size[-1]]

            ## Find the mean error on the model using CV
            for train, test in kf.split(X_target):
                sample_size = [self.sample_size[0], len(train)]
                ## This initialization always remains constant; so it basically has no effect. We do this for the change in sample_size.
#                 model = Stage2_TrAdaBoostR2(self.base_estimator,
#                                         sample_size = sample_size,
#                                         n_estimators = self.n_estimators,
#                                         learning_rate = self.learning_rate,
#                                         loss = self.loss,
#                                         random_state = self.random_state)

                model = AdaBoostRegressor(self.base_estimator, #DecisionTreeRegressor(max_depth=6),
                                          n_estimators = self.n_estimators)
                                          # learning_rate = self.learning_rate,
                                          # loss = self.loss,
                                          # random_state = self.random_state)

                # model = xgb.XGBRegressor(objective ='reg:squarederror',
                #                         learning_rate = self.learning_rate,
                #                         max_depth = 5,
                #                         n_estimators = self.n_estimators)


                ## Divide the dataset into source and target data.
                X_train = np.concatenate((X_source, X_target[train]))
                y_train = np.concatenate((y_source, y_target[train]))
                X_test = X_target[test]
                y_test = y_target[test]

                ## make sure the sum weight of the target data do not change with CV's split sampling i.e. Normalizing them by multiplying them with a factor.
                ## target_weight_train remians the same over all the CV iterations.
                target_weight_train = target_weight[train]*np.sum(target_weight)/np.sum(target_weight[train])
                ## Make this model learn.
                ## source_weight remains the same over all the CV iterations.
                model.fit(X_train, y_train, sample_weight = np.concatenate((source_weight, target_weight_train)))
                ## Get the predictions for the model fitted on the test set.
                y_predict = model.predict(X_test)
                ## Append the error into a list and then we would take mean of this list.
                error.append(mean_squared_error(y_predict, y_test))

            ## Add the mean of all the errors obtained using KFold CV in the errors list.
            ## Each value in 'errors' list corresponds to the model in the 'models' list.
            self.errors_.append(np.array(error).mean())
            ## Find the updated sample_weights
            # sample_weight = self._twostage_adaboostR2(istep, X, y, sample_weight)

            ## Updating the sample weights by finding adjusted error first.
            y_predict = model.predict(X)
            error_vect = np.abs(y_predict - y)
            error_max = error_vect.max()

            if error_max != 0.:
                error_vect = error_vect/error_max ## error_vect now has the adjusted error.

            if self.loss == 'square':
                error_vect **= 2
            elif self.loss == 'exponential':
                error_vect = 1. - np.exp(- error_vect)

            estimator_error = (sample_weight * error_vect).sum()

            # beta = self._beta_binary_search(istep, sample_weight, error_vect, stp = 1e-30)
            # beta = self.get_beta(eta, wt, i, n, m)
            n = len(y_source) ## Length of source dataset
            m = len(y_target) ## length of target dataset
            beta = (m/(n+m) + istep*(1-m/(n+m))/(self.steps-1))
            beta2 = estimator_error / (1. - estimator_error)


            if not beta:
                print("can't find beta, break")
                break

            # if not istep == self.steps - 1:
            ## Updating source instances weight using the equation.
            ## Original Equation
            sample_weight[:-self.sample_size[-1]] = (sample_weight[:-self.sample_size[-1]]* np.power(beta,error_vect[:-self.sample_size[-1]]))* (self.learning_rate/sample_weight.sum())
#             sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[-1]:]*(self.learning_rate/sample_weight.sum())
            sample_weight[-self.sample_size[-1]:] = (sample_weight[-self.sample_size[-1]:]* np.power(beta2,1 - error_vect[-self.sample_size[-1]:]))* (self.learning_rate/sample_weight.sum())

            ## First equation that Jianzhao suggested
            # sample_weight[:-self.sample_size[-1]] = ((0.25*sample_weight[-self.sample_size[-1]:].mean())* self.learning_rate)/ sample_weight.sum()
            # sample_weight[-self.sample_size[-1]:] = (sample_weight[-self.sample_size[-1]:]* self.learning_rate)/sample_weight.sum()

            ## Second equation that Jianzhao suggested
            # sample_weight[:-self.sample_size[-1]] = 0.2*sample_weight[:-self.sample_size[-1]]* self.learning_rate/ sample_weight.sum()
            # sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[-1]:]* self.learning_rate/sample_weight.sum()

            ## If the sample_weights do not exist then break out.
            if sample_weight is None:
                break
            ## If the mean error comes out to be 0 then we have found the perfect model.
            if np.array(error).mean() == 0:
                break

            ## Stop if the sum of sample weights has become non-positive
            sample_weight_sum = np.sum(sample_weight)
            if sample_weight_sum <= 0:
                break

            # if istep < self.steps - 1:
            #     ## When we reach the end of the no. of steps then normalize the sample_weights
            #     sample_weight = sample_weight/sample_weight_sum

        return self


    def _twostage_adaboostR2(self, istep, X, y, sample_weight):

        ## some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)
        ## Creating a deepcopy of the estimator, so that changes to it does not change the original.
        estimator = copy.deepcopy(self.base_estimator)

        ############################## Do we need this step?? ############################################################
        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        ## Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        ## searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)
        ## Fit on the bootstrapped sample and obtain a prediction for all samples in the training set.
        ## Fit on the same model used for deepcopy and using the same sample weights and then make predictions on them.
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        ##################################################################################################################

        ## Updating the sample weights.
        y_predict = estimator.predict(X)
        ## Calculate error.
        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        beta = self._beta_binary_search(istep, sample_weight, error_vect, stp = 1e-30)

        if not istep == self.steps - 1:
            ## Updating source instances weight using the equation.
            sample_weight[:-self.sample_size[-1]] *= np.power(beta,(error_vect[:-self.sample_size[-1]]) * self.learning_rate)
        return sample_weight


    def _beta_binary_search(self, istep, sample_weight, error_vect, stp):
        # calculate the specified sum of weight for the target data
        n_target = self.sample_size[-1]
        n_source = np.array(self.sample_size).sum() - n_target
        theoretical_sum = n_target/(n_source+n_target) + istep/(self.steps-1)*(1-n_target/(n_source+n_target))
        # for the last iteration step, beta is 0.
        if istep == self.steps - 1:
            beta = 0.
            return beta
        # binary search for beta
        L = 0.
        R = 1.
        beta = (L+R)/2
        sample_weight_ = copy.deepcopy(sample_weight)
        sample_weight_[:-n_target] *= np.power(
                    beta,
                    (error_vect[:-n_target]) * self.learning_rate)
        sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
        updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)

        while np.abs(updated_weight_sum - theoretical_sum) > 0.01:
            if updated_weight_sum < theoretical_sum:
                R = beta - stp
                if R > L:
                    beta = (L+R)/2
                    sample_weight_ = copy.deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                                beta,
                                (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    print("At step:", istep+1)
                    print("Binary search's goal not meeted! Value is set to be the available best!")
                    print("Try reducing the search interval. Current stp interval:", stp)
                    break

            elif updated_weight_sum > theoretical_sum:
                L = beta + stp
                if L < R:
                    beta = (L+R)/2
                    sample_weight_ = copy.deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                                beta,
                                (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    print("At step:", istep+1)
                    print("Binary search's goal not meeted! Value is set to be the available best!")
                    print("Try reducing the search interval. Current stp interval:", stp)
                    break
        return beta


    def predict(self, X):
        # select the model with the least CV error
        fmodel = self.models_[np.array(self.errors_).argmin()]
        predictions = fmodel.predict(X)
        return predictions
