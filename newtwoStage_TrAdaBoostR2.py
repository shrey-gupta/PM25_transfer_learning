import os
from functools import reduce
from math import ceil

import numpy as np
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from scipy.optimize import minimize_scalar
from xgboost import DMatrix, train
from sklearn.utils.extmath import stable_cumsum
from sklearn.ensemble.weight_boosting import BaseDecisionTree, BaseForest, is_regressor, check_X_y, check_array, check_random_state #DTYPE,
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

try:
    from tradaboost.validation import cross_val_score, cross_val_predict
except:
    from validation import cross_val_score, cross_val_predict
# from sklearn.model_selection import cross_val_score

__all__ = ["TradaboostClassifier", "TradaboostRegressor"]


class TradaboostClassifier(object):
    def __init__(self, learner, epoches):
        self.learner = learner
        self.epoches = epoches
        self.models = None
        self.betas = None

    def train(self, same_X, diff_X, same_y, diff_y):

        assert same_X.size == same_y, "same dist data size mismatch"
        assert diff_X.size == diff_y, "diff dist data size mismatch"

        self.models = np.ndarray(self.epoches, dtype=object)
        l_all = len(same_y) + len(diff_y)
        w0 = np.ones(l_all)
        X = pd.concat(diff_X, same_X)
        y = np.array(diff_y + same_y)
        n = len(diff_y)
        m = len(same_y)

        wt = w0
        betas = np.zeros(self.epoches)
        for i in range(self.epoches):
            wt = wt / np.sum(wt)
            model = self.learner.fit(X, y)
            self.models.append(model)
            y_predict = model.predict(X)

            loss = np.abs(y_predict - y)
            eta = ((loss * wt[n+1:]) / wt[n+1:].sum()).sum()

            if eta > 0.5:
                eta = 0.5
            if eta == 0:
                self.epoches = i
                break  # 防止过拟合

            beta_t = eta/(1-eta)
            beta = 1/(1+np.sqrt(2*np.log(n)/self.epoches))

            beta_factor = np.power(beta, loss[:n])
            beta_t_factor = np.power(beta_t, loss[n+1:]*-1)
            wt[:n] = wt[:n] * beta_factor
            wt[n+1:] = wt[n+1:] * beta_t_factor

            betas[i] = beta_t
        self.betas = betas

    def predict(self, X):
        l = X.shape[0]
        predicts = np.zeros(l)
        res = np.array([model.predict(X) for model in self.models])
        for i in range(l):
            left = np.sum(np.log(self.betas[ceil(self.epoches/2)+1:]) * -1 * res[ceil(self.epoches/2)+1:])
            right = np.sum(np.log(self.betas[ceil(self.epoches/2)+1:]) * -0.5)
            predicts[i] = 1 if left >= right else 0

        return predicts


class AdaBoostRegressorDash(AdaBoostRegressor):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):

        super(AdaBoostRegressorDash, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state)

        self.n = None
        self.m_indices = None

    def fit(self, X, y, n=None, sample_weight=None, indices=None):
        self.n = n
        self.m_indices = np.where(indices >= n) if indices is not None else indices ## All the indices which are > n i.e. target data indices [n+1:m]

        ## Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
            isinstance(self.base_estimator, (BaseDecisionTree, BaseForest))):
            dtype = np.float64 #DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc'] ## Accepting sparse matrix in the two formats: 'csr' and 'csc'

        ## Validating the Input. X should be 2D and y should be 1D.
        ## 'is_regressor' checks if the given learner is a regressor and return T or F based on that
        X, y = check_X_y(X, y, accept_sparse = accept_sparse, dtype = dtype,
                         y_numeric = is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            ## First validate the input using check_array.
            ## 'ensure_2d' checks if value error is to be raised when the array is not 2D. False means to not raise an error.
            sample_weight = check_array(sample_weight, ensure_2d = False)
            # Normalize existing sample weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        ## ??????????? Check parameters. Why is this even here? This can be a problem
        self._validate_estimator()

        ## Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        ## Boosting for the number of estimators entered by the user. We return the best estimator.
        for iboost in range(self.n_estimators):
            ## Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            ## Early termination when sample_weight is None
            if sample_weight is None:
                break

            ## Store the estimator weights and errors for the boosting iteration no.
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            ## Stop the boosting loop if the error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)
            m_sample_weight_sum = np.sum(sample_weight[self.m_indices] if self.m_indices else sample_weight[self.n+1:]) ## Sum of sample weights from [n+1:m] indices.
            lack = 1 - sample_weight_sum ## Check how less is sum of sample weights from 1.
            lack_factor = m_sample_weight_sum / (lack + m_sample_weight_sum) if lack * m_sample_weight_sum != 0 else 1

            ## Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize the weights for indices [n+1:m].
                if self.m_indices is None:
                    sample_weight[self.n+1:] /= lack_factor
                else:
                    sample_weight[self.m_indices] /= lack_factor

        ## Assert checks if condition returns True. If it returns False, an assertion error is raised.
        if sample_weight is not None:
            assert np.abs(sample_weight.sum() - 1.0) < 0.01 , "sample weight sum is %s rather than 1" % sample_weight.sum()
        return self

    def _boost(self, iboost, X, y, sample_weight, random_state):
        ## When size of source dataset is not provided.
        assert self.n is not None, "The size of source is not given!"

        estimator = self._make_estimator(random_state = random_state)

        ## Weighted sampling of the training set with replacement
        ## For NumPy >= 1.7.0 use np.random.choice
        cdf = stable_cumsum(sample_weight)
        cdf /= cdf[-1]
        uniform_samples = random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        ## searchsorted returns a scalar

        bootstrap_idx = np.array(bootstrap_idx, copy = False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()
        # print("estimator_error %s" % estimator_error)

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1. / beta)

        if not iboost == self.n_estimators - 1:
            if self.m_indices is None:
                sample_weight[self.n+1:] *= np.power(beta,(1. - error_vect[self.n+1:]) * self.learning_rate)
            else:
                sample_weight[self.m_indices] *= np.power(beta,(1. - error_vect[self.m_indices]) * self.learning_rate)

        return sample_weight, estimator_weight, estimator_error


def transfer_stratified_split(tsX, tsy, ttX, tty, cv=3):
    ss = StratifiedKFold(cv)
    ts = StratifiedKFold(cv)
    n = len(tsy)
    for ssg, tsg, in zip(ss.split(tsX, tsy.as_matrix()[:, 1]), ts.split(ttX, tty.as_matrix()[:, 1])):
        yield (np.hstack((ssg[0], tsg[0]+n)), np.hstack((ssg[1], tsg[1]+n)))

# for i in transfer_stratified_split([[1],[2],[3],[4], [9]], [0,1,0,1,0], [[5], [6],[7],[8]], [1, 0,1,0], 2):
#     print(i)

class TradaboostRegressor(object):
    def __init__(self, N, S, F, learner, parallel=2, **learner_params):
        """
        :param N:  the maximum number of boosting iterations N
        :param S:  the number of steps S
        :param F:  the number of folds F for cross validation
        :param learner:  a base learning algorithm
        :return:
        """
        self.S = S
        self.N = N
        self.F = F
        self.learner = learner
        self.model_t = None
        self.parallel = parallel
        self.learner_params = learner_params

    def train(self, tsX, tsy, ttX, tty):
        """
        :param tsX: source data X
        :param tsy: source data y
        :param ttX: target data X
        :param tty: target data y
        :return:
        """
        n = len(tsy) ## Length of source dataset
        m = len(tty) ## Length of target dataset
        wt = np.ones(n+m)/(n+m) ## Initial weights created as array of length (n+m)

        try:
            X = pd.concat([tsX, ttX], ignore_index=True) ## Creating a dataset with both source and target
            y = pd.concat([tsy, tty], ignore_index=True).as_matrix()[:, 1] ## Creating one column matrix for the 'Y' attribute
        except TypeError:
            X = np.vstack([tsX, ttX]) ## Stacks array in sequence, vertically.
            y = np.hstack([tsy, tty]) ## Stacks array in sequence, horizontally.

        scores = np.full(self.S, -999, dtype=np.float64) ## scores is an array of size 'S' filled with -999 and type = float64

        wts = []
        larger_times = 0
        total_bad_times = 0
        best_score = -999

        ## This executes for S iterations.
        for i in range(self.S):
            ## Printing weights of source samples.
            print(wt[:n].sum())

            ## If all weights of source samples are 0.
            if not wt[:n].any():
                print("source sample weight are all zero, break")
                break

            ## Calling AdaBoost.r2' with base learner and it's chosen parameters and the no. of boosting iterations.
            model_t = AdaBoostRegressorDash(self.learner(**self.learner_params), self.N)
            # score = cross_val_score(model_t, X, y, fit_params={"sample_weight": wt, "n": n}, cv=self.F, n_jobs=self.parallel).sum()/self.F

            fit_params = {
                "sample_weight": wt*(n+m),
                "n": n
            }

            cv = transfer_stratified_split(tsX, tsy, ttX, tty, cv=self.F)
            cross_y_predict = cross_val_predict(model_t, X, y, fit_params=fit_params, cv=cv, n_jobs=self.parallel)
            score = mean_squared_error(y[n+1:], cross_y_predict[n+1:])*-1               #TODO: metrics from input arg
            print("epoch %s: score %s" % (i, score))

            if score > best_score:
                best_score = score
                total_bad_times = 0
            else:
                total_bad_times += 1

            if i > 0 and score <= scores[i-1]:
                larger_times += 1
                print("score is smaller than the last time %s:%s" % (score, scores[i-1]))
                if larger_times > 4 or total_bad_times > 10:
                    print("training is not getting any better, break")
                    break
            else:
                larger_times = 0

            ## Add score of the iteration to the scores list
            scores[i] = score
            ## Add weight of the iteration to the weights list
            wts.append(wt.copy())


            lnr = self.learner(**self.learner_params)
            lnr.fit(X, y, sample_weight = wt*(m+n))
            y_predict = lnr.predict(X)
            eta = np.abs(y - y_predict)

            print("rms: %s" % mean_squared_error(y, y_predict))
            print("eta sum: %s" % eta[n+1:].sum())    # why eta is the same in every epoch? wt?
            print("eta max: %s" % eta[n+1:].max())

            if eta.max() > 0:
                eta /= eta.max() ## ?????? This is adjusted error. But we have to look at it's formula more carefully.
            else:
                print("no loss, break")
                break

            beta = self.get_beta(eta, wt, i, n, m)

            if not beta:
                print("can't find beta, break")
                break

            print("get beta: %s" % beta)

            wt[:n] = wt[:n]*np.power(beta, eta[:n]) ## weight of [1:n] instances updates
            wt = wt/wt.sum() ## All instances divided by normalizing constant wt.sum()

        t = scores.argmax()
        print("%sth is the best model, score:%s" % (t, scores[t]))

        self.model_t = AdaBoostRegressorDash(self.learner(**self.learner_params), self.N)
        self.model_t.fit(X, y, sample_weight=wts[t]*(n+m), n=n)
        self.model_t.best_score = scores[t]

    def predict(self, X):
        assert self.model_t is not None, "You need to train your model first!"
        return self.model_t.predict(X)

    def get_beta(self, eta, weight, t, n, m):
        target = (m/(n+m) + t*(1-m/(n+m))/(self.S-1))

        def funct(beta):
            wt = weight.copy()
            wt[:n] *= np.power(beta, eta[:n])
            loss = wt[n+1:].sum() - target*wt.sum()
            # print("%s %s" % (beta, loss))
            return np.abs(loss)

        # res = minimize_scalar(funct, method="bounded", bounds=(-10*7, 10**7), options={'xatol': 1e-02})
        res = minimize_scalar(funct, method="golden", options={'xtol': 1.4901161193847656e-08,'maxiter': 7000})
        print("target weight sum: %s, loss: %s" % (target, res["fun"]))

        return res.x if res["success"] else None
