"""
Transfer Adaboost
"""

import inspect
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from adapt.utils import check_arrays, check_one_array, check_estimator

EPS = np.finfo(float).eps

def _get_median_predict(X, predictions, weights):
    sorted_idx = np.argsort(predictions, axis=-1)
    # Find index of median prediction for each sample
    weight_cdf = np.cumsum(weights[sorted_idx], axis=-1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[..., -1][..., np.newaxis]
    median_idx = median_or_above.argmax(axis=-1)
    new_predictions = None
    for i in range(median_idx.shape[1]):
        median_estimators = sorted_idx[np.arange(len(X)), i, median_idx[:, i]]
        if new_predictions is None:
            new_predictions = predictions[np.arange(len(X)), i, median_estimators].reshape(-1,1)
        else:
            new_predictions = np.concatenate((
                new_predictions,
                predictions[np.arange(len(X)), i, median_estimators].reshape(-1,1)
            ), axis=1)
    return new_predictions


def _binary_search(func, verbose=1):
    left=0
    right=1
    tol=1.e-3
    best=1
    best_score=1
    for i in range(1000):
        if np.abs(func(left)) < tol:
            best = left
            break
        elif np.abs(func(right)) < tol:
            best = right
            break
        else:
            midle = (left + right) / 2
            if func(midle) < best_score:
                best = midle
                best_score = func(midle)
            if func(midle) * func(left) <= 0:
                right = midle
            else:
                left = midle
    if i >= 999 and verbose:
        print("Binary search has not converged."
              " Set value to the current best.")
    return best


class TrAdaBoost:
    """
    Transfer AdaBoost for Classification

    TrAdaBoost algorithm is a **supervised** instances-based domain
    adaptation method suited for **classification** tasks.

    The method is based on a "**reverse boosting**" principle where the
    weights of source instances poorly predicted decrease at each
    boosting iteration whereas the ones of target instances increase.

    The algorithm performs the following steps:

    - **1.** Normalize weights: :math:`\\sum w_S + \\sum w_T = 1`.
    - **2.** Fit an estimator :math:`f` on source and target labeled data
      :math:`(X_S, y_S), (X_T, y_T)` with the respective importances
      weights: :math:`w_S, w_T`.
    - **3.** Compute error vectors of training instances:

      - :math:`\\epsilon_S = L_{01}(f(X_S), y_S)`.
      - :math:`\\epsilon_T = L_{01}(f(X_T), y_T)`.

    - **4.** Compute total weighted error of target instances:
      :math:`E_T = \\frac{1}{n_T} w_T^T \\epsilon_T`.
    - **5.** Update source and target weights:

       - :math:`w_S = w_S \\beta^{\\epsilon_S}`.
       - :math:`w_T = w_T \\beta_T^{-\\epsilon_T}`.

      Where:

      - :math:`\\beta = 1 \\setminus (1 + \\sqrt{2 \\text{ln} n_S \\setminus N})`.
      - :math:`\\beta_T = E_T \\setminus (1 - E_T)`.

    - **6.** Return to step **1** and loop until the number :math:`N`
      of boosting iteration is reached.

    The prediction are then given by the vote of the :math:`N \\setminus 2`
    last computed estimators weighted by their respective parameter
    :math:`\\beta_T`.

    Parameters
    ----------
    estimator : sklearn estimator or tensorflow Model (default=None)
        Base estimator used to learn the task.
        If estimator is ``None``, a ``LogisticRegression``
        instance is used as base estimator.

    n_estimators : int (default=10)
        Number of boosting iteration.

    verbose : int (default=1)
        Verbosity level.

    random_state : int (default=None)
        Seed of random generator.
    Attributes
    ----------
    estimators_ : list of object
        List of fitted estimators

    estimator_errors_ : 1D array of float
        Array of weighted estimator errors computed on
        labeled target data.

    estimator_weights_ : 1D array of float
        Array of estimator importance weights.

    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.

    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from adapt.instance_based import TrAdaBoost
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> np.random.seed(0)
    >>> Xs = np.random.random((100, 2))
    >>> Xt = np.random.random((100, 2))
    >>> ys = (Xs[:, [0]] < 0.5).astype(int)
    >>> yt = (Xt[:, [1]] < 0.5).astype(int)
    >>> dtc = DecisionTreeClassifier(max_depth=5)
    >>> dtc.fit(np.concatenate((Xs, Xt[:10])),
    ...         np.concatenate((ys, yt[:10])))
    >>> dtc.score(Xt, yt)
    0.55
    >>> tr = TrAdaBoost(DecisionTreeClassifier(max_depth=5),
    ...                 n_estimators=20, random_state=1)
    >>> tr.fit(Xs, ys, Xt[:10], yt[:10])
    Iteration 0 - Error: 0.1000
    ...
    Iteration 19 - Error: 0.0000
    >>> (tr.predict(Xt) == yt.ravel()).mean()
    0.59

    See also
    --------
    TrAdaBoostR2, TwoStageTrAdaBoostR2
    References
    ----------
    .. [1] `[1] <http://www.cs.ust.hk/~qyang/Docs/2007/tradaboost.pdf>`_ Dai W., \
Yang Q., Xue G., and Yu Y. "Boosting for transfer learning". In ICML, 2007.
    """

    def __init__(self, estimator=None, n_estimators=10,
                 verbose=1, random_state=None):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.task_ = "class"
        if isinstance(self, TrAdaBoostR2):
            self.task_ = "reg"

        self.base_estimator_ = check_estimator(estimator, copy=True,
                                               force_copy=True,
                                               task=self.task_)
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.random_state = random_state


    def fit(self, Xs, ys, Xt, yt,
            sample_weight_src=None,
            sample_weight_tgt=None,
            **fit_params):
        """
        Fit TrAdaBoost

        Parameters
        ----------
        Xs : numpy array
            Source input data.
        ys : numpy array
            Source output data.
        Xt : numpy array
            Target input data.

        yt : numpy array
            Target output data.

        sample_weight_src : numpy array, (default=None)
            Initial sample weight of source data

        sample_weight_tgt : numpy array, (default=None)
            Initial sample weight of target data
        fit_params : key, value arguments
            Arguments given to the fit method of the
            estimator.
        Returns
        -------
        self : returns an instance of self
        """
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        Xs, ys, Xt, yt = check_arrays(Xs, ys, Xt, yt)

        n_s = len(Xs)
        n_t = len(Xt)

        if sample_weight_src is None:
            sample_weight_src = np.ones(n_s) / (n_s + n_t)
        if sample_weight_tgt is None:
            sample_weight_tgt = np.ones(n_t) / (n_s + n_t)

        sum_weights = (sample_weight_src.sum() +
                       sample_weight_tgt.sum())
        sample_weight_src = sample_weight_src / sum_weights
        sample_weight_tgt = sample_weight_tgt / sum_weights

        self.sample_weights_src_ = []
        self.sample_weights_tgt_ = []
        self.estimators_ = []
        self.estimator_errors_ = []

        for iboost in range(self.n_estimators):
            self.sample_weights_src_.append(
                np.copy(sample_weight_src))
            self.sample_weights_tgt_.append(
                np.copy(sample_weight_tgt))

            sample_weight_src, sample_weight_tgt = self._boost(
                iboost, Xs, ys, Xt, yt,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )

            if self.verbose >= 1:
                print("Iteration %i - Error: %.4f"%
                      (iboost, self.estimator_errors_[-1]))

            if sample_weight_src is None:
                break

            sum_weights = (sample_weight_src.sum() +
                           sample_weight_tgt.sum())
            sample_weight_src = sample_weight_src / sum_weights
            sample_weight_tgt = sample_weight_tgt / sum_weights

        self.estimator_errors_ = np.array(self.estimator_errors_)
        self.estimator_weights_ = np.array([
            -np.log(err / (1-err) + EPS) + 2*EPS
            for err in self.estimator_errors_])
        return self


    def _boost(self, iboost, Xs, ys, Xt, yt,
               sample_weight_src, sample_weight_tgt,
               **fit_params):

        X = np.concatenate((Xs, Xt))
        y = np.concatenate((ys, yt))
        sample_weight = np.concatenate((sample_weight_src,
                                        sample_weight_tgt))

        estimator = check_estimator(self.base_estimator_,
                                    copy=True, force_copy=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if "sample_weight" in inspect.signature(estimator.fit).parameters:
                estimator.fit(X, y,
                              sample_weight=sample_weight,
                              **fit_params)
            else:
                bootstrap_index = np.random.choice(
                len(X), size=len(X), replace=True, p=sample_weight)
                estimator.fit(X[bootstrap_index], y[bootstrap_index],
                              **fit_params)

        ys_pred = estimator.predict(Xs)
        yt_pred = estimator.predict(Xt)

        if ys_pred.ndim == 1:
            ys_pred = ys_pred.reshape(-1, 1)
            yt_pred = yt_pred.reshape(-1, 1)

        if self.task_ == "reg":
            error_vect_src = np.abs(ys_pred - ys).mean(tuple(range(1, ys.ndim)))
            error_vect_tgt = np.abs(yt_pred - yt).mean(tuple(range(1, yt.ndim)))
            error_vect = np.concatenate((error_vect_src, error_vect_tgt))

            error_max = error_vect.max() + EPS
            if error_max != 0:
                error_vect /= error_max
                error_vect_src /= error_max
                error_vect_tgt /= error_max
        else:
            if isinstance(estimator, BaseEstimator):
                error_vect_src = (ys_pred != ys).astype(float).ravel()
                error_vect_tgt = (yt_pred != yt).astype(float).ravel()
                error_vect = np.concatenate((error_vect_src, error_vect_tgt))
            else:
                if ys.shape[1] == 1:
                    error_vect_src = (np.abs(ys_pred - ys) > 0.5).astype(float).ravel()
                    error_vect_tgt = (np.abs(yt_pred - yt) > 0.5).astype(float).ravel()
                else:
                    error_vect_src = (ys_pred.argmax(1) != ys.argmax(1)).astype(float).ravel()
                    error_vect_tgt = (yt_pred.argmax(1) != yt.argmax(1)).astype(float).ravel()

        error_vect = np.concatenate((error_vect_src, error_vect_tgt))

        if isinstance(self, _AdaBoostR2):
            estimator_error = (sample_weight * error_vect).sum()
        else:
            estimator_error = ((sample_weight_tgt * error_vect_tgt).sum() /
                               sample_weight_tgt.sum())

        if estimator_error > 0.5:
            estimator_error = 0.5

        beta_t = estimator_error / (1. - estimator_error)

        beta_s = 1. / (1. + np.sqrt(
            2. * np.log(len(Xs)) / self.n_estimators
        ))

        if not iboost == self.n_estimators - 1:
            if isinstance(self, _AdaBoostR2):
                sample_weight_tgt = (sample_weight_tgt *
                np.power(beta_t, (1 - error_vect_tgt)))

                sample_weight_tgt *= ((1. - sample_weight_src.sum()) /
                                      sample_weight_tgt.sum())
            else:
                # Source updating weights
                sample_weight_src *= np.power(
                    beta_s, error_vect_src)

                # Target updating weights
                sample_weight_tgt *= np.power(
                    beta_t, - error_vect_tgt)

        self.estimators_.append(estimator)
        self.estimator_errors_.append(estimator_error)

        return sample_weight_src, sample_weight_tgt


    def predict(self, X):
        """
        Return weighted vote of estimators.

        Parameters
        ----------
        X : array
            Input data.
        Returns
        -------
        y_pred : array
            Vote results.
        """
        X = check_one_array(X)
        N = len(self.estimators_)
        weights = self.estimator_weights_[int(N/2):]
        predictions = []
        for est in self.estimators_[int(N/2):]:
            if isinstance(est, BaseEstimator):
                y_pred = est.predict_proba(X)
            else:
                y_pred = est.predict(X)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
                if y_pred.shape[1] == 1:
                    y_pred = np.concatenate((1-y_pred, y_pred),
                                            axis=1)
            predictions.append(y_pred)
        predictions = np.stack(predictions, -1)
        weighted_vote = predictions.dot(weights).argmax(1)
        return weighted_vote


    def predict_weights(self, domain="src"):
        """
        Return sample weights.

        Return the final source importance weighting.

        Parameters
        ----------
        domain : str (default="tgt")
            Choose between ``"source", "src"`` and
            ``"target", "tgt"``.
        Returns
        -------
        weights : source sample weights
        """
        if hasattr(self, "sample_weights_src_"):
            if domain in ["src", "source"]:
                return self.sample_weights_src_[-1]
            elif domain in ["tgt", "target"]:
                return self.sample_weights_tgt_[-1]
            else:
                raise ValueError("`domain `argument "
                                 "should be `tgt` or `src`, "
                                 "got, %s"%domain)
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit' first.")



class TrAdaBoostR2(TrAdaBoost):
    """
    Transfer AdaBoost for Regression

    TrAdaBoostR2 algorithm is a **supervised** instances-based domain
    adaptation method suited for **regression** tasks.

    The method is based on a "**reverse boosting**" principle where the
    weights of source instances poorly predicted decrease at each
    boosting iteration whereas the ones of target instances increase.

    The algorithm performs the following steps:

    - **1.** Normalize weights: :math:`\\sum w_S + \\sum w_T = 1`.
    - **2.** Fit an estimator :math:`f` on source and target labeled data
      :math:`(X_S, y_S), (X_T, y_T)` with the respective importances
      weights: :math:`w_S, w_T`.
    - **3.** Compute error vectors of training instances:

      - :math:`\\epsilon_S = L(f(X_S), y_S)`.
      - :math:`\\epsilon_T = L(f(X_T), y_T)`.

    - **4** Normalize error vectors:

      - :math:`\\epsilon_S = \\epsilon_S \\setminus
        max_{\\epsilon \\in \\epsilon_S \cup \\epsilon_T} \\epsilon`.
      - :math:`\\epsilon_T = \\epsilon_T \\setminus
        max_{\\epsilon \\in \\epsilon_S \cup \\epsilon_T} \\epsilon`.

    - **5.** Compute total weighted error of target instances:
      :math:`E_T = \\frac{1}{n_T} w_T^T \\epsilon_T`.


    - **6.** Update source and target weights:

       - :math:`w_S = w_S \\beta^{\\epsilon_S}`.
       - :math:`w_T = w_T \\beta_T^{-\\epsilon_T}`.

      Where:

      - :math:`\\beta = 1 \\setminus (1 + \\sqrt{2 \\text{ln} n_S \\setminus N})`.
      - :math:`\\beta_T = E_T \\setminus (1 - E_T)`.

    - **7.** Return to step **1** and loop until the number :math:`N`
      of boosting iteration is reached.

    The prediction are then given by the weighted median of the
    :math:`N \\setminus 2` last estimators.

    Parameters
    ----------
    estimator : sklearn estimator or tensorflow Model (default=None)
        Base estimator used to learn the task.
        If estimator is ``None``, a ``LinearRegression``
        instance is used as base estimator.

    n_estimators : int (default=10)
        Number of boosting iteration.

    verbose : int (default=1)
        Verbosity level.

    random_state : int (default=None)
        Seed of random generator.
    Attributes
    ----------
    estimators_ : list of object
        List of fitted estimators

    estimator_errors_ : 1D array of float
        Array of weighted estimator errors computed on
        labeled target data.

    estimator_weights_ : 1D array of float
        Array of estimator importance weights.
    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.

    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from adapt.instance_based import TrAdaBoostR2
    >>> np.random.seed(0)
    >>> Xs = np.random.random((100, 2))
    >>> Xt = np.random.random((100, 2))
    >>> ys = Xs[:, [0]]
    >>> yt = Xt[:, [1]]
    >>> lr = LinearRegression()
    >>> lr.fit(np.concatenate((Xs, Xt[:10])),
    ...        np.concatenate((ys, yt[:10])))
    >>> np.abs(lr.predict(Xt) - yt).mean()
    0.30631...
    >>> tr = TrAdaBoostR2(n_estimators=20)
    >>> tr.fit(Xs, ys, Xt[:10], yt[:10])
    Iteration 0 - Error: 0.4396
    ...
    Iteration 19 - Error: 0.0675
    >>> np.abs(tr.predict(Xt) - yt).mean()
    0.05801...
    See also
    --------
    TrAdaBoost, TwoStageTrAdaBoostR2
    References
    ----------
    .. [1] `[1] <https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf>`_ \
D. Pardoe and P. Stone. "Boosting for regression transfer". In ICML, 2010.
    """

    def predict(self, X):
        """
        Return weighted median of estimators.

        Parameters
        ----------
        X : array
            Input data.
        Returns
        -------
        y_pred : array
            Median results.
        """
        X = check_one_array(X)
        N = len(self.estimators_)
        weights = self.estimator_weights_
        weights = weights[int(N/2):]
        predictions = []
        for est in self.estimators_[int(N/2):]:
            y_pred = est.predict(X)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            predictions.append(y_pred)
        predictions = np.stack(predictions, -1)
        return _get_median_predict(X, predictions, weights)


class _AdaBoostR2(TrAdaBoostR2):
    """
    AdaBoostR2 object with fixed sample weights.
    """
    pass


class TwoStageTrAdaBoostR2(TrAdaBoostR2):
    """
    Two Stage Transfer AdaBoost for Regression

    TwoStageTrAdaBoostR2 algorithm is a **supervised** instances-based
    domain adaptation method suited for **regression** tasks.

    The method is based on a "**reverse boosting**" principle where the
    weights of source instances poorly predicted decrease at each
    boosting iteration whereas the ones of target instances increase.

    This "two stages" version of TrAdaBoostR2 algorithm update separately
    the weights of source and target instances.

    In a first stage, the weights of source instances are
    frozen whereas the ones of target instances are updated according to
    the classical AdaBoostR2 algorithm. In a second stage, the weights of
    target instances are now frozen whereas the ones of source instances
    are updated according to the TrAdaBoost algorithm.

    At each first stage, a cross-validation score is computed with the
    labeled target data available. The CV scores obtained are used at
    the end to select the best estimator whithin all boosting iterations.

    The algorithm performs the following steps:

    - **1.** Normalize weights: :math:`\\sum w_S + \\sum w_T = 1`.
    - **2.** Fit an AdaBoostR2 estimator :math:`f` on source and target
      labeled data :math:`(X_S, y_S), (X_T, y_T)` with the respective
      importances initial weights: :math:`w_S, w_T`. During training
      of the AdaBoost estimator, the source weights :math:`w_S` are
      frozen.
    - **3.** Compute a cross-validation score on :math:`(X_T, y_T)`
    - **4.** Compute error vectors of training instances:

      - :math:`\\epsilon_S = L(f(X_S), y_S)`.
      - :math:`\\epsilon_T = L(f(X_T), y_T)`.

    - **5** Normalize error vectors:

      - :math:`\\epsilon_S = \\epsilon_S \\setminus
        max_{\\epsilon \\in \\epsilon_S \cup \\epsilon_T} \\epsilon`.
      - :math:`\\epsilon_T = \\epsilon_T \\setminus
        max_{\\epsilon \\in \\epsilon_S \cup \\epsilon_T} \\epsilon`.

    - **6.** Update source and target weights:

       - :math:`w_S = w_S \\beta_S^{\\epsilon_S} \\setminus Z`.
       - :math:`w_T = w_T \\setminus Z`.

      Where:

      - :math:`Z` is a normalizing constant.
      - :math:`\\beta_S` is chosen such that the sum of target weights
        :math:`w_T` is equal to :math:`\\frac{n_T}{n_T + n_S}
        + \\frac{t}{N - 1}(1 - \\frac{n_T}{n_T + n_S})` with :math:`t`
        the current boosting iteration number. :math:`\\beta_S` is found
        using binary search.

    - **7.** Return to step **1** and loop until the number :math:`N`
      of boosting iteration is reached.

    The prediction are then given by the best estimator according
    to cross-validation scores.

    Parameters
    ----------
    estimator : sklearn estimator or tensorflow Model (default=None)
        Base estimator used to learn the task.
        If estimator is ``None``, a ``LinearRegression``
        instance is used as base estimator.

    n_estimators : int (default=10)
        Number of boosting iteration.

    n_estimators_fs : int (default=10)
        Number of boosting iteration in first stage
        (given to AdaboostR2 estimators)

    cv: int, optional (default=5)
        Split cross-validation parameter.

    verbose : int (default=1)
        Verbosity level.

    random_state : int (default=None)
        Seed of random generator.
    Attributes
    ----------
    estimators_ : list of object
        List of fitted AdaboostR2 estimators for each
        first stage.

    estimator_errors_ : 1D array of float
        Array of cross-validation MAE computed on
        labeled target data.
    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.
    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from adapt.instance_based import TwoStageTrAdaBoostR2
    >>> np.random.seed(0)
    >>> Xs = np.random.random((100, 2))
    >>> Xt = np.random.random((100, 2))
    >>> ys = Xs[:, [0]]
    >>> yt = Xt[:, [1]]
    >>> lr = LinearRegression()
    >>> lr.fit(np.concatenate((Xs, Xt[:10])),
    ...        np.concatenate((ys, yt[:10])))
    >>> np.abs(lr.predict(Xt) - yt).mean()
    0.30631...
    >>> tr = TwoStageTrAdaBoostR2()
    >>> tr.fit(Xs, ys, Xt[:10], yt[:10])
    Iteration 0 - Cross-validation score: 0.3154 (0.1813)
    ...
    Iteration 9 - Cross-validation score: 0.0015 (0.0009)
    >>> np.abs(tr.predict(Xt) - yt).mean()
    0.00126...
    See also
    --------
    TrAdaBoost, TrAdaBoostR2
    References
    ----------
    .. [1] `[1] <https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf>`_ \
D. Pardoe and P. Stone. "Boosting for regression transfer". In ICML, 2010.
    """
    def __init__(self,
                 estimator = None,
                 n_estimators = 10,
                 n_estimators_fs = 10,
                 cv = 5,
                 verbose = 1,
                 random_state = None):
        super().__init__(estimator,
                         n_estimators,
                         verbose,
                         random_state)
        self.n_estimators_fs = n_estimators_fs
        self.cv = cv


    def fit(self, Xs, ys, Xt, yt,
            sample_weight_src = None,
            sample_weight_tgt = None,
            **fit_params):
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        Xs, ys, Xt, yt = check_arrays(Xs, ys, Xt, yt)

        n_s = len(Xs)
        n_t = len(Xt)

        ## When the source and target instances have no initial weights
        if sample_weight_src is None:
            sample_weight_src = np.ones(n_s) / (n_s + n_t)
        if sample_weight_tgt is None:
            sample_weight_tgt = np.ones(n_t) / (n_s + n_t)

        ## When the source and target instances have initial weights
        sum_weights = (sample_weight_src.sum() +
                       sample_weight_tgt.sum())
        sample_weight_src = sample_weight_src / sum_weights
        sample_weight_tgt = sample_weight_tgt / sum_weights

        X = np.concatenate((Xs, Xt)) ## Concatenate X[source + target]
        y = np.concatenate((ys, yt)) ## Concatenate y[source + target]

        self.sample_weights_src_ = []
        self.sample_weights_tgt_ = []
        self.estimators_ = []
        self.estimator_errors_ = []

        for iboost in range(self.n_estimators):
            self.sample_weights_src_.append(np.copy(sample_weight_src))
            self.sample_weights_tgt_.append(np.copy(sample_weight_tgt))

            cv_score = self._cross_val_score(
                Xs, ys, Xt, yt,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )

            if self.verbose >= 1:
                print("Iteration %i - Cross-validation score: %.4f (%.4f)"%
                      (iboost, np.mean(cv_score), np.std(cv_score)))

            self.estimator_errors_.append(cv_score.mean())

            sample_weight_src, sample_weight_tgt = self._boost(
                iboost, Xs, ys, Xt, yt,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )

            if sample_weight_src is None:
                break

            sum_weights = (sample_weight_src.sum() + sample_weight_tgt.sum())

            sample_weight_src = sample_weight_src / sum_weights
            sample_weight_tgt = sample_weight_tgt / sum_weights

        self.estimator_errors_ = np.array(self.estimator_errors_)
        return self


    def _boost(self, iboost, Xs, ys, Xt, yt,
               sample_weight_src, sample_weight_tgt, **fit_params):

        estimator = _AdaBoostR2(estimator=self.base_estimator_,
                                n_estimators=self.n_estimators_fs,
                                verbose=self.verbose-1,
                                random_state=self.random_state)

        if self.verbose > 1:
            print("First Stages...")

        estimator.fit(Xs, ys, Xt, yt,
                      sample_weight_src,
                      sample_weight_tgt,
                      **fit_params)

        ys_pred = estimator.predict(Xs)
        yt_pred = estimator.predict(Xt)

        if ys_pred.ndim == 1:
            ys_pred = ys_pred.reshape(-1, 1)
            yt_pred = yt_pred.reshape(-1, 1)

        error_vect_src = np.abs(ys_pred - ys).mean(tuple(range(1, ys.ndim)))
        error_vect_tgt = np.abs(yt_pred - yt).mean(tuple(range(1, yt.ndim)))
        error_vect = np.concatenate((error_vect_src, error_vect_tgt))

        error_max = error_vect.max() + EPS
        if error_max != 0:
            error_vect /= error_max
            error_vect_src /= error_max
            error_vect_tgt /= error_max

        beta = self._get_beta(iboost,
                              sample_weight_src,
                              sample_weight_tgt,
                              error_vect_src,
                              error_vect_tgt)

        if not iboost == self.n_estimators - 1:
            sample_weight_src *= np.power(
                beta, error_vect_src
            )

        self.estimators_.append(estimator)
        return sample_weight_src, sample_weight_tgt


    def _get_beta(self, iboost, sample_weight_src, sample_weight_tgt,
                  error_vect_src, error_vect_tgt):

        n_s = len(sample_weight_src)
        n_t = len(sample_weight_tgt)

        K_t = (n_t/(n_s + n_t) + (iboost/(self.n_estimators - 1)) *
               (1 - n_t/(n_s + n_t)))
        C_t = sample_weight_tgt.sum() * ((1 - K_t) / K_t)

        def func(x):
            return np.dot(sample_weight_src,
                   np.power(x, error_vect_src)) - C_t
        return _binary_search(func, self.verbose)


    def _cross_val_score(self, Xs, ys, Xt, yt,
                         sample_weight_src, sample_weight_tgt,
                         **fit_params):
        if len(Xt) >= self.cv:
            cv = self.cv
        else:
            cv = len(Xt)

        tgt_index = np.arange(len(Xt))
        split = int(len(Xt) / cv)
        scores = []
        for i in range(cv):
            if i == cv-1:
                test_index = tgt_index[i * split:]
            else:
                test_index = tgt_index[i * split: (i + 1) * split]
            train_index = list(set(tgt_index) - set(test_index))

            estimator = check_estimator(self.base_estimator_,
                                        copy=True, force_copy=True,
                                        display_name="base_estimator_")

            X = np.concatenate((Xs, Xt[train_index]))
            y = np.concatenate((ys, yt[train_index]))

            sample_weight = np.concatenate((sample_weight_src, sample_weight_tgt[train_index]))

            if (len(train_index) > 0 and sample_weight_tgt[train_index].sum() != 0):
                sample_weight[-len(train_index):] *= (sample_weight_tgt.sum() /sample_weight_tgt[train_index].sum())

            if "sample_weight" in inspect.signature(estimator.fit).parameters:
                estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
            else:
                bootstrap_index = np.random.choice(len(X), size=len(X), replace=True, p=sample_weight)
                estimator.fit(X[bootstrap_index], y[bootstrap_index],**fit_params)

            y_pred = estimator.predict(Xt[test_index])

            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            scores.append(np.abs(y_pred - yt[test_index]).mean())
        return np.array(scores)


    def predict(self, X):
        """
        Return predictions of the best estimator according
        to cross-validation scores.

        Parameters
        ----------
        X : array
            Input data.
        Returns
        -------
        y_pred : array
            Best estimator predictions.
        """
        X = check_one_array(X)
        best_estimator = self.estimators_[
            self.estimator_errors_.argmin()]
        return best_estimator.predict(X)


    def predict_weights(self, domain="src"):
        """
        Return sample weights.

        Return the source importance weighting
        of the best estimator.

        Parameters
        ----------
        domain : str (default="tgt")
            Choose between ``"source", "src"`` and
            ``"target", "tgt"``.
        Returns
        -------
        weights : source sample weights
        """
        if hasattr(self, "sample_weights_src_"):
            arg = self.estimator_errors_.argmin()
            if domain in ["src", "source"]:
                return self.sample_weights_src_[arg]
            elif domain in ["tgt", "target"]:
                return self.sample_weights_tgt_[arg]
            else:
                raise ValueError("`domain `argument "
                                 "should be `tgt` or `src`, "
                                 "got, %s"%domain)
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit' first.")
