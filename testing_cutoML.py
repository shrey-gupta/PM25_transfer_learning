from sklearn import datasets
from sklearn.model_selection import train_test_split
from cutoml.cutoml import CutoRegressor

if __name__ == "__main__":
    dataset = datasets.load_boston()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=0
    )

    ctr = CutoRegressor(k_folds=3, n_jobs=-1, verbose=0) #, random_state=0)
    ctr.fit(X=X_train, y=y_train)
    print(ctr.score(X = X_test, y = y_test))
    print(ctr.best_estimator.named_steps[AdaBoostRegressor].best_estimator_)

#regression_model
