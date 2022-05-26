import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class DurationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X["duration"] = X["dropOff_datetime"] - X["pickup_datetime"]
        X["duration"] = X["duration"].dt.total_seconds() / 60
        return X


class DurationOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, min_value = 1, max_value = 60, verbose=False):
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        rows_count = X.shape[0]
        X = X.loc[ (X["duration"] >= self.min_value) & (X["duration"] <= self.max_value) ]
        if self.verbose:
            print("Removed", rows_count - X.shape[0], "rows")
        return X, X["duration"]


class IdImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_impute: list, fill_value):
        self.columns_to_impute = columns_to_impute
        self.fill_value = fill_value

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for col in self.columns_to_impute:
            X[col] = X[col].fillna(self.fill_value)
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode: list):
        self.columns_to_encode = columns_to_encode
        self.vectorizer = DictVectorizer(sparse=False)

    def fit(self, X, y=None):
        X = X[self.columns_to_encode].copy()
        X[self.columns_to_encode] = X[self.columns_to_encode].astype(str)
        self.vectorizer.fit(X.to_dict('records'))
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X[self.columns_to_encode].copy()
        X[self.columns_to_encode] = X[self.columns_to_encode].astype(str)
        dicts = X[self.columns_to_encode].to_dict('records')
        array = self.vectorizer.transform(dicts)
        return array