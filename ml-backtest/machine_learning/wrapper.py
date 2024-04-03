import numpy as np
from joblib import dump
from machine_learning.interface import MachineLearningInterface, TargetInterface
from sklearn.model_selection import train_test_split
from machine_learning.data import DataProcessing
from typing import Type
import pandas as pd
import os


class MachineLearning:

    def __init__(self, ml_class: Type[MachineLearningInterface],
                 df: pd.DataFrame, results: pd.DataFrame,
                 target_class: Type[TargetInterface] = None):
        self.__df = df
        self.__results = results
        if target_class is not None:
            self.__target_class = target_class(trades=self.__results, data=self.__df)
            self.__target_class.target_engineer()
            self.__results = self.__target_class.trades
        self.__ml = ml_class(self.__df)

    def run(self, dp_pattern=None) -> None:
        self.__ml.feature_engineer()

        dp = DataProcessing(df=self.__df, results=self.__results,
                            rows=self.__ml.get_rows, columns=self.__ml.get_columns)

        if dp_pattern is not None:
            dp.add_pattern_features(dp_pattern=dp_pattern)

        X = dp.get_before()
        Xr = np.around(X, decimals=4)
        y = dp.get_target()

        X_train, X_test, y_train, y_test = \
            train_test_split(Xr, y, test_size=0.2, random_state=42)

        self.__train(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    def __train(self, X_train, X_test, y_train, y_test) -> None:
        print("\033[91mTraining in progress...\033[0m")
        if y_train is not None or X_train is not None:
            self.__ml.train(x_train=X_train,
                            y_train=y_train,
                            x_test=X_test,
                            y_test=y_test)
            self.__ml.predict(x_train=X_train,
                              y_train=y_train,
                              x_test=X_test,
                              y_test=y_test)

    def get_data(self) -> pd.DataFrame:
        return self.__df

    def get_util(self):
        return self.__ml.get_model(), self.__ml.get_columns, self.__ml.get_rows

    def dump_model(self, filename):
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_filename = os.path.join(models_dir, filename + '.joblib')
        dump(self.__ml.get_model(), model_filename)
        print(f"\033[91mModel saved to {model_filename}\033[0m")
