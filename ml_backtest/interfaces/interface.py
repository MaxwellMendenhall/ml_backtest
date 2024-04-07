import pandas as pd
from ml_backtest.machine_learning import DataProcessing
from datetime import datetime
from typing import final
from sklearn.base import BaseEstimator
from typing import Optional, List
import numpy as np


class MachineLearningInterface:

    def __init__(self, data: pd.DataFrame,
                 rows: Optional[int] = None,
                 columns: Optional[List[str]] = None):

        if type(self).get_model != MachineLearningInterface.get_model:
            raise TypeError("get_model method should not be overridden")

        if isinstance(data, pd.DataFrame):
            self.data = data
            self.model = None
            self.predictions = None
            self.get_rows = rows if rows is not None else 10
            self.get_columns = columns if columns is not None else ['close']

        else:
            print('Data being passed into MachineLearningWorker is not a list of type DataContainer.')

    def feature_engineer(self):
        """
        Create new features or change current ones. Method will be called
        before training and predict. Make sure to do all desired changes in here.

        :return:
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    def train(self, x_train, y_train, x_test, y_test):
        """
        Will be called after feature engineer.
        Define desired model in here. Fit model in here also.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    def predict(self, x_train, y_train, x_test, y_test):
        """
        Will be called after train. Used to specifying current statistics
        for the trained mode.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    @final
    def get_model(self):
        """
        Returns the model used in training.

        :return: model
        """
        return self.model


class TargetInterface:
    def __init__(self, trades: pd.DataFrame, data: pd.DataFrame):
        self.trades = trades
        self.data = data

    def target_engineer(self):
        """
        Will be called for defining the target values. If not defined
        default target calculation (high - entry diff) will take
        precedent.
        """
        pass


class Strategy:
    def __init__(self):
        self.positions = []
        self.in_position = False
        self.model = None
        self.__columns = None
        self.__rows = None
        self.__df = None
        self.market_open_time = None
        self.market_close_time = None
        self.cs_patterns = False

    def init(self):
        raise NotImplementedError("Method 'init()' must be defined in the subclass.")

    def on_data(self, index, low, high, close, open, dates):
        """This method will be called for each row of data.
        Override this method with the strategy logic.
        """
        raise NotImplementedError("Method 'on_data()' must be defined in the subclass.")

    @final
    def buy(self, price, take_profit=None, stop_loss=None, entry_time=None, metadata=None):
        if not self.in_position:  # Check if not already in a position
            position = {
                'type': 'long',
                'entry price': price,
                'take profit': take_profit,
                'stop loss': stop_loss,
                'entry time': entry_time
            }
            if metadata is not None:
                position['metadata'] = metadata
            self.positions.append(position)
            self.in_position = True

    @final
    def sell(self, price, take_profit=None, stop_loss=None, entry_time=None, metadata=None):
        if not self.in_position:  # Check if not already in a position
            position = {
                'type': 'short',
                'entry price': price,
                'take profit': take_profit,
                'stop loss': stop_loss,
                'entry time': entry_time
            }
            if metadata is not None:
                position['metadata'] = metadata
            self.positions.append(position)
            self.in_position = True

    @final
    def set_ml(self, model: Optional[BaseEstimator] = None, columns=None, rows=None, df=None, cs_pattern=False):
        self.model = model
        self.__columns = columns
        self.__rows = rows
        self.__df = df
        self.cs_patterns = cs_pattern

    @final
    def predict(self, current_entry_time: int, cs_features: np.ndarray = None):
        # Ensure that all necessary parameters are set
        if self.model is None or self.__columns is None or self.__rows is None or self.__df is None:
            raise ValueError("Model, columns, rows, or DataFrame not set")

        # Prepare the data
        column_indices = [self.__df.columns.get_loc(c) for c in self.__columns]
        # Convert the single entry time into an array for compatibility
        entry_times = np.array([current_entry_time])
        # Call the static method from DataProcessing class
        if cs_features is not None:
            processed_data = DataProcessing.process_entries(self.__df.to_numpy(), entry_times, self.__rows,
                                                            column_indices, candlestick_features=cs_features)
        else:
            processed_data = DataProcessing.process_entries(self.__df.to_numpy(), entry_times, self.__rows,
                                                            column_indices)
        # Reshape the processed data if necessary to match the input shape expected by the model
        prediction = self.model.predict(processed_data)

        if prediction.size == 1:
            value_as_float = float(prediction.item())
        else:
            raise ValueError(
                "The array contains more than one element and cannot be directly converted to a single float.")
        return value_as_float

    @final
    def trading_hours(self, date: str) -> bool:

        if isinstance(date, np.int64):  # Checking for Unix timestamp (ml backtest)
            dt_object = datetime.utcfromtimestamp(date)
            current_time = dt_object.time()
        elif isinstance(date, str):  # Checking for str object (normal backtest)
            current_datetime = datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p")
            current_time = current_datetime.time()
        else:  # If date is neither return false
            return False

        # check if the converted time falls between user inputted open and close time
        return self.market_open_time <= current_time <= self.market_close_time
