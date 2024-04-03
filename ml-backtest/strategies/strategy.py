from sklearn.base import BaseEstimator
from typing import Optional
import numpy as np
from machine_learning.data import DataProcessing, CandleStickDataProcessing
from datetime import datetime
from typing import final


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


class CandleStickPatterns:
    @staticmethod
    def is_inverted_hammer(current_open, current_close, current_high, current_low):
        body_length = abs(current_open - current_close)
        upper_shadow = current_high - max(current_open, current_close)
        total_length = current_high - current_low
        lower_shadow = min(current_open, current_close) - current_low

        return (((current_high - current_low) > 3 * body_length) and
                (upper_shadow / (0.001 + total_length) > 0.6) and
                (lower_shadow / (0.001 + total_length) < 0.4))

    @staticmethod
    def is_bullish_engulfing(current_open, current_close, prev_open, prev_close):
        return (current_close >= prev_open > prev_close >= current_open and
                current_close > current_open and
                current_close - current_open > prev_open - prev_close)
