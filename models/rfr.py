from machine_learning.interface import MachineLearningInterface, TargetInterface
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import talib


class RandomForestRegressorTrainer(MachineLearningInterface):

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        # this is the number of rows before each target you want trained
        self.get_rows = 10
        # these are the columns you want trained, in my case I want the column
        # that is already there and a column I am adding in the feature_engineer()
        # method

        # 'EMA_Diff', 'SMA_Diff', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist'
        self.get_columns = ['EMA_Diff', 'SMA_Diff', 'MACD_hist']

    def feature_engineer(self):
        # here is where you can add addition columns of features you want to be used in training
        # just make sure you edit the 'self.data' with the features you want as that is the dataframe being
        # used in training
        self.data['SMA'] = self.data['close'].rolling(window=10).mean()
        self.data['EMA'] = talib.EMA(self.data['close'], timeperiod=10)
        self.data['RSI'] = talib.RSI(self.data['close'], timeperiod=14)
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(self.data['close'],
                                                                                         fastperiod=12, slowperiod=26,
                                                                                         signalperiod=9)
        self.data['SMA_Diff'] = self.data['SMA'].diff()
        self.data['EMA_Diff'] = self.data['EMA'].diff()

    def train(self, x_train, y_train, x_test, y_test):
        # here is where you define the model you want for training
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(x_train, y_train)

    def predict(self, x_train, y_train, x_test, y_test):
        # here is where the predictions will appear
        # you can get different values for the predictions
        # like calculating the residuals
        self.predictions = self.model.predict(x_test)

        mse = mean_squared_error(y_test, self.predictions)
        print(f"Mean Squared Error: {mse}")


class BasicTarget(TargetInterface):
    def target_engineer(self):
        min_value = self.trades['target'].min()
        self.trades['target'] = self.trades['target'].apply(lambda x: min(x, 15))
        self.trades['target'] = self.trades['target'].apply(lambda x: max(x, min_value))

        print(self.trades.to_string())
