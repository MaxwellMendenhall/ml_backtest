import pandas as pd
from backtest.backtest import Backtest
from strategies.invertedhammer import InvertedHammer
from machine_learning.wrapper import MachineLearning
from models.rfr import RandomForestRegressorTrainer
from machine_learning.data import CandleStickDataProcessing

if __name__ == '__main__':
    #
    # This is an example for backtesting a inverted hammer strategy with machine learning
    #
    df = pd.read_csv('YOUR CSV FILE WITH DATA.csv')

    # Make sure your csv file has OHLC with dates. Rename them like shown below
    df = df.rename(columns={'Time': 'date', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low'})
    strategy = InvertedHammer()

    backtest = Backtest(df, strategy)
    print(backtest.results())

    ml = MachineLearning(ml_class=RandomForestRegressorTrainer,
                         df=df,
                         results=backtest.get_trades())
    ml.run(dp_pattern=CandleStickDataProcessing.calculate_inverted_hammer_features)
    ml.dump_model(filename='YOUR FILE NAME')
    model, columns, rows = ml.get_util()
    data = ml.get_data()

    # make sure to pass ml.get_data() in as the data for ml backtesting
    ml_backtest = Backtest(data, strategy, model=model, columns=columns, rows=rows, cs_pattern=True)
    print(ml_backtest.results())
