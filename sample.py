from ml_backtest import Backtest, MachineLearning
from ml_backtest.strategies import InvertedHammer
from ml_backtest.models import RandomForestRegressorTrainer
from ml_backtest.machine_learning import CandleStickDataProcessing
from ml_backtest.data import Data

if __name__ == '__main__':
    #
    # This is an example for backtesting a inverted hammer strategy with machine learning
    #

    # Make sure your csv file has OHLC with dates. Rename them like shown below
    # df = pd.read_csv('YOUR CSV FILE NAME.csv')
    # df = df.rename(columns={'Time': 'date', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low'})
    strategy = InvertedHammer()

    backtest = Backtest(Data.data(), strategy)
    print(backtest.get_results())

    ml = MachineLearning(ml_class=RandomForestRegressorTrainer,
                         df=Data.data(),
                         results=backtest.get_trades(),
                         rows=10,
                         columns=['EMA_Diff', 'SMA_Diff', 'MACD_hist'])
    ml.run(dp_pattern=CandleStickDataProcessing.calculate_inverted_hammer_features)
    ml.dump_model(filename='YOUR MODEL FILE NAME')
    model, columns, rows = ml.get_util()
    data = ml.get_data()

    # make sure to pass ml.get_data() in as the data for ml backtesting
    ml_backtest = Backtest(data, strategy, model=model, columns=columns, rows=rows, cs_pattern=True)
    print(ml_backtest.get_results())
