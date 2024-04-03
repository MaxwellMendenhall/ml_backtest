import pandas as pd
from ml_backtest import Backtest, MachineLearning
from ml_backtest.strategies import InvertedHammer
from ml_backtest.models import RandomForestRegressorTrainer
from ml_backtest.machine_learning import CandleStickDataProcessing
from ml_backtest.data import Data


def test_backtesting_with_machine_learning():
    df = Data.data().rename(columns={'Time': 'date', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low'})

    strategy = InvertedHammer()

    backtest = Backtest(df, strategy)
    backtest_results = backtest.results()
    assert backtest_results is not None

    ml = MachineLearning(ml_class=RandomForestRegressorTrainer,
                         df=df,
                         results=backtest.get_trades())
    ml.run(dp_pattern=CandleStickDataProcessing.calculate_inverted_hammer_features)

    model, columns, rows = ml.get_util()
    assert model is not None

    data = ml.get_data()
    assert not data.empty

    ml_backtest = Backtest(data, strategy, model=model, columns=columns, rows=rows, cs_pattern=True)
    ml_backtest_results = ml_backtest.results()
    assert ml_backtest_results is not None
