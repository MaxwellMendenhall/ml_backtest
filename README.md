---
description: >-
  Trading strategy backtesting with machine learning optimizing best exit point
  (aka highest point of trade) for each trade, for maximizing profit.
---

# Backtesting with Machine Learning

This software backtest a trading strategy and runs the results through a user defined machine learning, with feature engineering to optmize the strategy as much as possible. An example output of this software would look like this.

<figure><img src=".gitbook/assets/Screenshot 2024-03-18 at 12.15.50 AM.png" alt=""><figcaption><p>The initial backtest</p></figcaption></figure>

<figure><img src=".gitbook/assets/Screenshot 2024-03-18 at 12.15.54 AM.png" alt=""><figcaption><p>The backtest with the machine learning optimization</p></figcaption></figure>

## How to use

First you need to import all the required classes.

```python
import pandas as pd
from backtest.backtest import Backtest
from strategies.invertedhammer import InvertedHammer
from machine_learning.wrapper import MachineLearning
from models.rfr import RandomForestRegressorTrainer
from machine_learning.data import CandleStickDataProcessing
```

After that make sure you rename your data-frame columns to these names if they are not already named that.

```python
df = pd.read_csv('newalldata.csv')
df = df.rename(columns={'Time': 'date', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low'})
```

After you have your data-frame prepped you can make an instance of the strategy you want and pass it into a Backtest instance.&#x20;

```python
strategy = InvertedHammer()
backtest = Backtest(df, strategy)
print(backtest.results())
```

After that, we have all the data we need for machine learning to take place. Just declare an instance of the machine learning class and pass the need info into it. The machine learning takes place when the `run` function is called on the class. We can dump the model (saving the model to be used as a standalone file) with the `dump_model` function.&#x20;

```python
ml = MachineLearning(ml_class=RandomForestRegressorTrainer,
                         df=df,
                         results=backtest.get_trades())
ml.run()
ml.dump_model('InvertedHammerRFRNasdaq')
```

After we trained the model, we want to backtest the model and see the results! The fun part! Two importnant functions you need to call for the backtest, `get_util` function and `get_data` function. The `get_util` will return a tuple of important values to be passed into the backtest class. The `get_data` will just be the data-frame as before but it includes all necessary added features during the call of `feature_engineering` function of the desired machine learning class.

```python
ml.dump_model(filename='InvertedHammerRFRNasdaq')
model, columns, rows = ml.get_util()
data = ml.get_data()

ml_backtest = Backtest(data, strategy, model=model, columns=columns, rows=rows)
print(ml_backtest.results())
```

Thats it! You should see similar output text wise as the outputs provided above. A more in depth _**how to use**_ guide to customize your machine learning and strategy can be found below.

* [Machine Learning Class Guide](backtesting-with-machine-learning/machine-learning-class-guide.md)
