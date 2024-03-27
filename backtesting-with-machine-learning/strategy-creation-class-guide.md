# Strategy Creation Class Guide

## Setup

Everything you need to know in order to start making your own strategies with machine learning. In this guide we will make a inverted hammer candle stick pattern strategy that will take trades every time a inverted hammer is formed. You can find the source code for this guide in the `strategies` dir.

To start you want to import the interface to make a child class for the backtest to use.

```python
from strategies.strategy import Strategy
```

After that we can created a class, name it whatever you want. We want to define the mandatory methods in this class which are `init()` and `on_data()`.

```python
class InvertedHammer(Strategy):
    def init(self):
        pass
    def on_data(self, index, low, high, close, open, dates):
        pass
```

Now it is time to fill in these methods. The `init()` is fairly simple to implement. We want to define trading hours in here for the strategy to run in between. You want to define your desired hours with military time where the first param of `time()` is the hour and the second is the minute. Make sure to import `datetime`.&#x20;

```python
from strategies.strategy import Strategy
from datetime import time

class InvertedHammer(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)
        
    def on_data(self, index, low, high, close, open, dates):
        pass
```

## Machine Learning with Backtest

```python
from strategies.strategy import Strategy
from datetime import time

class InvertedHammer(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)
        
    def on_data(self, index, low, high, close, open, dates):
        ...
        
        ...
```

## on\_data() Trading Logic

Now it is time to implement the `on_data()`. Here is where the bulk of the trading logic is done. `on_data()` is called on every row of new data, treating it like the close to simulate real market conditions. You will have access to previous and current data, meaning you cannot see the data in the future. You can operate any trading strategy you want as long as it does not reside on limit orders. Market orders are the only orders that this backtest software works with (this is due to the complexity of brokerages and actually implementing limit orders would reside on tick data which only brokerages have). We use a new import called CandleStickPatterns that holds all the logic for detecting candle stick patterns. The methods just return a simple boolean variable (true or fasle) if a pattern is detected or not.

```python
from strategies.strategy import Strategy, CandleStickPatterns
from datetime import time

class InvertedHammer(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)
        
    def on_data(self, index, low, high, close, open, dates):
        if index > 0:
            current_close = close[index]
            current_open = open[index]
            current_low = low[index]
            current_high = high[index]
            current_date = dates[index]

            is_inverted_hammer = CandleStickPatterns.is_inverted_hammer(current_open=current_open,
                                                                        current_low=current_low,
                                                                        current_high=current_high,
                                                                        current_close=current_close)

            # To ensure it's at the bottom of a downtrend, check if the current close is less than the previous close
            in_downtrend = current_close < close[index - 1]

            # Enter a long position if an inverted hammer is detected at the bottom of a downtrend
            # and within trading hours
            if is_inverted_hammer and in_downtrend and not self.in_position and self.trading_hours(current_date):
                ...
                
                
```

All the trading logic to decide if a trade should be entered or not is defined above for inverted hammer. Notice we have a function called `trading_hours()` that resides in the super class. Trading hours and the `market_open_time` and `market_close_time` work in tandem. The `...` is where we will implement the actual position generation in the following steps.&#x20;

## Machine Learning with Backtest

When you want to backtest with the machine learning model and use the same strategy we need to add a simple condition check to check of a model is available or not for the strategy to use. We can use the `model` variable provided from the super class.&#x20;

```python
from strategies.strategy import Strategy, CandleStickPatterns
from machine_learning.data import CandleStickDataProcessing
from datetime import time

class InvertedHammer(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)
        
    def on_data(self, index, low, high, close, open, dates):
        if index > 0:
            current_close = close[index]
            current_open = open[index]
            current_low = low[index]
            current_high = high[index]
            current_date = dates[index]

            is_inverted_hammer = CandleStickPatterns.is_inverted_hammer(current_open=current_open,
                                                                        current_low=current_low,
                                                                        current_high=current_high,
                                                                        current_close=current_close)

            # To ensure it's at the bottom of a downtrend, check if the current close is less than the previous close
            in_downtrend = current_close < close[index - 1]

            # Enter a long position if an inverted hammer is detected at the bottom of a downtrend
            # and within trading hours
            if is_inverted_hammer and in_downtrend and not self.in_position and self.trading_hours(current_date):
            
                if self.model is not None:
                    if self.cs_patterns:
                        np_l = CandleStickDataProcessing.calculate_inverted_hammer_features(**trade_metadata)
                        np_l_2d = np_l.reshape(1, -1)
                        prediction = self.predict(current_date, np_l_2d)
                        take_profit = entry_price + prediction
                    else:
                        prediction = self.predict(current_date)
                        take_profit = entry_price + prediction
                else:
                    take_profit = entry_price + 50
                stop_loss = entry_price - 37
                entry_time = dates[index]
                 
                ...
                
                
```

In the provided code inside the condtional check of `self.model` we check if a machine learning model is passed into tha backtest, if it is run some calculations for the take profit.&#x20;

## Buy and Sell Methods

You make longs and shorts with the `buy()` and `sell()` methods provided. The buy and sell methods require some data to be passed in. You can see how they are implemented below.&#x20;



| price                                                                     | take\_profit                                                  | stop\_loss                                                          | entry\_time                                        | meta\_data                                                                                                       |
| ------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| The price at which you want the strategy to enter at in the current data. | Number of points from price you want the strategy to exit at. | Number of points from price you want the strategy to cut losses at. | Time of current data that a position is filled at. | Any meta data you want to use during the machine learning process that can only be calculated at entry of trade. |

For inverted hammer this is the actual implementation of the `buy` method we will be using.

```python
from strategies.strategy import Strategy, CandleStickPatterns
from machine_learning.data import CandleStickDataProcessing
from datetime import time

class InvertedHammer(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)
        
    def on_data(self, index, low, high, close, open, dates):
        if index > 0:
            current_close = close[index]
            current_open = open[index]
            current_low = low[index]
            current_high = high[index]
            current_date = dates[index]

            is_inverted_hammer = CandleStickPatterns.is_inverted_hammer(current_open=current_open,
                                                                        current_low=current_low,
                                                                        current_high=current_high,
                                                                        current_close=current_close)

            # To ensure it's at the bottom of a downtrend, check if the current close is less than the previous close
            in_downtrend = current_close < close[index - 1]

            # Enter a long position if an inverted hammer is detected at the bottom of a downtrend
            # and within trading hours
            if is_inverted_hammer and in_downtrend and not self.in_position and self.trading_hours(current_date):
            
                if self.model is not None:
                    if self.cs_patterns:
                        np_l = CandleStickDataProcessing.calculate_inverted_hammer_features(**trade_metadata)
                        np_l_2d = np_l.reshape(1, -1)
                        prediction = self.predict(current_date, np_l_2d)
                        take_profit = entry_price + prediction
                    else:
                        prediction = self.predict(current_date)
                        take_profit = entry_price + prediction
                else:
                    take_profit = entry_price + 50
                stop_loss = entry_price - 37
                entry_time = dates[index]
                 
                self.buy(price=entry_price, take_profit=take_profit,
                         stop_loss=stop_loss, entry_time=entry_time, metadata=trade_metadata)
                
                
```

Congrats! You just made a inverted hammer strategy that implments machine learning!
