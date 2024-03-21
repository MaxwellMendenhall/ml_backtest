from strategies.strategy import Strategy, CandleStickPatterns
from machine_learning.data import CandleStickDataProcessing
from datetime import time


class InvertedHammer(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)

    def on_data(self, index, low, high, close, open, dates):
        # Check if the current candlestick is an inverted hammer
        # Ensure index is not 0 to have previous data for trend confirmation
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
            if is_inverted_hammer and in_downtrend and not self.in_position and self.trading_hours(current_date):
                # Make the metadata for the candle stick features for Machine Learning
                trade_metadata = {
                    'current_open': current_open,
                    'current_low': current_low,
                    'current_high': current_high,
                    'current_close': current_close
                }
                # enter at the next candle's open which is current_close
                entry_price = current_close

                # Calculate take profit and stop loss as per your strategy
                # if model is passed into backtest, use model to predict take profit
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
