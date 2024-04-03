from ml_backtest.interfaces import Strategy
from ml_backtest.data import CandleStickPatterns
from ml_backtest.machine_learning import CandleStickDataProcessing
from datetime import time


class BullishEngulfing(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)

    def on_data(self, index, low, high, close, open, dates):
        if index > 0:
            current_close = close[index]
            current_open = open[index]
            current_date = dates[index]
            prev_close = close[index - 1]
            prev_open = open[index - 1]

            is_bullish_engulfing = CandleStickPatterns.is_bullish_engulfing(current_open=current_open,
                                                                            current_close=current_close,
                                                                            prev_open=prev_open,
                                                                            prev_close=prev_close)

            # Enter a long position if a bullish engulfing pattern is detected
            if is_bullish_engulfing and not self.in_position and self.trading_hours(current_date):
                # Make the metadata for the candle stick features for Machine Learning
                trade_metadata = {
                    'current_open': current_open,
                    'current_close': current_close,
                    'prev_open': prev_open,
                    'prev_close': prev_close
                }
                # Assuming you want to enter at the next candle's open, which is current_close
                entry_price = current_close
                # Calculate take profit and stop loss as per your strategy
                if self.model is not None:
                    if self.cs_patterns:
                        np_l = CandleStickDataProcessing.calculate_engulfing_features(**trade_metadata)
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

                self.buy(price=entry_price, take_profit=take_profit, stop_loss=stop_loss,
                         entry_time=entry_time, metadata=trade_metadata)
