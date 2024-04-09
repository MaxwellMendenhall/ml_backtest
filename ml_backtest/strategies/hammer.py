from ml_backtest.interfaces import Strategy
from ml_backtest.data import CandleStickPatterns
from ml_backtest.machine_learning import CandleStickDataProcessing
from datetime import time


class Hammer(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)

    def on_data(self, index, lows, highs, closes, opens, dates):
        if index > 0:
            current_open = opens[index]
            current_high = highs[index]
            current_low = lows[index]
            current_close = closes[index]
            current_date = dates[index]

            is_hammer = CandleStickPatterns.is_hammer(current_open=current_open,
                                                      current_low=current_low,
                                                      current_high=current_high,
                                                      current_close=current_close)

            in_downtrend = current_close < closes[index - 1]

            if is_hammer and in_downtrend and not self.in_position and self.trading_hours(current_date):
                trade_metadata = {
                    'current_open': current_open,
                    'current_close': current_close,
                    'current_low': current_low,
                    'current_high': current_high
                }
                entry_price = current_close

                if self.model is not None:
                    if self.cs_patterns:
                        np_l = CandleStickDataProcessing.calculate_hammer_features(**trade_metadata)
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
