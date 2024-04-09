from ml_backtest.interfaces import Strategy
from ml_backtest.data import CandleStickPatterns
from ml_backtest.machine_learning import CandleStickDataProcessing
from datetime import time


class MorningStarDoji(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)

    def on_data(self, index, lows, highs, closes, opens, dates):
        if index > 1:
            prev_open = opens[index - 1]
            prev_close = closes[index - 1]
            prev_high = highs[index - 1]
            prev_low = lows[index - 1]

            b_prev_open = opens[index - 2]
            b_prev_close = closes[index - 2]
            b_prev_high = highs[index - 2]
            b_prev_low = lows[index - 2]

            current_open = opens[index]
            current_close = closes[index]
            current_high = highs[index]
            current_low = lows[index]
            current_date = dates[index]

            is_morning_star_doji = CandleStickPatterns.is_morning_star_doji(b_prev_open=b_prev_open,
                                                                            b_prev_low=b_prev_low,
                                                                            b_prev_high=b_prev_high,
                                                                            b_prev_close=b_prev_close,
                                                                            prev_open=prev_open,
                                                                            prev_close=prev_close,
                                                                            prev_high=prev_high,
                                                                            prev_low=prev_low,
                                                                            current_close=current_close,
                                                                            current_high=current_high,
                                                                            current_low=current_low,
                                                                            current_open=current_open)

            if is_morning_star_doji and not self.in_position and self.trading_hours(current_date):
                trade_metadata = {
                    'b_prev_open': b_prev_open,
                    'b_prev_close': b_prev_close,
                    'prev_high': prev_high,
                    'prev_low': prev_low,
                    'current_open': current_open,
                    'current_close': current_close
                }
                entry_price = current_close

                if self.model is not None:
                    if self.cs_patterns:
                        np_l = CandleStickDataProcessing.calculate_morning_star_doji_features(**trade_metadata)
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
