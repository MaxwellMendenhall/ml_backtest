from ml_backtest.interfaces import Strategy
from ml_backtest.data import CandleStickPatterns
from ml_backtest.machine_learning import CandleStickDataProcessing
from datetime import time


class MorningStarDoji(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)

    def on_data(self, index, lows, highs, closes, opens, dates):
        # Ensure there are at least two previous candles to compare with
        if index > 1:
            # Previous period data
            prev_open = opens[index - 1]
            prev_close = closes[index - 1]
            prev_high = highs[index - 1]
            prev_low = lows[index - 1]

            # Before previous period data
            b_prev_open = opens[index - 2]
            b_prev_close = closes[index - 2]
            b_prev_high = highs[index - 2]
            b_prev_low = lows[index - 2]

            # Current period data
            current_open = opens[index]
            current_close = closes[index]
            current_high = highs[index]
            current_low = lows[index]
            current_date = dates[index]

            # Check for Morning Star Doji pattern
            is_morning_star_doji = (b_prev_close < b_prev_open and
                                    abs(b_prev_close - b_prev_open) / (0.001 + b_prev_high - b_prev_low) >= 0.7 and
                                    abs(prev_close - prev_open) / (0.001 + prev_high - prev_low) < 0.1 and
                                    current_close > current_open and
                                    abs(current_close - current_open) / (0.001 + current_high - current_low) >= 0.7 and
                                    b_prev_close > prev_close and
                                    b_prev_close > prev_open and
                                    prev_close < current_open and
                                    prev_open < current_open and
                                    current_close > b_prev_close and
                                    (prev_high - max(prev_close, prev_open)) > (3 * abs(prev_close - prev_open)) and
                                    (min(prev_close, prev_open) - prev_low) > (3 * abs(prev_close - prev_open)))

            # Enter a long position if a Morning Star Doji pattern is detected
            if is_morning_star_doji and not self.in_position and self.trading_hours(current_date):
                # Assuming entry at the current close
                entry_price = current_close
                # Calculate take profit and stop loss as per your strategy
                if self.model is not None:
                    prediction = self.predict(current_date)
                    take_profit = entry_price + prediction
                else:
                    take_profit = entry_price + 50
                stop_loss = entry_price - 37
                entry_time = dates[index]

                self.buy(price=entry_price, take_profit=take_profit, stop_loss=stop_loss, entry_time=entry_time)
