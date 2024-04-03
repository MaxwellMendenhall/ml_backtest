from ml_backtest.interfaces import Strategy
from ml_backtest.data import CandleStickPatterns
from ml_backtest.machine_learning import CandleStickDataProcessing
from datetime import time


class DragonFlyDoji(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)

    def on_data(self, index, lows, highs, closes, opens, dates):
        if index > 0:  # Ensure there's at least one candle to compare against
            current_open = opens[index]
            current_close = closes[index]
            current_high = highs[index]
            current_low = lows[index]
            current_date = dates[index]

            # Adjusted calculation to avoid division by zero
            body_range = abs(current_close - current_open)
            total_range = current_high - current_low
            upper_shadow = current_high - max(current_close, current_open)
            lower_shadow = min(current_close, current_open) - current_low

            # Check for Dragonfly Doji pattern with safeguards against division by zero
            is_dragonfly_doji = (body_range / (total_range + 0.001) < 0.1 and
                                 lower_shadow > (3 * body_range) and
                                 upper_shadow < body_range)

            if is_dragonfly_doji and not self.in_position and self.trading_hours(current_date):
                entry_price = current_close  # Assuming entry at the close of the Dragonfly Doji
                if self.model is not None:
                    prediction = self.predict(current_date)
                    take_profit = entry_price + prediction
                else:
                    take_profit = entry_price + 50
                stop_loss = entry_price - 37
                entry_time = dates[index]

                self.buy(price=entry_price, take_profit=take_profit, stop_loss=stop_loss, entry_time=entry_time)
