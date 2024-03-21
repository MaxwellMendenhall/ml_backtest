from strategies.strategy import Strategy
from datetime import time


class PiercingPattern(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)

    def on_data(self, index, lows, highs, closes, opens, dates):
        # Ensure there's at least one previous candle to compare with
        if index > 0:
            current_open = opens[index]
            current_close = closes[index]
            current_date = dates[index]
            prev_open = opens[index - 1]
            prev_close = closes[index - 1]
            prev_low = lows[index - 1]

            # Check for Piercing Pattern
            is_piercing_pattern = (prev_close < prev_open and
                                   current_open < prev_low and
                                   prev_open > current_close > prev_close + ((prev_open - prev_close) / 2))

            # Enter a long position if a Piercing Pattern is detected
            if is_piercing_pattern and not self.in_position and self.trading_hours(current_date):
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
