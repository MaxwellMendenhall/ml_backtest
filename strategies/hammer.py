from strategies.strategy import Strategy
from datetime import time


class Hammer(Strategy):
    def init(self):
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 10)

    def on_data(self, index, lows, highs, closes, opens, dates):
        # Ensure there's at least one previous candle to compare with for trend analysis
        if index > 0:
            current_open = opens[index]
            current_high = highs[index]
            current_low = lows[index]
            current_close = closes[index]
            current_date = dates[index]

            # Check for Hammer pattern
            is_hammer = (((current_high - current_low) > 3 * (current_open - current_close)) and
                         ((current_close - current_low) / (0.001 + current_high - current_low) > 0.6) and
                         ((current_open - current_low) / (0.001 + current_high - current_low) > 0.6))

            # To ensure it's at the bottom of a downtrend, check if the current close is less than the previous close
            in_downtrend = current_close < closes[index - 1]

            # Enter a long position if a Hammer pattern is detected at the bottom of a downtrend
            if is_hammer and in_downtrend and not self.in_position and self.trading_hours(current_date):
                # Assuming entry at the next candle's open, which is the current close
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
