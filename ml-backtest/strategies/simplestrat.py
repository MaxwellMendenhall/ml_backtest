from strategies.strategy import Strategy


class SimpleStrategy(Strategy):

    def on_data(self, index, low, high, close, open, dates):
        # print(f"Current Index: {index}")
        # print("Dates up to current index: ", dates[:index + 1])
        # print("Opens up to current index: ", open[:index + 1])
        # print("Highs up to current index: ", high[:index + 1])
        # print("Lows up to current index: ", low[:index + 1])
        # print("Closes up to current index: ", close[:index + 1])

        current_close = close[index]
        current_open = open[index]
        current_date = dates[index]

        if not self.in_position:  # Only attempt to enter a position if not already in one
            if current_close > current_open:
                self.buy(price=current_close,
                         take_profit=current_close * 1.05,
                         stop_loss=current_close * 0.95,
                         entry_time=current_date)
            elif current_close < current_open:
                self.sell(price=current_close,
                          take_profit=current_close * 0.95,
                          stop_loss=current_close * 1.05,
                          entry_time=current_date)