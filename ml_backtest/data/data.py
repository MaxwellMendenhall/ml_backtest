import pandas as pd


class CandleStickPatterns:
    @staticmethod
    def is_inverted_hammer(current_open, current_close, current_high, current_low):
        body_length = abs(current_open - current_close)
        upper_shadow = current_high - max(current_open, current_close)
        total_length = current_high - current_low
        lower_shadow = min(current_open, current_close) - current_low

        return (((current_high - current_low) > 3 * body_length) and
                (upper_shadow / (0.001 + total_length) > 0.6) and
                (lower_shadow / (0.001 + total_length) < 0.4))

    @staticmethod
    def is_bullish_engulfing(current_open, current_close, prev_open, prev_close):
        return (current_close >= prev_open > prev_close >= current_open and
                current_close > current_open and
                current_close - current_open > prev_open - prev_close)


class Data:
    @staticmethod
    def data():
        df = pd.read_csv('./last_4000_rows.csv')
        return df
