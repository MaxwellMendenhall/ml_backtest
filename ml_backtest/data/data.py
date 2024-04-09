import pandas as pd
import os


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

    @staticmethod
    def is_bullish_harami(current_open, current_close, prev_open, prev_close):
        return (prev_open > prev_close and
                prev_close <= current_open < current_close <= prev_open and
                current_close - current_open < prev_open - prev_close)

    @staticmethod
    def is_dragonfly_doji(current_open, current_close, current_high, current_low):
        # Adjusted calculation to avoid division by zero
        body_range = abs(current_close - current_open)
        total_range = current_high - current_low
        upper_shadow = current_high - max(current_close, current_open)
        lower_shadow = min(current_close, current_open) - current_low

        # Check for Dragonfly Doji pattern with safeguards against division by zero
        return (body_range / (total_range + 0.001) < 0.1 and
                lower_shadow > (3 * body_range) and
                upper_shadow < body_range)

    @staticmethod
    def is_hammer(current_open, current_close, current_high, current_low):
        # Check for Hammer pattern
        return (((current_high - current_low) > 3 * abs(current_open - current_close)) and
                ((current_close - current_low) / (0.001 + current_high - current_low) > 0.6) and
                ((current_open - current_low) / (0.001 + current_high - current_low) > 0.6))

    @staticmethod
    def is_morning_star(b_prev_open, b_prev_close, prev_open, prev_close, current_open, current_close):
        # Check for Morning Star pattern
        return (max(b_prev_open, b_prev_close) < prev_close < prev_open and
                current_close > current_open > max(b_prev_open, b_prev_close))

    @staticmethod
    def is_morning_star_doji(b_prev_open, b_prev_close, b_prev_high, b_prev_low,
                             prev_open, prev_close, prev_high, prev_low,
                             current_open, current_close, current_high, current_low):
        return (b_prev_close < b_prev_open and
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

    @staticmethod
    def is_piercing_pattern(prev_open, prev_close, prev_low, current_open, current_close):
        return (prev_close < prev_open and
                current_open < prev_low and
                prev_open > current_close > prev_close + ((prev_open - prev_close) / 2))


class Data:
    @staticmethod
    def data():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        csv_file_path = os.path.join(dir_path, 'last_4000_rows.csv')
        df = pd.read_csv(csv_file_path)
        return df
