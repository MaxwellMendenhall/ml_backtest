import numpy as np
import pandas as pd


class DataProcessing:
    def __init__(self, df, results, rows, columns):
        """
        Initialize the class with the dataframe, results, and other parameters.
        Data alteration will happen in the `__prep_data` function.
        """
        self.__df = df
        self.__results = results
        self.__rows = rows
        self.__columns = columns
        self.__candlestick_features = None

        # Initialize placeholders for transformed data
        self.__target = None
        self.__all_data = None
        self.__entry_times = None
        self.__columns_indices = None

    def __prep_data(self):
        """
        Prepare data by converting dates to Unix timestamps and processing as required.
        """
        # Convert 'date' in df and 'entry time' in results to Unix timestamp
        self.__df['date'] = pd.to_datetime(self.__df['date'], format='%m/%d/%Y %I:%M:%S %p').astype('int64') // 10 ** 9
        self.__results['entry time'] = pd.to_datetime(self.__results['entry time'],
                                                      format='%m/%d/%Y %I:%M:%S %p').astype(
            'int64') // 10 ** 9

        df_np = self.__df.to_numpy()  # Including 'date' as Unix timestamps

        # Convert 'entry time' & 'high - entry diff' to numpy array
        entry_times_np = self.__results['entry time'].values
        high_np = self.__results['high - entry diff'].values

        # Adjust column indices to include 'date' column
        # Subtracting 1 to adjust for 0-based indexing in numpy
        column_indices = [self.__df.columns.get_loc(c) - 1 for c in self.__columns]

        # Store processed data
        self.__target = high_np
        self.__all_data = df_np
        self.__entry_times = entry_times_np
        self.__columns_indices = column_indices

    @staticmethod
    def process_entries(all_data, entry_times, rows, columns_indices, candlestick_features=None) -> np.ndarray:
        """
        Static method to process data entries, making it reusable for different data sources.

        :param candlestick_features:
        :param all_data: Numpy array of all data, including 'date' as Unix timestamps.
        :param entry_times: Numpy array of entry times.
        :param rows: Number of rows to include before each entry time.
        :param columns_indices: Indices of the columns to include in the output.
        :return: Processed data as a numpy array.
        """
        output_data = []
        for i, entry_time in enumerate(entry_times):
            indices = np.where(all_data[:, 0] <= entry_time)[0]
            if indices.size > 0:
                matching_index = indices[-1]
            else:
                continue

            start_index = max(0, matching_index - rows + 1)
            before_data = all_data[start_index:matching_index + 1, columns_indices]
            single_row = before_data.flatten()

            # Append candlestick features if available
            if candlestick_features is not None and i < len(candlestick_features):
                single_row = np.concatenate([single_row, candlestick_features[i]])

            output_data.append(single_row)

        return np.array(output_data)

    def get_before(self) -> np.ndarray:
        """
        Adapts the original get_before method to use the refactored static method for processing.
        """
        self.__prep_data()
        return self.process_entries(self.__all_data, self.__entry_times,
                                    self.__rows, self.__columns_indices, self.__candlestick_features)

    def add_pattern_features(self, dp_pattern):
        features_list = self.__results['metadata'].apply(
            lambda metadata: dp_pattern(**metadata)
        )

        # Since each item is already a numpy array, we use np.vstack to stack them vertically
        self.__candlestick_features = np.vstack(features_list.values)

    def get_target(self) -> np.ndarray:
        return self.__target


class CandleStickDataProcessing:
    @staticmethod
    def calculate_basic_features(current_open, current_close, current_high, current_low):
        body_length = abs(current_open - current_close)
        upper_shadow_length = current_high - max(current_open, current_close)
        lower_shadow_length = min(current_open, current_close) - current_low
        candlestick_length = current_high - current_low
        return {
            "body_length": body_length,
            "upper_shadow_length": upper_shadow_length,
            "lower_shadow_length": lower_shadow_length,
            "candlestick_length": candlestick_length
        }

    @staticmethod
    def calculate_engulfing_features(current_open, current_close, prev_open, prev_close):
        current_body = abs(current_close - current_open)
        previous_body = abs(prev_close - prev_open)
        engulfing_ratio = current_body / previous_body if previous_body else 0

        return engulfing_ratio

    @staticmethod
    def calculate_inverted_hammer_features(current_open, current_close, current_high, current_low):
        body_length = abs(current_open - current_close)
        upper_shadow_length = current_high - max(current_open, current_close)
        total_length = current_high - current_low
        upper_to_body_ratio = upper_shadow_length / (
                body_length + 0.001)  # Adding a small number to avoid division by zero
        body_to_total_ratio = body_length / (total_length + 0.001)

        return np.array([upper_to_body_ratio, body_to_total_ratio])
