from ml_backtest.interfaces import Strategy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator
from typing import Optional, List


class Backtest:
    def __init__(self, data: pd.DataFrame, strategy: Strategy,
                 initial_cash=100000, model: Optional[BaseEstimator] = None,
                 columns: Optional[List[str]] = None, rows: Optional[int] = None,
                 cs_pattern: bool = False):
        self.__data = data
        self.__strategy = strategy
        self.__cash = initial_cash
        self.__initial_cash = initial_cash
        self.__equity_peak = initial_cash
        self.__max_drawdown = 0
        self.__open_positions = []
        self.__completed_trades = []
        self.__trade_counter = 0
        self.__long_wins = 0
        self.__short_wins = 0
        self.__long_loses = 0
        self.__short_loses = 0
        self.__long_evens = 0
        self.__short_evens = 0
        self.__gross_profit = 0
        self.__gross_loss = 0
        self.__start_time = None
        self.__end_time = None
        self.__model = model
        self.__columns = columns
        self.__rows = rows
        self.__cs_pattern = cs_pattern
        self.__backtest = None

        self.__run()
        self.__results()

    def __run(self):
        dates = self.__data['date'].values
        lows = self.__data['low'].values
        highs = self.__data['high'].values
        close = self.__data['close'].values
        open = self.__data['open'].values

        self.__start_time = dates[0]
        self.__end_time = dates[-1]

        self.__strategy.init()

        # pass the model into the strategy interface if a model is init with backtest
        if self.__columns is not None and self.__rows is not None and self.__model is not None:
            self.__strategy.set_ml(model=self.__model, columns=self.__columns, rows=self.__rows,
                                   df=self.__data, cs_pattern=self.__cs_pattern)

        for index in tqdm(range(len(dates)), total=len(dates)):
            # Assuming strategy.on_data can be adapted or is simplified for demonstration
            self.__strategy.on_data(index, lows[:index + 1], highs[:index + 1],
                                    close[:index + 1], open[:index + 1], dates[:index + 1])

            for position in list(self.__strategy.positions):  # Iterate over a shallow copy if positions are modified
                exit_price = None

                if 'highest high' not in position or highs[index] > position['highest high']:
                    position['highest high'] = highs[index]

                    # Calculate the difference between 'Open' at entry and the new 'highest high'
                    # Assumes 'open at entry' is stored in the position when the position is created
                    position['target'] = position['highest high'] - position['entry price']

                if position['type'] == 'long':
                    if lows[index] <= position['stop loss']:
                        exit_price = position['stop loss']
                    elif highs[index] >= position['take profit']:
                        exit_price = position['take profit']
                elif position['type'] == 'short':
                    if highs[index] >= position['stop loss']:
                        exit_price = position['stop loss']
                    elif lows[index] <= position['take profit']:
                        exit_price = position['take profit']

                if exit_price is not None:
                    position['exit price'] = exit_price
                    position['exit time'] = dates[index]
                    self.__close_position(position, exit_price)
                    self.__trade_counter += 1

    def __close_position(self, position, current_price):
        # Calculate profit or loss
        # print(position, " ", current_price)
        if position['type'] == 'long':
            profit_loss = current_price - position['entry price']
            if profit_loss > 0:
                self.__long_wins += 1
                self.__gross_profit += profit_loss
            elif profit_loss < 0:
                self.__long_loses += 1
                self.__gross_loss += profit_loss
            else:
                self.__long_evens += 1
        else:  # short position
            profit_loss = position['entry price'] - current_price
            if profit_loss > 0:
                self.__short_wins += 1
                self.__gross_profit += profit_loss
            elif profit_loss < 0:
                self.__short_loses += 1
                self.__gross_loss += profit_loss
            else:
                self.__short_evens += 1

        # Update cash balance
        self.__cash += profit_loss  # This simplification assumes 1 share per trade

        if self.__cash > self.__equity_peak:
            self.__equity_peak = self.__cash
        else:
            # Calculate drawdown from peak
            drawdown = self.__equity_peak - self.__cash
            # Update max drawdown if this drawdown is larger
            self.__max_drawdown = max(self.__max_drawdown, drawdown)

        # Move position from open to completed trades
        self.__strategy.positions.remove(position)

        # Add any statistics to the completed trades
        position['exit price'] = current_price
        position['size'] = 1
        position['pnl'] = profit_loss
        self.__completed_trades.append(position)
        self.__strategy.in_position = False

    def __results(self) -> pd.DataFrame:

        wins = self.__short_wins + self.__long_wins
        loses = self.__short_loses + self.__long_loses
        self.__backtest_result = [{'start time': self.__start_time,
                                   'end time': self.__end_time,
                                   '# of trades': self.__trade_counter,
                                   '# of wins': wins,
                                   '# of loses': loses,
                                   'win rate': f'{np.around((wins / (wins + loses)) * 100, decimals=2)}%',
                                   '# of long wins': self.__long_wins,
                                   '# of long loses': self.__long_loses,
                                   '# of long evens': self.__long_evens,
                                   '# of short wins': self.__short_wins,
                                   '# of short loses': self.__short_loses,
                                   '# of short evens': self.__short_evens,
                                   'net profit': np.around(self.__gross_profit + self.__gross_loss, decimals=2),
                                   'max drawdown': f'-{np.around(self.__max_drawdown, decimals=2)}',
                                   'gross profit': np.around(self.__gross_profit, decimals=2),
                                   'gross loss': np.around(self.__gross_loss, decimals=2),
                                   'profit factor': np.around(self.__gross_profit / abs(self.__gross_loss),
                                                              decimals=2)}]

        self.__backtest_df = pd.DataFrame(self.__backtest_result)
        self.__backtest_df = self.__backtest_df.T

    def get_trades(self) -> pd.DataFrame:
        completed_trades_df = pd.DataFrame(self.__completed_trades)
        return completed_trades_df

    def get_results(self) -> pd.DataFrame:
        return self.__backtest_df
