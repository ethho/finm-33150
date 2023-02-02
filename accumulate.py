import os
import pandas as pd
import sys

sys.path.append(os.path.realpath('src'))

from ubacktester import (
    PriceFeed, BacktestEngine, BasicStrategy, px_plot, BuyAndHold,
    NaiveQuantileStrat, ClockBase, AccumulationStratBase, TradesFeed, BookFeed,
)
from profiler import profiler

class AccumulateRunner:
    HW4_TRADES_CSV = 'tests/data/mini_trades_narrow_BTC-USD_2021.delim'
    HW4_BOOK_CSV = 'data/Crypto/2021/For_Homework/book_narrow_BTC-USD_2021.delim'

    def __init__(self, side: int = 1):
        assert side in (1, -1)
        self.side = side

    def _get_trades(self):
        fp = self.HW4_TRADES_CSV
        df = pd.read_csv(fp, delim_whitespace=True)
        df.rename(columns={
            'timestamp_utc_nanoseconds': 'dt',
        }, inplace=True)
        df.sort_values(by='dt', inplace=True)
        df['Side'] = df['Side'].astype(int)
        print(
            f"Dates in trades data {fp=} range between "
            f"{df['dt'].min()} ({pd.to_datetime(df['dt'].min())}) and "
            f"{df['dt'].max()} ({pd.to_datetime(df['dt'].max())})"
        )
        df.drop(columns=['received_utc_nanoseconds'], inplace=True)
        assert not (df['Side'] == 0).any()
        df = df[df['Side'] / self.side > 0]
        # breakpoint()
        return df

    def _get_book(self, start_ns: int, end_ns: int):
        fp = self.HW4_BOOK_CSV
        df = pd.read_csv(fp, delim_whitespace=True)
        df.rename(columns={
            'timestamp_utc_nanoseconds': 'dt',
        }, inplace=True)
        df.sort_values(by='dt', inplace=True)
        # breakpoint()
        print(
            f"Dates in book data {fp=} range between "
            f"{df['dt'].min()} ({pd.to_datetime(df['dt'].min())}) and "
            f"{df['dt'].max()} ({pd.to_datetime(df['dt'].max())})"
        )
        # df = df[start_ns <= df['dt'] <= end_ns]
        # assert not df.empty, f"empty df between {start_ns=} and {end_ns=}"
        return df

    @profiler()
    def test_accumulate_strat(self):
        trades = self._get_trades()
        trades_feed = TradesFeed.from_df(trades)
        # TODO: don't include timestamps for trades of opposite sign
        dti = trades['dt'].sort_values().drop_duplicates().values
        as_dt = pd.to_datetime(dti)
        clock = ClockBase(dti)

        # book = self._get_book(
        #     start_ns=1618090132515484000,
        #     end_ns=1618137171463921000
        # )
        # book_feed = BookFeed.from_df(book)

        be = BacktestEngine(clock=clock)
        be.add_feed(trades_feed, name='trades')
        strat1 = AccumulationStratBase(cash_equity=1e4, side=self.side)
        be.add_strategy(strat1)
        be.run()

        # strat1.plot(
        #     show=False,
        #     # include_cols=['daily_pct_returns'],
        #     # scale_cols={'nshort': 40, 'nlong': 40}
        #     include_cols=['returns', 'nshort', 'nlong', ],
        #     scale_cols={'nshort': 40, 'nlong': 40, }
        # )

if __name__ == '__main__':
    runner = AccumulateRunner()
    runner.test_accumulate_strat()