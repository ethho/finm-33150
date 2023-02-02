import os
import numpy as np
from math import ceil
import pandas as pd
import sys
from memoize.dataframe import memoize_df

sys.path.append(os.path.realpath('src'))

from ubacktester import (
    PriceFeed, BacktestEngine, BasicStrategy, px_plot, BuyAndHold,
    NaiveQuantileStrat, ClockBase, AccumulationStratBase, TradesFeed, BookFeed,
    downsample_to_pow
)
from profiler import profiler

class AccumulateRunner(dict):
    HW4_TRADES_CSV = 'tests/data/mini_trades_narrow_BTC-USD_2021.delim'
    HW4_BOOK_CSV = 'data/Crypto/2021/For_Homework/book_narrow_BTC-USD_2021.delim'
    downsample_rate = 6 # 1e6 ns, or 1 ms

    def __init__(self, side: int = 1, start_date='1970-01-01'):
        assert side in (1, -1)
        self.side = side
        self.start_date = start_date
        self.start_date_ns = pd.to_datetime(start_date, unit='ns').value

    def mark_qualified_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        df['dt_ds'] = (
            pd.Series(df.index, dtype=np.int64)
            .apply(downsample_to_pow, args=[self.downsample_rate])
            .values
        )
        grp = df.groupby('dt_ds', group_keys=False).apply(self._mark_qualified)
        grp.index.name = 'dt'
        # breakpoint()
        return grp

    def _mark_qualified(self, df):
        if len(df) == 1:
            df['is_qual'] = 1
            return df
        if self.side > 0:
            qual_price = df['PriceMillionths'].max()
        else:
            qual_price = df['PriceMillionths'].min()
        qualified_mask = df['PriceMillionths'] == qual_price
        df['is_qual'] = qualified_mask.astype(int)
        return df

    @memoize_df(cache_dir='data/memoize', cache_lifetime_days=None)
    def get_trades_data(self, fp, downsample_rate):
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
        # df = df[df['Side'] / self.side > 0]
        df.set_index('dt', inplace=True)
        # df = df.loc[start_date_ns:]
        df = self.mark_qualified_trades(df)

        df = df.convert_dtypes()
        # breakpoint()

        return df

    def _get_book_data(self, start_ns: int, end_ns: int):
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
        target_prt_rate = 0.01
        fee_rate = 50 # basis points on the notional

        df = self.get_trades_data(fp=self.HW4_TRADES_CSV, downsample_rate=self.downsample_rate)
        if not 'dt' in df.columns:
            df.reset_index(inplace=True)

        # Define masks for same side and qualifying trades
        same_side = df['Side'] * self.side > 0
        qual_mask = same_side & df['is_qual']

        # Calculate cumulative volume over time for each side, for all trades,
        # and for qualifying trades.
        df.loc[same_side, 'cum_volm_side'] = df.loc[same_side, 'SizeBillionths'].cumsum()
        df.loc[~same_side, 'cum_volm_side'] = df.loc[~same_side, 'SizeBillionths'].cumsum()
        df['cum_volm_all'] = df.loc[:, 'SizeBillionths'].cumsum()
        df['cum_volm_qual'] = pd.NA
        df.loc[qual_mask, 'cum_volm_qual'] = df.loc[qual_mask, 'SizeBillionths'].cumsum()
        df = df.convert_dtypes()

        # Calculate target participation for each qualifying trade (billionths).
        # In theory, the below calculation should get us the same as
        # df['cum_volm_qual'] * target_prt_rate. They're not exactly equal
        # due to the rounding we do with astype(int)
        df['target_prt'] = (same_side.astype(int) * df['is_qual'] * target_prt_rate * df['SizeBillionths'])#.astype(int)
        df['target_prt_cumsum'] = df['target_prt'].cumsum()
        # Approximately equal:
        # df['target_prt_cumsum'] = (df['cum_volm_qual'] * target_prt_rate).astype(int)

        # Calculate notional (billionths), fees (billionths), and VWAP
        df['notional'] = (df['target_prt'] * (df['PriceMillionths'] / 1e6))#.astype(int)
        # df['notional_cumsum'] = df['notional'].cumsum().astype(int)
        df['vwap_cumsum'] = df['notional'].cumsum().astype(int).div(df['target_prt'].cumsum().astype(int))
        df['fees'] = (df['notional'] * fee_rate / 1e4).astype(int)
        df['market_vwap'] = (
            (df['SizeBillionths'] * (df['PriceMillionths'] / 1e6)).cumsum() /
            (df['SizeBillionths']).cumsum())

        df['since_arrival'] = df['dt_ds'] - df['dt_ds'].iloc[0]

        # DEBUG
        # df['market_vwap_side'] = (
        #     (df.loc[same_side, 'SizeBillionths'] * (df.loc[same_side, 'PriceMillionths'] / 1e6)).cumsum() /
        #     (df.loc[same_side, 'SizeBillionths']).cumsum())
        # assert not (df['market_vwap'] - df['vwap_cumsum'] > 2.).any()
        vwap = (
            (df['SizeBillionths'] * (df['PriceMillionths'] / 1e6)).sum() /
            (df['SizeBillionths']).sum())
        sample = df.iloc[200:210, :]

        breakpoint()
        return df

    @profiler()
    def test_accumulate_strat_ubacktester(self):
        trades = self.get_trades_data(fp=self.HW4_TRADES_CSV, downsample_rate=self.downsample_rate)
        trades_feed = TradesFeed.from_df(trades)
        # TODO: don't include timestamps for trades of opposite sign
        dti = trades['dt'].sort_values().drop_duplicates().values
        as_dt = pd.to_datetime(dti)
        clock = ClockBase(dti)

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