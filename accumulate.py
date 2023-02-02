import os
import numpy as np
from math import ceil
import pandas as pd
import sys
from dataclasses import dataclass, field, asdict, make_dataclass
from memoize.dataframe import memoize_df

sys.path.append(os.path.realpath('src'))

from ubacktester import (
    BacktestEngine, px_plot, FeedBase, StrategyBase,
    ClockBase, downsample_to_pow
)
from profiler import profiler


@dataclass
class TradesFeed(FeedBase):
    USE_NS_DT = True

    # timestamp_utc_nanoseconds: int
    name: str
    PriceMillionths: int = 0
    SizeBillionths: int = 0
    Side: int = 0
    is_qual: int = 0
    cum_volm_side: int = 0
    cum_volm_all: int = 0
    cum_volm_qual: int = 0

@dataclass
class BookFeed(FeedBase):
    USE_NS_DT = True

    # timestamp_utc_nanoseconds: int
    name: str
    Ask1PriceMillionths: int = 0
    Bid1PriceMillionths: int = 0
    Ask2PriceMillionths: int = 0
    Bid2PriceMillionths: int = 0
    Bid1SizeBillionths: int = 0
    Ask1SizeBillionths: int = 0
    Bid2SizeBillionths: int = 0
    Ask2SizeBillionths: int = 0

def downsample_to_pow(val: int, pow10: int = 6) -> int:
    n = pow10 + 1
    hi, lo = str(val)[:-n], str(val)[-n:]
    roundup = lambda x: int(ceil(x / 10.0)) * 10
    suffix = str(roundup(int(lo[:2])))[0] + (pow10 * '0')
    final = int(hi + suffix)
    assert len(str(final)) == len(str(val))
    return final

@dataclass
class AccumulationStratBase(StrategyBase):
    USE_NS_DT = True
    downsample_rate: int = 6 # 1e6 ns, or 1 ms
    side: int = 1
    i = 0

    def last_n_trades(self, n: int = 1):
        df = self.feeds['trades'][-n:]
        return df

    def last_n_qual(self, n: int = 1):
        """Get the last `n` qualifying trades."""
        # If I want to achieve a 3% participation rate overall, what should I set my target rate to?

        df = self.last_n_trades(5)
        df['dt_ds'] = (
            pd.Series(df.index, dtype=np.int64)
            .apply(downsample_to_pow, args=[self.downsample_rate])
            .values
        )

        grp = df.groupby('dt_ds', group_keys=False).apply(self._mark_qualified)
        grp.index.name = 'dt'
        # breakpoint()
        return df

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
        # breakpoint()
        return df

    def step(self):
        self.i += 1
        if self.i % 1000 == 0:
            last = self.last_n_qual(5)['Side'].sum()

class InsufficientRowsError(Exception):
    pass

class AccumulateRunner(dict):
    # HW4_TRADES_CSV = 'tests/data/mini_trades_narrow_BTC-USD_2021.delim'
    # HW4_BOOK_CSV = 'data/Crypto/2021/For_Homework/book_narrow_BTC-USD_2021.delim'

    def __init__(
        self, side: int = 1,
        downsample_rate: int = 6, # 1e6 ns, or 1 ms
    ):
        assert side in (1, -1)
        self.side = side
        self.downsample_rate = downsample_rate

    @profiler()
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
    def get_trades_data(
        self, fp, downsample_rate, side, start_date_ns,
        row_limit,
    ):
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
        # assert not (df['Side'] == 0).any()
        df = df[df['Side'] / side > 0]
        df.set_index('dt', inplace=True)
        df = df.loc[start_date_ns:].iloc[:int(row_limit)]
        df = self.mark_qualified_trades(df)
        df = df.convert_dtypes()
        # breakpoint()
        return df

    def _get_book_data(self, fp, start_ns: int, end_ns: int):
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
    def run_accumulate_strat(
        self, fp,
        start_date='1970-01-01', # trim data starting at this date
        target_prt_rate=0.01, # 1% of traded volume
        target_notional=1e6, # stop trading when notional has reached this
        fee_rate=50, # basis points on notional
        row_limit=1e5, # number of trades to pull from data
    ):
        start_date_ns = pd.to_datetime(start_date, unit='ns').value
        df = self.get_trades_data(
            fp=fp, downsample_rate=self.downsample_rate, side=self.side,
            start_date_ns=start_date_ns, row_limit=row_limit,
        )
        df = df.convert_dtypes()
        if 'dt' in df.columns:
            df.set_index('dt', inplace=True)

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

        traded_notional = df['notional'].sum() / 1e9
        if traded_notional < target_notional:
            # Raise error if we haven't reached target_notional
            raise InsufficientRowsError(
                f"{traded_notional=} {target_notional=}"
            )
        else:
            # Trim dataframe once we've reached target_notional
            last_trade = df[df['notional'].cumsum() / 1e9 > target_notional].iloc[0]
            last_idx = int(last_trade.name)
            df = df.iloc[:last_idx+1]
        return df

    @profiler()
    def _test_accumulate_strat_ubacktester(self, fp):
        trades = self.get_trades_data(fp=fp, downsample_rate=self.downsample_rate)
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
    runner = AccumulateRunner(side=1, downsample_rate=6)
    runner.run_accumulate_strat(
        fp='data/Crypto/2021/For_Homework/trades_narrow_BTC-USD_2021.delim',
        start_date='1970-01-01', # trim data before this date
        target_prt_rate=0.01, # 1% of traded volume
        target_notional=1e6, # stop trading when notional has reached this
        fee_rate=50, # basis points on notional
        row_limit=1e5, # number of trades to pull from data
    )