import os
import numpy as np
from math import ceil
import multiprocessing
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


def downsample_to_pow(val: int, pow10: int = 6) -> int:
    n = pow10 + 1
    hi, lo = str(val)[:-n], str(val)[-n:]
    roundup = lambda x: int(ceil(x / 10.0)) * 10
    suffix = str(roundup(int(lo[:2])))[0] + (pow10 * '0')
    final = int(hi + suffix)
    assert len(str(final)) == len(str(val))
    return final

class InsufficientRowsError(Exception):
    pass

class AccumulateRunner(dict):

    def __init__(
        self, side: int = 1,
        downsample_rate: int = 6, # 1e6 ns, or 1 ms
    ):
        assert side in (1, -1)
        self.side = side
        self.downsample_rate = downsample_rate

    # @profiler()
    def mark_qualified_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        df['dt_ds'] = (
            pd.Series(df.index, dtype=np.int64)
            .apply(downsample_to_pow, args=[self.downsample_rate])
            .values
        )
        df['Side'] = df['Side'].replace(0, -1 * self.side)
        try:
            df['side_sign'] = (df['Side'] / df['Side'].abs()).astype(int)
        except pd.errors.IntCastingNaNError:
            raise
        grp = df.groupby(['dt_ds', 'side_sign'], group_keys=False).apply(self._mark_qualified)
        grp.index.name = 'dt'
        grp.drop(columns='side_sign', inplace=True)
        # breakpoint()
        return grp

    def _mark_qualified(self, df):
        if len(df) == 1:
            df['is_qual'] = 1
            return df
        side = df['side_sign'].iloc[0]
        # assert (df['side_sign'] == side).all()
        if side > 0:
            qual_price = df['PriceMillionths'].max()
        else:
            qual_price = df['PriceMillionths'].min()
        qualified_mask = df['PriceMillionths'] == qual_price
        df['is_qual'] = qualified_mask.astype(int)
        # breakpoint()
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
        # df = df[df['Side'] / side > 0]
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

    # @profiler()
    # @memoize_df(cache_dir='data/memoize', cache_lifetime_days=None)
    def run_accumulate_strat(
        self, fp,
        start_date='1970-01-01', # trim data starting at this date
        target_prt_rate=0.01, # 1% of traded volume
        target_notional=1e6, # stop trading when notional has reached this
        fee_rate=50, # basis points on notional
        row_limit=4e5, # number of trades to pull from data
        side=1,
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
        df['target_prt'] = (same_side.astype(int) * df['is_qual'] * target_prt_rate * df['SizeBillionths'])
        df['target_prt_cumsum'] = df['target_prt'].cumsum()
        # Approximately equal:
        # df['target_prt_cumsum'] = (df['cum_volm_qual'] * target_prt_rate).astype(int)

        # Calculate notional (billionths), fees (billionths), and VWAP
        df['notional'] = (df['target_prt'] * (df['PriceMillionths'] / 1e6))
        # df['notional_cumsum'] = df['notional'].cumsum().astype(int)
        try:
            df['vwap'] = df['notional'].cumsum().div(df['target_prt'].cumsum().replace(0., pd.NA))
        except ZeroDivisionError:
            raise
        df['fees'] = (df['notional'] * fee_rate / 1e4).astype(int)
        # df['market_vwap'] = (
        #     (df['SizeBillionths'] * (df['PriceMillionths'] / 1e6)).cumsum() /
        #     (df['SizeBillionths']).cumsum())

        df['since_arrival'] = df['dt_ds'] - df['dt_ds'].iloc[0]

        # DEBUG
        # df['market_vwap_side'] = (
        #     (df.loc[same_side, 'SizeBillionths'] * (df.loc[same_side, 'PriceMillionths'] / 1e6)).cumsum() /
        #     (df.loc[same_side, 'SizeBillionths']).cumsum())
        vwap = (
            (df['SizeBillionths'] * (df['PriceMillionths'] / 1e6)).sum() /
            (df['SizeBillionths']).sum())
        # print(f'{vwap=}')
        # assert not (df['market_vwap'] > df['vwap']).sum()
        if vwap < df['vwap'].iloc[-1]:
            print(f"Warning: {(df['market_vwap'] > df['vwap']).sum()}")

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
            df = df.loc[:last_idx]
        return df

def accumulate_runner_wrapper(
    start_date,
    target_prt_rate=0.01,
    target_notional=1e6,
    fee_rate=50,
    row_limit=4e5,
    side=1,
    downsample_rate=6,
):
    kwargs = dict(
        fp='data/Crypto/2022/trades_narrow_BTC-USD_2022.delim',
        # start_date='1970-01-01', # trim data before this date
        start_date=start_date, # trim data before this date
        target_prt_rate=target_prt_rate, # 1% of traded volume
        target_notional=target_notional, # stop trading when notional has reached this
        fee_rate=fee_rate, # basis points on notional
        row_limit=row_limit, # number of trades to pull from data
        side=side,
    )
    runner = AccumulateRunner(side=side, downsample_rate=downsample_rate)
    try:
        df = runner.run_accumulate_strat(**kwargs)
    except (AssertionError, InsufficientRowsError) as err:
        print(f"Error while processing call with {kwargs=}: {str(err)}")
        return
    return df

if __name__ == '__main__':
    start_dates = [
        # '2021-04-11',
        # '2021-04-13',
        # '2021-04-15',
        # '2021-04-17',
        # '2021-04-19',
        # '2021-04-21',
        # '2022-01-30',
    ]
    # sides = [1, -1]
    sides = [-1]
    start_dates = [
        dt.isoformat() for dt in
        pd.date_range('2022-01-30', '2022-02-01', periods=10)
    ]
    # breakpoint()

    # Parallel version
    from functools import partial
    for side in sides:
        kw = dict(
            side=side, target_notional=1e7, target_prt_rate=0.1
        )
        with multiprocessing.Pool(2) as p:
            results = p.map(partial(globals()[f'accumulate_runner_wrapper'], **kw), start_dates)
    
    # Serial version
    # for side in sides:
    #     for start_date in start_dates:
    #         df = accumulate_runner_wrapper(
    #             start_date, side=side, target_notional=1e7, target_prt_rate=0.1
    #         )
        