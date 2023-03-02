#!/usr/bin/env python
"""
Author: Ethan Ho
Date: 3/2/2023
License: MIT
Usage:

python3 final_proj.py
# or from Python3: from final_proj import main as final_proj_main
"""

import os
import re
import sys
import json
import math
from typing import List, Dict, Tuple, Optional
import functools
import pandas as pd
import numpy as np
from scipy.stats import norm, probplot
import quandl
import plotly.express as px
try:
    from memoize.dataframe import memoize_df
except ImportError as err:
    print(str(err))
    raise Exception(
        f"Missing dependency memoize. Please install using pip:\n\n"
        f"pip install git+https://github.com/ethho/memoize.git"
    )
try:
    from lmfit.models import SkewedGaussianModel
except ImportError as err:
    print(str(err))
    raise Exception(
        f"Missing dependency memoize. Please install using pip:\n\n"
        f"pip install lmfit"
    )

TENOR_WEEK_MAP = {
    (1, 'm'): 4,
    (2, 'm'): 8,
    (3, 'm'): 13,
    (4, 'm'): 17,
    (6, 'm'): 26,
    (12, 'm'): 52,
    (1, 'y'): 52,
    (2, 'y'): 52 * 2, # 104
    (3, 'y'): 52 * 3, # 156
    (5, 'y'): 52 * 5, # 260
    (7, 'y'): 52 * 7, # 364
    (10, 'y'): 52 * 10, # 520
    (20, 'y'): 52 * 20, # 1040
    (30, 'y'): 52 * 30, # 1560
}

def get_secrets(fp='./secrets.json'):
    """
    Reads secret values such as API keys from a JSON-formatted file at `fp`.
    """
    with open(fp, 'r') as f:
        data = json.load(f)
    return data

def get_quandl_api_key() -> str:
    """
    Returns Quandl API key stored in secrets.json.
    """
    secrets = get_secrets()
    key = secrets.get('NASTAQ_DATA_API_KEY')
    assert key, f"NASTAQ_DATA_API_KEY field in secrets.json is empty or does not exist"
    return key

def strip_str_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame, strips values in columns with string or object
    dtype. I noticed that this was an issue when I saw some m_ticker values
    like "AAPL       " with trailing whitespace.
    """
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].str.strip()
    return df

@memoize_df(cache_dir='data/memoize', cache_lifetime_days=None)
def fetch_quandl_table(
    name, start_date, end_date, **kw
) -> pd.DataFrame:
    df = quandl.get_table(
        name,
        date={'gte': start_date, 'lte': end_date},
        api_key=get_quandl_api_key(),
        paginate=True,
        **kw
    )
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df.reset_index(inplace=True)
    return df

@memoize_df(cache_dir='data/memoize', cache_lifetime_days=None)
def fetch_quandl_quotemedia_prices(
    start_date, end_date, ticker
) -> pd.DataFrame:
    return fetch_quandl_table(
        name= 'QUOTEMEDIA/PRICES',
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
    )

@memoize_df(cache_dir='data/memoize', cache_lifetime_days=None)
def fetch_quandl_yc(
    name, start_date, end_date,
) -> pd.DataFrame:
    df = quandl.get(
        name,
        start_date=start_date,
        end_date=end_date,
        api_key=get_quandl_api_key(),
    ).reset_index().rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    return df

@memoize_df(cache_dir='data/memoize', cache_lifetime_days=None)
def fetch_quandl_spot(
    symbol, **kw
) -> pd.DataFrame:
    df = quandl.get(
        f'CUR/{symbol}',
        **kw
    ).reset_index().rename(columns={
        'DATE': 'date',
        'RATE': f'USD/{symbol}',
    })
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    return df

def unique_index_keys(df, level=0) -> List[str]:
    return df.index.get_level_values(level=level).unique().tolist()

def get_next_day_of_week(date, day_of_week: int) -> str:
    """
    Monday = 0, Wednesday = 2
    """
    as_dt = pd.to_datetime(date)
    days_until = (day_of_week - as_dt.day_of_week) % 7
    out_dt = as_dt + pd.to_timedelta(days_until, 'D')
    return out_dt.strftime('%Y-%m-%d')

def get_standard_yc_cols(cols: List, col_prefix='') -> Dict:
    out = dict()
    for col_raw in cols:
        col = col_raw.lower()
        col = re.sub(r'-year', 'y', col)
        col = re.sub(r'-month', 'm', col)
        if col_prefix:
            col = col_prefix + '_' + col
        out[col_raw] = col
    return out

def get_yc(*args, col_prefix='', **kw):
    df = fetch_quandl_yc(*args, **kw)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns=get_standard_yc_cols(df.columns, col_prefix), inplace=True)
    return df

@functools.lru_cache()
def get_col_groups(cols) -> Dict:
    """
    Usage: get_col_groups(tuple(yc_daily.columns.tolist()))
    """
    out = dict()
    for col in cols:
        prefix, tenor_raw = col.split('_')
        tenor, unit = tenor_raw[:-1], tenor_raw[-1]
        if prefix not in out:
            out[prefix] = list()
        item = {
            'col': col,
            'country': prefix,
            'tenor': tenor,
            'unit': unit
        }
        out[prefix].append(item)
    return out

def bond_price(zcb, coupon_rate, tenor, coupon_freq):
    """
    Copied from Zero_And_Spot_Curves.ipynb
    """
    times = np.arange(tenor, 0, step=-coupon_freq)[::-1]
    if times.shape[0] == 0:
        p = 1.0
    else:
        r = np.interp(times, zcb.index.values, zcb.values) # Linear interpolation
        p = np.exp(-tenor*r[-1]) + coupon_freq * coupon_rate * np.exp(-r*times).sum()
    return p

@functools.lru_cache()
def tenor_wk_to_years(wk: int) -> float:
    """
    Convert tenor from weeks to years.
    """
    return wk / 52
    # Equivalently,
    # return wk * 7 / 364

@functools.lru_cache()
def tenor_years_to_wk(yr: float) -> float:
    """
    Convert tenor from years to weeks.
    """
    return yr * 52
    # Equivalently,
    # return yr * 364 / 7


def _zcb_from_spot(spot, tenor, cpn_freq, spot_curve):
    """
    Calculate ZCB (discount) rate for `tenor` (years) from `spot` rate,
    given 1 / `cpn_freq` coupons are given per year.

    Adapted from compute_zcb_curve in Zero_And_Spot_Curves.ipynb.
    """
    if tenor <= 1.:
        # US T-bills (<=1 year maturity) have no coupons
        return spot
    times = np.arange(tenor-cpn_freq, 0, step=-cpn_freq)[::-1]
    coupon_half_yr = cpn_freq * spot
    z = np.interp(times, spot_curve.index.values, spot_curve.values)
    preceding_coupons_val = (coupon_half_yr*np.exp(-z*times)).sum()
    return -np.log((1-preceding_coupons_val)/(1+coupon_half_yr))/tenor


def zcb_from_spot(row, cpn_freq, spot_curve, **kw):
    """
    Wrapper around _zcb_from_spot. Used in pd.DataFrame.apply.
    """
    tenor = row.name # in years
    spot = row.spot
    return _zcb_from_spot(spot, tenor, cpn_freq, spot_curve)


def pr_from_spot(row, cpn_freq, spot_curve, holding_period, **kw):
    """
    Calculate ZCB (discount) price from `row.spot` rate, given 1 / `cpn_freq`
    coupons are given per year.

    Lightweight wrapper around `bond_price` function from
    Zero_And_Spot_Curves.ipynb
    """
    tenor = row.name # in years
    spot = row.spot

    # For example T = 5 years and S = 5 years - 1 month
    # Note that we can pass `holding_period` = 0 to get pr_t
    T = tenor
    S = T - holding_period
    return bond_price(
        spot_curve,
        coupon_rate=spot,
        tenor=S,
        coupon_freq=cpn_freq
    )


def get_zcb_curve_at_t(
    spot_wk: pd.Series,
    coupons_per_year=2,
    holding_period=28/364.,
    # Equivalently,
    # holding_period=4/52.,
):
    """
    Given the spot rate `spot_wk` indexed by tenor (in weeks), calculate
    zero coupon rate, zero-coupon factor (price), and forward rate & factor.
    """
    cpn_freq = 1 / float(coupons_per_year)

    # Arrange spot rates into a DataFrame for easier analysis
    df = pd.DataFrame(
        data={
            'spot': spot_wk.values,
        },
        # Convert tenor index from weeks into years for below calculations
        index=[tenor_wk_to_years(tenor_wk) for tenor_wk in spot_wk.index],
    )

    # Shared kwargs to pass to functions in apply
    kw = dict(
        cpn_freq=cpn_freq,
        spot_curve=df['spot'].copy(),
    )

    # Zero-coupon rate
    df['zcb'] = df.apply(zcb_from_spot, axis=1, **kw)
    # Zero-coupon bond price at maturity S = tenor - 1 month
    df['pr_s'] = df.apply(pr_from_spot, axis=1, holding_period=holding_period, **kw)
    # Zero-coupon bond price at maturity T = tenor - 0 months
    df['pr_t'] = df.apply(pr_from_spot, axis=1, holding_period=0., **kw)
    # Forward discount factor (F)
    df['fwd_factor'] = df.pr_t / df.pr_s
    # Forward discount rate (f)
    df['fwd'] = -np.log(df['fwd_factor']) / holding_period

    # Convert index back from years to weeks
    df.index = pd.Index([tenor_years_to_wk(tenor_yr) for tenor_yr in df.index])
    # Convert to a series with a MultiIndex
    ser = df.stack(dropna=False)
    ser.index.set_names(['tenor_wk', 'metric'], inplace=True)
    return ser

def get_4wk_value(df: pd.DataFrame, holding_period, **kw):
    # Calculate tenor in years.
    # There should only be one tenor in the input DataFrame `df`.
    tenors_wk = df.index.get_level_values(level=0).unique()
    assert len(tenors_wk), tenors_wk
    tenor_wk = tenors_wk[0]
    tenor = tenor_wk_to_years(tenor_wk)

    pr_s = df.loc[tenor_wk, 'pr_s']
    # NOTE: this assumes that the freq == holding_period
    pr_t_old= df.loc[tenor_wk, 'pr_t'].shift(1)#, freq='28D')
    val = pr_s / pr_t_old
    return val

def calculate_from_spot(
    df_raw,
    holding_period=28/364.,
    # Equivalently,
    # holding_period=4/52.,
) -> pd.DataFrame:
    """
    Calculate zero-coupon metrics (rate, factor, forward rate, etc.)
    given `df_raw`, which contains annualized spot rates in percent.

    TODO
    Review holding_period. Namely, 364/28 = 13.0 trades/year. But our actual
    holding_period = 4 weeks = 28 days.
    """
    # Get groups by column prefix
    grps: Dict[List[Dict]] = get_col_groups(tuple(df_raw.columns.tolist()))
    zcb_dict = dict()
    for country, cols in grps.items():
        # Convert each tenor into units of weeks
        for item in cols:
            item['tenor_wk'] = TENOR_WEEK_MAP[(int(item['tenor']), item['unit'])]
        df = df_raw.rename(columns={
            item['col']: item['tenor_wk']
            for item in cols
        })

        # Since U.S Treasury uses a 365-366 day/year definition for
        # calculating "annualized rate",
        # (see https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions)
        # convert rates into our definition, which is a 364-day
        # year (7 days * 52.0 weeks = 364 days).
        df *= 364 / 365.2

        # Compute ZCB rates, factors, and forward rate & factor at each time t
        zcb = (df / 100.).apply(get_zcb_curve_at_t, axis=1)

        # Double-check that TimeDelta between DatetimeIndex values equals
        # holding_period = 28 days
        idx_timedelta = pd.Series(zcb.index.values - np.roll(zcb.index.values, shift=1)).iloc[1:]
        assert pd.to_timedelta(idx_timedelta.unique()[0]).days == 28, idx_timedelta.unique()

        # Calculate value at each tenor, assuming that we
        # bought the bond and held for `holding_period` before selling.
        val_df = (
            zcb
            .stack(0, dropna=False)
            [['pr_s', 'pr_t']]
            .swaplevel()
            .sort_index()
            .groupby(level=0, group_keys=False)
            .apply(get_4wk_value, axis=1, holding_period=holding_period)
            .T
        )
        val_df.columns = pd.MultiIndex.from_tuples([
            (tenor_wk, 'val') for tenor_wk in val_df.columns
        ], names=['tenor_wk', 'metric'])

        # Merge 1-month value `val` into the DataFrame
        zcb = zcb.merge(val_df, how='left', left_index=True, right_index=True)

        # Final tweaks to indices
        zcb.columns.set_names(['tenor_wk', 'metric'], inplace=True)
        zcb.index.set_names(['date'], inplace=True)
        zcb.sort_index(axis=0, inplace=True)
        zcb.sort_index(axis=1, inplace=True)
        assert (zcb.dtypes == np.float64).all(), f"some columns have dtype != np.float64"

        # Replace non-sensical values with NaN
        # For example, the 1-month forward rate of 1-month T-bills
        zcb.loc[:, (4.0, 'fwd')] = float('nan')

        zcb_dict[country] = zcb
    return zcb_dict

def unstack_zcb_df(in_df):
    df = in_df.copy()
    idx_df = pd.DataFrame(df.columns.str.split('_').tolist(), columns=['tenor', 'figure'])
    idx_df.tenor = idx_df.tenor.astype(float)
    idx_df.replace(0.08, 7/364., inplace=True)
    idx = pd.MultiIndex.from_frame(idx_df)
    df.columns = idx
    df = df.unstack().reorder_levels([1, 2, 0])
    return df

def get_position_returns(df):
    """
    Calculate the return series for every possible hedged position every month.
    Project Part 1B
    """
    # breakpoint()
    pass

def read_uszcb(
    zcb_out_fp='./data/uszcb.csv',
):
    """
    Example of how to load DataFrame from uszcb.csv.
    """
    df = pd.read_csv(
        zcb_out_fp,
        header=[0, 1],
        index_col=0,
        parse_dates=[0],
        dtype=float
    )
    df.columns = pd.MultiIndex.from_tuples([
        (int(float(tenor)), str(metric)) for tenor, metric in df.columns]
    )
    return df

def main(
    zcb_out_fp='./data/uszcb.csv',
):
    start_date = '1990-01-01'
    end_date = '2022-12-16'

    # Construct a DatetimeIndex containing dates to trade on.
    # We choose to trade every 4 weeks, since this is the tenor of
    # the 1-month T-bill that we will use for funding.
    # We choose every 4th Wednesday to trade on.
    daily_idx = pd.date_range(start_date, end_date)
    first_wed = get_next_day_of_week(start_date, 2)
    wed_idx_w_holidays = pd.date_range(first_wed, end_date, freq='28D')
    assert all(date.day_of_week == 2 for date in wed_idx_w_holidays)

    wed_idx = pd.to_datetime([
        date for date in wed_idx_w_holidays
        # if date not in pd.to_datetime([
        #     # Remove Wednesdays that fall on holidays
        #     '2012-12-26', '2013-12-25', '2014-01-01', '2018-12-26',
        #     '2019-12-25', '2020-01-01',
        # ])
    ])
    # assert len(wed_idx_w_holidays) > len(wed_idx)

    # Fetch Quandl yield curve (YC) data for each country
    countries = {
        'USA': 'USD',
    }
    yc_dict = {
        country: (
            get_yc(f'YC/{country}', start_date=start_date, end_date=end_date, col_prefix=country.lower())
            .reindex(daily_idx)
            .fillna(method='ffill')
            .iloc[1:, :]
        ) for country in countries.keys()
    }
    yc_daily = pd.concat(yc_dict.values(), axis=1)
    yc_monthly = yc_daily.loc[wed_idx].copy()
    zcb_all_countries = calculate_from_spot(yc_monthly)
    df = zcb_all_countries['usa']
    df.to_csv(zcb_out_fp)
    print(f'Wrote US ZCB rates to {zcb_out_fp}')

    # Part 1B: calculate position (PS) returns
    ps_df = get_position_returns(df)

    return df


if __name__ == '__main__':
    main(*sys.argv[1:])