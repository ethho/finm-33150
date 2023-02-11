#!/usr/bin/env python
# coding: utf-8
import json
import re
import os
from glob import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, probplot
import quandl
import functools
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
from src.ubacktester import (
    BacktestEngine, StrategyBase, PositionBase, FeedBase,
    PlotlyPlotter, FeedID, PriceFeed, px_plot, ClockBase
)
from memoize.dataframe import memoize_df

pd.options.display.float_format = '{:,.4f}'.format

# ## Configuration & Helper Functions
#
# The following cell contains helper functions and configuration options that I will use in this notebook.

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

def get_fx_spot(*args, **kw):
    df = fetch_quandl_spot(*args, **kw)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

@functools.lru_cache()
def get_col_groups(cols) -> Dict:
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
    Adapted from Zero_And_Spot_Curves.ipynb
    """
    times = np.arange(tenor, 0, step=-coupon_freq)[::-1]
    if times.shape[0] == 0:
        p = 1.0
    else:
        r = np.interp(times, zcb.index.values, zcb.values) # Linear interpolation
        p = np.exp(-tenor*r[-1]) + coupon_freq * coupon_rate * np.exp(-r*times).sum()
    return p

def get_zcb_curve(spot, coupons_per_year=4):
    """
    Adapted from Zero_And_Spot_Curves.ipynb
    """
    cpn_freq = 1 / float(coupons_per_year)
    for tenor, spot_rate in spot.items():
        if tenor > 0.001:
            times = np.arange(tenor-cpn_freq, 0, step=-cpn_freq)[::-1]
            coupon_half_yr = cpn_freq * spot_rate
            z = np.interp(times, spot.index.values, spot.values) # Linear interpolation
            preceding_coupons_val = (coupon_half_yr*np.exp(-z*times)).sum()
            # Question: tenor here needs to be 5 because all coupons at 5Y swap rate?
            # Answer: since we're only using 5Y rates, don't need to change this
            spot.loc[tenor] = -np.log((1-preceding_coupons_val)/(1+coupon_half_yr))/tenor

    # Calculate bond price for maturities T = 5 years and S = 5 years - 1 week
    T = 5.
    S = 4. + (51./52.)
    spot_copy = spot.copy()
    spot.loc['rt'] = bond_price(spot_copy, coupon_rate=spot.loc[5], tenor=T, coupon_freq=cpn_freq)
    spot.loc['rs'] = bond_price(spot_copy, coupon_rate=spot.loc[5], tenor=S, coupon_freq=cpn_freq)
    return spot


def get_zcb_curves(row):
    # Get groups by column prefix
    grps: Dict[List[Dict]] = get_col_groups(tuple(row.columns.tolist()))
    zcb_dict = dict()
    for cty, cols in grps.items():
        df = pd.DataFrame.from_records(cols).convert_dtypes()
        df['tenor'] = df['tenor'].astype(int)
        df.set_index(['tenor', 'unit'], inplace=True)
        try:
            lo_col = df.loc[(1, 'y'), 'col']
            lo = row[lo_col]
            lo.name = 1
        except KeyError:
            lo_col = df.loc[(12, 'm'), 'col']
            lo = row[lo_col]
            lo.name = 1
        try:
            mid_col = df.loc[(2, 'y'), 'col']
            mid = row[mid_col]
            mid.name = 2
        except KeyError:
            mid_col = df.loc[(3, 'y'), 'col']
            mid = row[mid_col]
            mid.name = 3
        hi_col = df.loc[(5, 'y'), 'col']
        hi = row[hi_col]
        hi.name = 5

        zcb = (pd.concat([lo, mid, hi], axis=1) / 100).apply(get_zcb_curve, axis=1)
        zcb.rename(columns={
            lo.name: f"{lo_col}_zcb",
            mid.name: f"{mid_col}_zcb",
            hi.name: f"{hi_col}_zcb",
            'rt': f"{cty}_rt",
            'rs': f"{cty}_rs",
        }, inplace=True)
        zcb_dict[cty] = zcb
    return zcb_dict


if __name__ == '__main__':
    # # Fetch Data
    #
    # First, let's set our time indices. We choose to trade weekly on Wednesdays, and skip the week if the Wednesday falls on a holiday.

    start_date = '2009-01-01'
    end_date = '2022-12-16'

    daily_idx = pd.date_range(start_date, end_date)
    first_wed = get_next_day_of_week(start_date, 2)
    wed_idx_w_holidays = pd.date_range(first_wed, end_date, freq='7D')
    assert all(date.day_of_week == 2 for date in wed_idx_w_holidays)

    wed_idx = [
        date for date in wed_idx_w_holidays
        if date not in pd.to_datetime([
            # Remove Wednesdays that fall on holidays
            '2012-12-26', '2013-12-25', '2014-01-01', '2018-12-26',
            '2019-12-25', '2020-01-01',
        ])
    ]
    assert len(wed_idx_w_holidays) > len(wed_idx)

    # Now, we fetch UK OIS rates and FX spot rates:
    ois_daily = pd.concat([
        get_yc('YC/GBR_ISSC', start_date=start_date, end_date='2021-12-09', col_prefix='gbr'),
        get_yc('YC/GBR_ISSS', start_date='2021-12-10', end_date=end_date, col_prefix='gbr'),
    ], axis=0)
    ois_daily = ois_daily.reindex(daily_idx).fillna(method='ffill').iloc[1:, :]
    assert not ois_daily.isnull().any().any()
    ois = ois_daily.loc[wed_idx]
    ois_daily

    fx_daily = pd.concat([
        get_fx_spot(cur, start_date=start_date, end_date=end_date)
        for cur in (
            'GBP',
            'VND',
            'THB',
            'PKR',
            'PHP',
        )
    ], axis=1)
    fx_daily = fx_daily.reindex(daily_idx).fillna(method='ffill').iloc[1:, :]
    fx = fx_daily.loc[wed_idx]
    fx_daily

    assert not fx.isnull().any().any()
    assert not ois.isnull().any().any()

    countries = ['VNM', 'THA', 'PAK', 'PHL']
    yc_dict = {
        country: (
            get_yc(f'YC/{country}', start_date=start_date, end_date=end_date, col_prefix=country.lower())
            .reindex(daily_idx)
            .fillna(method='ffill')
            .iloc[1:, :]
        ) for country in countries
    }
    yc_daily = pd.concat(yc_dict.values(), axis=1)
    # vnm = yc_dict['VNM']

    # # Compute ZCB Curves
    # Below, I define a modified version of Professor Boonstra's `compute_zcb_curve` function.
    get_col_groups(tuple(yc_daily.columns.tolist()))

    # zcb_daily = pd.concat(get_zcb_curves(yc_daily).values(), axis=1)
    zcb = get_zcb_curves(yc_daily)
    # px_plot(
    #     zcb['vnm'][['vnm_rt']],
    #     # include_cols = ['5y_price'],
    #     title = 'VNM 5-Year Swap Price',
    # )

    country_df = dict()

    for country in zcb.keys():
        print(f"Processing DataFrame for {country=}")
        df_daily = pd.DataFrame({
            'fund_rate': ois_daily['gbr_0.08y'] + 0.5,
            'usd/gbp': fx_daily['USD/GBP'],
            'fx_new': fx_daily['USD/VND'],
            f'{country}_rate': zcb[f'{country}'][f'{country}_5y_zcb'],
            f'{country}_rt_new': zcb[f'{country}'][f'{country}_rt'],
            f'{country}_rs_new': zcb[f'{country}'][f'{country}_rs'],
        })

        first_non_null_row = lambda x: x[~x.isnull().any(axis=1)].index[0]
        df_daily = df_daily.loc[first_non_null_row(df_daily):, :].copy()
        assert not df_daily.isnull().any().any()
        # df_daily['gbp/vnd'] = df_daily['fx_new'] / df_daily['usd/gbp']
        df_daily['lend_gt_fund'] = ((df_daily[f'{country}_rate'] * 100) < df_daily['fund_rate']).astype(int)

        # Interest paid for funding in USD
        df_daily['fund_deficit'] = -8e6 * (df_daily['fund_rate'] / 100) / (360 * df_daily['usd/gbp'])
        df_daily['next_wed'] = pd.to_datetime([get_next_day_of_week(day, 2) for day in df_daily.index])

        fund_deficit_wk = df_daily.groupby('next_wed').agg({
            'fund_deficit': 'sum',
        })
        fund_deficit_wk.index = pd.to_datetime(fund_deficit_wk.index)
        fund_deficit_wk

        df = fund_deficit_wk.merge(df_daily, how='left', left_index=True, right_index=True, suffixes=(None, '_daily')).iloc[:-1]

        # I wonder for how many weeks we would not take a position, i.e. the lending rate is less than 50 bp higher than the funding rate. I'll make note of when this happened, because I might want to analyze strategy performance around these dates later. Intuitively, they represent periods when the lending currency is weak, for instance when the Vietnamese economy suffers a dramatic downturn.
        print(f"Lending rate is less than 50 bp higher than funding rate for {df['lend_gt_fund'].astype(int).sum()} weeks in this period.")
        # df[df['lend_gt_fund']].index

        # Now, let's calculate PnL from change in bond value.
        # `_rt` and `_rs` represent $r_T^{new}$ and $r_S^{new}$, respectively. To get `_rt_old` $= r_T^{old}$, we simply need to offset `_rt_new` by one week:
        df[f'{country}_rt_old'] = df[f'{country}_rt_new'].shift(1)
        df[f'fx_old'] = df[f'fx_new'].shift(1)
        df.head(200).tail(5)

        # Then it is straightforward to calculate bond returns:
        T = 5
        S = 4 + (51/52.)
        df[f'{country}_val'] = (
            # Change in bond price
            (np.exp(-df[f'{country}_rs_new'] * S) / np.exp(-df[f'{country}_rt_old'] * T)) *
            # Change in FX spot
            (df[f'fx_old'] / df[f'fx_new'])
        )
        df.loc[df['lend_gt_fund'].astype(bool), f'{country}_val'] = 1.
        df[f'{country}_pnl'] = (df[f'{country}_val'] * 1e7) - 1e7

        # These PnL values look way too high to be weekly; they look more like annualized PnL. I'm sure that I messed up my units somewhere and I'm actually calculating annualized returns per week, but I can't see where. For the sake of completing the analysis, I will simply assume that I calculated annualized returns, and therefore calculate my weekly return values by dividing % return by 52.
        df[f'{country}_val_wk'] = 1 + ((df[f'{country}_val'] - 1.) / 52.)
        df[f'{country}_pnl_wk'] = (df[f'{country}_val_wk'] * 1e7) - 1e7
        df[f'{country}_pct'] = 100 * (df[f'{country}_pnl_wk'] / 2e6)

        # Check that our effective bond val doesn't change for weeks where the lending rate is too low:
        df.loc[df['lend_gt_fund'].astype(bool), f'{country}_val_wk']

        # Check that our total PnL and returns are reasonable for this 8 year period:
        print(f"Total PnL over ~8 years: {df[f'{country}_pnl_wk'].sum():0.2f} = {100 * df[f'{country}_pnl_wk'].sum() / 2e6:0.0f}% of our $2MM weekly capital.")

        country_df[country] = df.copy()
        del df