#!/usr/bin/env python

import os
import re
import sys
import json
from typing import List, Dict, Tuple, Optional
import functools
import pandas as pd
import numpy as np
from scipy.stats import norm, probplot
import quandl
import plotly.express as px
from memoize.dataframe import memoize_df
from lmfit.models import SkewedGaussianModel

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
    get_col_groups(tuple(yc_daily.columns.tolist()))
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
    Adapted from Zero_And_Spot_Curves.ipynb
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
    Convert tenor in weeks to years.
    """
    return wk * 7 / 364.

def get_zcb_curve(spot, coupons_per_year=2, holding_period=7/364.):
    """
    Adapted from Zero_And_Spot_Curves.ipynb
    """
    cpn_freq = 1 / float(coupons_per_year)
    for tenor_wk, spot_rate in spot.items():
        tenor = tenor_wk_to_years(tenor_wk)
        if tenor <= 0.001:
            continue
        if tenor <= 1.:
            # US T-bills (<=1 year maturity) have no coupons
            spot.loc[tenor_wk] = spot_rate
        else:
            times = np.arange(tenor-cpn_freq, 0, step=-cpn_freq)[::-1]
            coupon_half_yr = cpn_freq * spot_rate
            idx_wk = [tenor_wk_to_years(tenor_wk) for tenor_wk in spot.index.values]
            z = np.interp(times, idx_wk, spot.values) # Linear interpolation
            preceding_coupons_val = (coupon_half_yr*np.exp(-z*times)).sum()
            spot.loc[tenor_wk] = -np.log((1-preceding_coupons_val)/(1+coupon_half_yr))/tenor

    spot_copy = spot.copy()
    spot_copy.index = [tenor_wk_to_years(tenor_wk) for tenor_wk in spot_copy.index]
    for tenor_wk, spot_rate in spot.items():
        tenor = tenor_wk_to_years(tenor_wk)
        # Calculate bond price for maturities,
        # for example T = 5 years and S = 5 years - 1 week
        T = tenor
        S = T - holding_period
        # tenor_name = str(int(tenor)) if tenor == int(tenor) else f"{tenor:0.2f}"
        tenor_name = str(tenor_wk)
        spot.loc[f'{tenor_name}_rt'] = bond_price(spot_copy, coupon_rate=spot.loc[tenor_wk], tenor=T, coupon_freq=cpn_freq)
        spot.loc[f'{tenor_name}_rs'] = bond_price(spot_copy, coupon_rate=spot.loc[tenor_wk], tenor=S, coupon_freq=cpn_freq)
    return spot

TENOR_WEEK_MAP = {
    (1, 'm'): 4,
    (2, 'm'): 8,
    (3, 'm'): 13,
    (4, 'm'): 17,
    (6, 'm'): 26,
    (12, 'm'): 52,
    (1, 'y'): 52,
    (2, 'y'): 52 * 2,
    (3, 'y'): 52 * 3,
    (5, 'y'): 52 * 5,
    (7, 'y'): 52 * 7,
    (10, 'y'): 52 * 10,
    (20, 'y'): 52 * 20,
    (30, 'y'): 52 * 30,
}

def get_zcb_curves(df_raw):
    # Get groups by column prefix
    grps: Dict[List[Dict]] = get_col_groups(tuple(df_raw.columns.tolist()))
    zcb_dict = dict()
    for cty, cols in grps.items():
        for item in cols:
            item['tenor_wk'] = TENOR_WEEK_MAP[(int(item['tenor']), item['unit'])]
        df = df_raw.rename(columns={
            item['col']: item['tenor_wk']
            for item in cols
        })
        zcb = (df / 100.).apply(get_zcb_curve, axis=1)

        holding_period = 30/364.
        rename_dict = dict()
        for item in cols:
            tenor_wk = item['tenor_wk']
            tenor = tenor_wk_to_years(tenor_wk)
            tenor_name = str(int(tenor_wk))
            rename_dict[tenor_wk] = f"{tenor_name}_zcb"
            T = tenor
            S = tenor - holding_period
            rt = zcb[f"{tenor_name}_rt"]
            rs = zcb[f"{tenor_name}_rs"]
            zcb[f'{tenor_name}_val'] = (
                (np.exp(-rs * S) / np.exp(-rt.shift(1) * T))
            )
            # zcb[f'{tenor_name}_ret'] = (zcb[f'{tenor_name}_val'] - zcb[f'{tenor_name}_val'].shift(1)) / zcb[f'{tenor_name}_val'].shift(1)
        zcb.rename(columns=rename_dict, inplace=True)
        zcb.index.name = 'date'
        zcb_dict[cty] = zcb
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

def main(out_fp='./data/uszcb.csv'):
    start_date = '1990-01-01'
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
    zcb = get_zcb_curves(yc_monthly)
    uszcb = zcb['usa']
    # df = unstack_zcb_df(uszcb)
    uszcb.to_csv(out_fp)
    print(f'Wrote US ZCB rates to {out_fp}')

if __name__ == '__main__':
    main(*sys.argv[1:])