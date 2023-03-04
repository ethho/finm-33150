#!/usr/bin/env python
"""
Author: Ethan Ho
Date: 3/2/2023
License: MIT
Usage:

python3 strat_returns.py
# or from Python3: from strat_returns import main as strat_returns_main
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
import matplotlib.pyplot as plt
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
from final_proj import read_uszcb, tenor_wk_to_years

@functools.lru_cache()
def tenor_wk_to_months(wk: int) -> float:
    """
    Convert tenor from weeks to months.
    """
    return wk * 13 / 52.


@functools.lru_cache()
def k_div_x1(tenor_wk: float, k: float = 1.) -> float:
    """
    Compute the following from Chua et al 2005, using `tenor_wk` in units
    of weeks:

    k/59 dollars are invested in the 60-month
    bond, with a duration of 59 months over the one-month holding period,
    the amount of cash invested in a bond of maturity of X months, with
    duration of (X-1) months, will be k/(X-1) dollar.
    """
    if tenor_wk <= 4.:
        return float('nan')
    return k / ((tenor_wk / 4) - 1)

def strat_1a_returns(
    zcb: pd.DataFrame,
    tenors: List[float],
    is_long: bool = True,
    broker_borrow_rate: float = 50,
    capital: float = 10_000_000,
    leverage: float = 5.,
    start_date: str = None,
):
    """
    Given bond values `val`, calculate returns assuming duration and cash-weighted
    positions for Strategy 1-A as described in Chua et al 2005.
    """
    val = zcb.stack(1).swaplevel().loc['val'][[4.] + tenors]
    long_mult = 1 if is_long else -1
    assert 4. in val.columns, f"please include a 4-week rate in `val`"

    # Trim data to start on first non-null
    if start_date is None:
        last_non_null = val[val.isnull().any(axis=1)].iloc[-1].name
        start_date = val.loc['2001-08-01':].iloc[1].name.strftime('%Y-%m-%d')
        print(f"Using first non-null date as {start_date=}")
    val = val.loc[start_date:].copy(deep=True)

    # Calculate the hedge factors for each position as described in Chua et al 2005.
    hedge_factors = long_mult * pd.Series({
        tenor_wk: k_div_x1(tenor_wk)
        for tenor_wk in val.columns
    })

    # Borrow (deposit) at the 4-week rate if we take a long (short) position.
    hedge_factors[4] = -hedge_factors[[col for col in hedge_factors.index if col != 4]].sum()

    # Normalize the hedge_factors, so that the sum of hedge factors for
    # our long leg equals `capital`. For example, our normalized hedge factors
    # would look like:
    #  tenor   factor
    # 4     -1.000000
    # 52     0.665208
    # 156    0.210066
    # 260    0.124726
    hedge_factors *= 1 / hedge_factors[hedge_factors > 0].sum()
    notional_val = hedge_factors * capital
    assert abs(hedge_factors.sum()) < 1e-8, f"not cash-neutral: {hedge_factors=}"

    # Rate of return in annualized basis points
    return_rate_no_fees = 1e4 * (val - 1)

    # If we had no borrow or tx fees, PnL is a straightforward calculation:
    pnl_port_no_fees = return_rate_no_fees * notional_val / 1e4
    pnl_tot_no_fees = pnl_port_no_fees.sum(axis=1)

    # ...but we need to add a tx fee to each trade.
    # Add a 0.5 bp transaction fee for every position, as described in
    # Chua 2005 Eq. 4.
    tx_cost_port = -pd.Series(
        0.00005 * tenor_wk_to_years(notional_val.index.values) * notional_val.values,
        index=notional_val.index,
    ).abs()
    tx_cost_tot = tx_cost_port.sum()

    # We also assume an additional broker-imposed `broker_borrow_rate` (50 bp by default)
    # on any cash borrowed on leverage (4/5 of `capital` by default).
    # Since we normalized our hedge ratios, this fee is simply 50 bp
    # on our `capital`.
    borrow_fee = -(broker_borrow_rate / (1e4 * 13.)) * (capital * (leverage - 1) / leverage)

    # We can now calculate PnL including fees:
    pnl_port_w_tx_cost = pnl_port_no_fees + tx_cost_port
    pnl_tot = pnl_port_w_tx_cost.sum(axis=1) + borrow_fee

    return {
        'hedge_factors': hedge_factors,
        'return_rate_no_fees': return_rate_no_fees,
        'pnl_tot': pnl_tot,
    }

def get_pnl_1A_135(
    zcb, signal: pd.DataFrame,
    capital: float = 10_000_000,
    leverage: float = 5.,
):
    """
    Calculate the return series for Strategy 1-A given a `signal`.
    """
    assert 'signal' in signal.columns

    # Strategy pt1A_135l: portfolio returns for long position on Strategy 1-A
    # portfolio using 1, 3, 5 year maturities
    long_results = strat_1a_returns(
        zcb,
        tenors=[52., 156., 260.],
        capital=capital,
        leverage=leverage,
        is_long=True,
    )

    # Strategy pt1A_135s: portfolio returns for short position on Strategy 1-A
    # portfolio using 1, 3, 5 year maturities
    short_results = strat_1a_returns(
        zcb,
        tenors=[52., 156., 260.],
        capital=capital,
        leverage=leverage,
        is_long=False,
    )

    # Choose long, short, or flat depending on `signal`
    pnl = pd.DataFrame({
        1:  long_results['pnl_tot'],
        -1: short_results['pnl_tot'],
        0: 0,
        'signal': signal['signal'],
        'pnl': float('nan'),
    }, index=short_results['pnl_tot'].index)
    pnl['pnl'] = pnl.apply(
        lambda row: row.loc[row.signal],
        axis=1
    )
    pnl['pnl_pct'] = 100 * pnl['pnl'] / (capital / leverage)
    pnl['long_pnl_pct'] = 100 * pnl[1] / (capital / leverage)
    pnl['short_pnl_pct'] = 100 * pnl[-1] / (capital / leverage)

    return pnl

def get_signal_n1A_135(
    zcb: pd.DataFrame,
    tenors=[52., 156., 260.],
    sigma_thresh=0.8,
    window_size=102,
) -> pd.Series:
    """
    Generate the trading signal for naive Strategy 1-A. The trading signal
    will be 1 if we should take a long position on the Strategy 1-A portfolio,
    0 if we should be flat, and -1 if we should short. A non-zero signal
    is emitted if the mean of the 4-week forward rate curve across all
    maturities is >= `sigma_thresh` (<= -`sigma_thresh` for short) standard
    deviations from the mean, where mean and STD are calculated
    over the last `window_size` 4-week periods.
    """
    # Here, we shift the signal by one period (4 weeks)
    # to avoid lookahead bias, so that the date index represents t,
    # the time when we're selling the position that we decided on a month ago.
    # This is consistent with our date indexing for portfolio returns data:
    # that date represents the day we _closed_ our 4-week long position.
    fwd = zcb.stack(1).swaplevel().loc['fwd'][tenors].shift(1)
    fwd_mean = fwd.mean(axis=1)
    df = fwd_mean.rolling(window=window_size).agg(['mean', 'std'])
    df['low_thresh']  = df['mean'] - df['std'] * sigma_thresh
    df['high_thresh'] = df['mean'] + df['std'] * sigma_thresh
    df['fwd'] = fwd_mean
    df['signal'] = 0
    df.loc[df['fwd'] >= df['high_thresh'], 'signal'] = -1
    df.loc[df['fwd'] <= df['low_thresh'], 'signal']  = 1
    return df

def main(
    zcb_fp='./data/final_proj/uszcb.csv',
    strat_results_out_fp='./data/final_proj/strat_n1A_135.csv',
):
    zcb = read_uszcb(zcb_fp)
    naive_signal = get_signal_n1A_135(
        zcb,
        tenors=[52., 156., 260.],
        sigma_thresh=1.,
        window_size=102,
    )
    strat_results = get_pnl_1A_135(zcb, signal=naive_signal)
    strat_results.to_csv(strat_results_out_fp)
    print(f"Wrote strategy returns to {strat_results_out_fp}")
    return strat_results['pnl']


if __name__ == '__main__':
    main(*sys.argv[1:])