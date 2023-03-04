#!/usr/bin/env python
"""
Author: Ethan Ho
Date: 3/2/2023
License: MIT
Usage:

python3 ps_returns.py
# or from Python3: from ps_returns import main as ps_returns_main
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
    traded_cash: float = 10_000_000, leverage: float = 5.,
):
    """
    Given bond values `val`, calculate returns assuming duration and cash-weighted
    positions for Strategy 1-A as described in Chua et al 2005.
    """
    val = zcb.stack(1).swaplevel().loc['val'][[4.] + tenors]
    long_mult = 1 if is_long else -1
    assert 4. in val.columns, f"please include a 4-week rate in `val`"

    # Calculate the hedge factors for each position as described in Chua et al 2005.
    hedge_factors = long_mult * pd.Series({
        tenor_wk: k_div_x1(tenor_wk)
        for tenor_wk in val.columns
    })

    # Borrow (deposit) at the 4-week rate if we take a long (short) position.
    hedge_factors[4] = -hedge_factors[[col for col in hedge_factors.index if col != 4]].sum()

    # Normalize the hedge_factors, so that the sum of hedge factors for
    # our long leg equals `traded_cash`. For example, our normalized hedge factors
    # would look like:
    #  tenor   factor
    # 4     -1.000000
    # 52     0.665208
    # 156    0.210066
    # 260    0.124726
    hedge_factors *= 1 / hedge_factors[hedge_factors > 0].sum()
    notional_val = hedge_factors * traded_cash
    assert abs(hedge_factors.sum()) < 1e-8, f"not cash-neutral: {hedge_factors=}"

    # Our bond value for the 4-week T-bill is always < 1.
    # Convert it to the reciprocal so that we can multiply hedge_factors
    # to get PnL.
    val[4.] = 1 / val[4.]

    # Rate of return in annualized basis points
    return_rate_no_fees = 1e4 * (val - 1)

    # If we had no borrow or tx fees, PnL is a straightforward calculation:
    pnl_pos_no_fees = return_rate_no_fees * notional_val / 1e4
    pnl_tot_no_fees = pnl_pos_no_fees.sum(axis=1)

    # ...but we need to add a tx fee to each trade.
    # Add a 0.5 bp transaction fee for every position, as described in
    # Chua 2005 Eq. 4.
    tx_cost_pos = -pd.Series(
        0.00005 * tenor_wk_to_years(notional_val.index.values) * notional_val.values,
        index=notional_val.index,
    ).abs()
    tx_cost_tot = tx_cost_pos.sum()

    # We also assume an additional broker-imposed `broker_borrow_rate` (50 bp by default)
    # on any cash borrowed on leverage (4/5 of `traded_cash` by default).
    # Since we normalized our hedge ratios, this fee is simply 50 bp
    # on our `traded_cash`.
    borrow_fee = -(broker_borrow_rate / (1e4 * 13.)) * (traded_cash * (leverage - 1) / leverage)

    # We can now calculate PnL including fees:
    pnl_pos_w_tx_cost = pnl_pos_no_fees + tx_cost_pos
    pnl_tot = pnl_pos_w_tx_cost.sum(axis=1) + borrow_fee

    return {
        'hedge_factors': hedge_factors,
        'return_rate_no_fees': return_rate_no_fees,
        'pnl_tot': pnl_tot,
    }

def get_pnl_1A_135(zcb, signal: pd.DataFrame):
    """
    Calculate the return series for Strategy 1-A.

    Project Part 1B: Calculate returns of all possible positions (PS) every month.
    Calculate return series for all possible positions that we can take.
    E.g. for Strategy 2A, calculate the PnL for every possible portfolio:

    Short 52-week, buy 104-week
    Buy 52-week, short 104-week
    Short 52-week, buy 156-week
    Buy 52-week, short 156-week
    ...
    Short 156-week, buy 1560-week
    Buy 156-week, short 1560-week
    """
    assert 'signal' in signal.columns

    # Strategy pt1A_135l: portfolio returns for long position on Strategy 1-A
    # portfolio using 1, 3, 5 year maturities
    long_results = strat_1a_returns(
        zcb,
        tenors=[52., 156., 260.],
        is_long=True
    )

    # Strategy pt1A_135s: portfolio returns for short position on Strategy 1-A
    # portfolio using 1, 3, 5 year maturities
    short_results = strat_1a_returns(
        zcb,
        tenors=[52., 156., 260.],
        is_long=False
    )

    # Choose long, short, or flat depending on `signal`
    pnl = pd.DataFrame({
        1:  long_results['pnl_tot'],
        -1: short_results['pnl_tot'],
        0: 0.,
        'signal': signal['signal'],
        'pnl': float('nan'),
    }, index=short_results['pnl_tot'].index)
    pnl['pnl'] = pnl.apply(
        lambda row: row.loc[row.signal],
        axis=1
    )

    return pnl

def get_signal_n1A_135(
    zcb: pd.DataFrame,
    tenors=[52., 156., 260.],
    sigma_thresh=1.,
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
    fwd = zcb.stack(1).swaplevel().loc['fwd'][tenors]
    fwd_mean = fwd.mean(axis=1)
    df = fwd_mean.rolling(window=window_size).agg(['mean', 'std'])
    df['low_thresh']  = df['mean'] - df['std'] * sigma_thresh
    df['high_thresh'] = df['mean'] + df['std'] * sigma_thresh
    df['fwd'] = fwd_mean
    df['signal'] = 0
    df.loc[df['fwd'] >= df['high_thresh'], 'signal'] = 1
    df.loc[df['fwd'] <= df['low_thresh'], 'signal'] = -1
    return df

def main(
    zcb_out_fp='./data/uszcb.csv',
):
    zcb = read_uszcb(zcb_out_fp)
    naive_signal = get_signal_n1A_135(
        zcb,
        tenors=[52., 156., 260.],
        sigma_thresh=1.,
        window_size=102,
    )
    ps_results = get_pnl_1A_135(zcb, signal=naive_signal)
    return ps_results['pnl_tot']


if __name__ == '__main__':
    main(*sys.argv[1:])