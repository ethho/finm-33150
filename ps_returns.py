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

def get_position_returns(zcb):
    """
    Calculate the return series for every possible hedged position every month.

    Project Part 1B: Calculate returns of all possible positions (PS) every month.
    Calculate return series for all possible positions that we can take.
    E.g. for Strategy 2A, calculate the return series as a proportion of
    unleveraged notional for every possible position:

    Short 52-week, buy 104-week
    Buy 52-week, short 104-week
    Short 52-week, buy 156-week
    Buy 52-week, short 156-week
    ...
    Short 156-week, buy 1560-week
    Buy 156-week, short 1560-week
    """
    notional = 10_000_000
    k = notional
    four_wk_rate = zcb.loc[:, (4., 'zcb')] * 4 / 52
    four_wk_interest_paid = lambda prop: ((1 + four_wk_rate) * notional * prop) - notional
    val = zcb.stack(1).swaplevel().loc['val']
    spot_4wk = zcb.stack(1).swaplevel().loc['spot', 4.]

    # Strategy n1A_135
    # int_paid = zcb.stack(1).swaplevel().loc['spot', 4.]
    # int_paid2 = (1 / zcb.stack(1).swaplevel().loc['val', 4.]) - 1
    # breakpoint()
    df = strat_1a_returns(
        zcb,
        tenors=[52., 156., 260.],
    )
    return df

def main(
    zcb_out_fp='./data/uszcb.csv',
):
    zcb = read_uszcb(zcb_out_fp)
    ps_returns = get_position_returns(zcb)


if __name__ == '__main__':
    main(*sys.argv[1:])