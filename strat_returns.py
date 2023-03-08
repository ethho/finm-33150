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
import itertools
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot
from statsmodels.regression.rolling import RollingWLS
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


class StrategyBase(dict):
    """
    Base class for strategy development.
    """

    def __init__(
        self,
        zcb: pd.DataFrame,
        tenors: List[float],
        file_stub: str,
        capital: float = 10_000_000,
        leverage: float = 5.,
        start_date: str = None,
        broker_borrow_rate: float = 50,
    ):
        self.zcb = zcb
        self.file_stub = file_stub
        self.tenors = tenors
        self.capital = capital
        self.leverage = leverage
        self.start_date = start_date
        self.broker_borrow_rate = broker_borrow_rate
        self._param_names = [
            'file_stub', 'tenors', 'capital', 'leverage',
            'start_date', 'broker_borrow_rate',
        ]

    def __call__(self, *args, **kw):
        return self.get_pnl(*args, **kw)

    def get_params(self) -> Dict:
        return {
            param_name: getattr(self, param_name)
            for param_name in self._param_names
        }

    def write_all(self):
        for df_name in (
            'hedge_factors',
            'signal',
            'pnl',
        ):
            df = getattr(self, df_name)
            fp = f"{self.file_stub}_{df_name}.csv"
            df.to_csv(fp)
            print(f"Wrote '{df_name}' to {fp}")

        # Write params as JSON
        params_fp = f"{self.file_stub}_params.json"
        with open(params_fp, "w") as f:
            f.write(json.dumps(self.get_params(), sort_keys=True, indent=2))
            print(f"Wrote 'params' to {params_fp}")

    def normalize_hedge_factors(self, hedge_factors):
        # Normalize the hedge_factors, so that the sum of hedge factors for
        # our long leg equals `capital`. For example, for Strategy 1-A,
        # our normalized hedge factors would look like:
        #  tenor   factor
        # 4     -1.000000
        # 52     0.665208
        # 156    0.210066
        # 260    0.124726
        hedge_factors *= 1 / hedge_factors[hedge_factors > 0].sum()
        assert abs(hedge_factors.sum()) < 1e-8, f"not cash-neutral: {hedge_factors=}"
        self.hedge_factors = hedge_factors
        return hedge_factors

    def strat_returns(
        self,
        is_long: bool = True,
        start_date: str = None,
    ):
        """
        Given bond values `val`, calculate returns.
        """
        val = self.zcb.stack(1).swaplevel().loc['val'][[4.] + self.tenors]
        assert 4. in val.columns, f"please include a 4-week rate in `val`"

        # Trim data to start on first non-null
        if start_date is None:
            last_non_null = val[val.isnull().any(axis=1)].iloc[-1].name
            start_date = val.loc['2001-08-01':].iloc[1].name.strftime('%Y-%m-%d')
            print(f"Using first non-null date as {start_date=}")
        val = val.loc[start_date:].copy(deep=True)
        self.val = val

        # Calculate and normalize hedge factors: the position sizes
        # we need to use to have a cash and duration neutral portfolio.
        hedge_factors_raw = self.get_hedge_factors(val, long_mult=1 if is_long else -1)
        self.hedge_factors = self.normalize_hedge_factors(hedge_factors_raw)

        # Rate of return in annualized basis points
        return_rate_no_fees = 1e4 * (val - 1)

        # If we had no borrow or tx fees, PnL is a straightforward calculation:
        notional_val = self.hedge_factors * self.capital
        pnl_pos_no_fees = return_rate_no_fees * notional_val / 1e4
        pnl_tot_no_fees = pnl_pos_no_fees.sum(axis=1)

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
        borrow_fee = -(self.broker_borrow_rate / (1e4 * 13.)) * (self.capital * (self.leverage - 1) / self.leverage)

        # We can now calculate PnL including fees:
        pnl_pos_w_tx_cost = pnl_pos_no_fees + tx_cost_port
        pnl_tot = pnl_pos_w_tx_cost.sum(axis=1) + borrow_fee
        total_fees = tx_cost_port.sum() + borrow_fee

        results = {
            'hedge_factors': self.hedge_factors,
            'return_rate_no_fees': return_rate_no_fees,
            'pnl_tot_no_fees': pnl_tot_no_fees,
            'pnl_tot': pnl_tot,
            'pnl_pos': pnl_pos_w_tx_cost,
            'total_fees': total_fees,
        }
        return results

    def get_pnl(self):
        """
        Calculate the monthly returns for long and short positions on this
        portfolio. Calculate PnL by choosing which (long, short, or flat)
        position to take based on the Strategy's `signal`.
        """
        self.signal = self.get_signal()
        assert 'signal' in self.signal.columns

        # Returns for the long position
        long_results = self.strat_returns(is_long=True)

        # Portfolio returns for short position
        short_results = self.strat_returns(is_long=False)

        # Get the total fees paid for long and short portfolios.
        # They _should_ be the same.
        long_fees  = abs(long_results['total_fees'])
        short_fees = abs(short_results['total_fees'])
        assert long_fees == short_fees, (long_fees, short_fees)
        self.fees = long_fees

        # Choose long, short, or flat depending on `signal`
        pnl = pd.DataFrame({
            1:  long_results['pnl_tot'],
            -1: short_results['pnl_tot'],
            0: 0,
            'pnl_no_fees': long_results['pnl_tot_no_fees'],
            # Same as -pnl_no_fees by definition
            # 'short_pnl_no_fees': short_results['pnl_tot_no_fees'],
            'signal': self.signal['signal'],
            'pnl': float('nan'),
        }, index=short_results['pnl_tot'].index)
        pnl['pnl'] = pnl.apply(
            lambda row: row.loc[row.signal],
            axis=1
        )
        collateral = self.capital / self.leverage
        pnl['pnl_pct'] = 100 * pnl['pnl'] / collateral
        pnl['pnl_no_fees_pct'] = 100 * pnl['pnl_no_fees'] / collateral
        pnl['long_pnl_pct'] = 100 * pnl[1] / collateral
        pnl['short_pnl_pct'] = 100 * pnl[-1] / collateral

        self.pnl = pnl
        return self.pnl

    def _get_pnl_if_same_sig(self, row):
        """
        Get the PnL depending on `signal`, and whether signal is the same as
        last time. If signal is same as last time, use the PnL with no fees
        since we already incorporated the total fees for the position during the
        first month.

        Used in `get_pnl` override in child classes.
        """
        if row.signal == 1:
            if row.same_sig:
                return row.pnl_no_fees
            else:
                return row[1]
        elif row.signal == -1:
            if row.same_sig:
                return -row.pnl_no_fees
            else:
                return row[-1]
        else:
            return 0.

    def fit_ewls(
        self,
        exog: pd.Series,
        equation="pnl ~ curv_z + 0",
        half_life: float = 6,
        contemp_window=0,
    ):
        """
        Fit a rolling exponentially weighted least squares regression model.

        Fit a univariate `statsmodels.regression.rolling.RollingWLS`
        with exponentially decaying weights (half life equal to `half_life`),
        using the PnL without fees as the endogenous variable, and `exog` as the
        exogenous variable. `equation` is a string representing the
        linear relationship. `contemp_window` is unused.
        """
        # Get strategy returns without fees
        long_strat_results = self.strat_returns(is_long=True)
        pnl = long_strat_results['pnl_tot_no_fees']
        pnl.name = 'pnl'
        df = pnl.to_frame(name='pnl').merge(
            exog, how='inner', left_index=True,
            right_index=True, suffixes=('_pnl', None))

        # Trim data to first non-null index
        null_indices = df[df[['pnl', exog.name]].isnull().any(axis=1)].index
        if not null_indices.empty:
            last_null = null_indices[-1]
            first_non_null = df.loc[last_null:].index[1]
            df = df.loc[first_non_null:]

        # Calculate exponential weights from half-life
        lambda_ = pow(1/2., 1/half_life)
        weights = np.array([pow(lambda_, t) for t in range(len(df) - contemp_window)])[::-1]
        weights *= 1e3

        # Fit EWLS
        exwt_mod = RollingWLS.from_formula(
            equation,
            data=df.iloc[contemp_window:],
            window=None,
            min_nobs=half_life * 2,
            expanding=True,
            weights=weights
        )
        exwt = exwt_mod.fit()
        return exwt


class Strat1A(StrategyBase):
    """
    Adapted from Strategy 1-A in Chua et al 2005.
    """

    def __init__(
        self,
        *args,
        sigma_thresh=0.8,
        window_size=102,
        **kw,
    ):
        super(Strat1A, self).__init__(*args, **kw)
        self.sigma_thresh = sigma_thresh
        self.window_size = window_size
        self._param_names.extend([
            'sigma_thresh', 'window_size',
        ])

    def get_hedge_factors(self, val, long_mult):
        """
        Assuming duration and cash-weighted
        positions for Strategy 1-A as described in Chua et al 2005.
        """
        # Calculate the hedge factors for each position as described in Chua et al 2005.
        hedge_factors = long_mult * pd.Series({
            tenor_wk: k_div_x1(tenor_wk)
            for tenor_wk in val.columns
        })

        # Borrow (deposit) at the 4-week rate if we take a long (short) position.
        hedge_factors[4] = -hedge_factors[[col for col in hedge_factors.index if col != 4]].sum()
        self.hedge_factors_raw = hedge_factors
        return self.hedge_factors_raw

    def get_signal(self) -> pd.Series:
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
        fwd = self.zcb.stack(1).swaplevel().loc['fwd'][self.tenors].shift(1)
        fwd_mean = fwd.mean(axis=1)
        df = fwd_mean.rolling(window=self.window_size).agg(['mean', 'std'])
        df['low_thresh']  = df['mean'] - df['std'] * self.sigma_thresh
        df['high_thresh'] = df['mean'] + df['std'] * self.sigma_thresh
        df['fwd'] = fwd_mean
        df['signal'] = 0
        df.loc[df['fwd'] >= df['high_thresh'], 'signal'] = -1
        df.loc[df['fwd'] <= df['low_thresh'], 'signal']  = 1
        return df


class Strat1C(Strat1A):
    """
    Strat1A, but the signal is the predicted
    PnL (with fees) from the rolling exponentially weighted linear regression on
    the Z-score of the mean forward rate.
    """

    def __init__(
        self,
        *args,
        window_size=102,
        half_life=12,
        ci_alpha=0.2,
        **kw,
    ):
        super(Strat1C, self).__init__(*args, **kw)
        self.window_size = window_size
        self.half_life = half_life
        self.ci_alpha = ci_alpha
        self._param_names.extend([
            'half_life', 'window_size', 'ci_alpha',
        ])

    def get_signal(self) -> pd.Series:
        """
        Generate the trading signal for Strategy 1-C. The trading signal
        will be 1 if we should take a long position on the 1-C portfolio,
        0 if we should be flat, and -1 if we should short. A non-zero signal
        is emitted if any value in the confidence interval (alpha equal to
        `ci_alpha`) of predicted PnL is greater than the trading fees.
        This prediction is generated by a univariate rolling linear regression
        with exponentially decaying weights. The endogenous variable is PnL and
        the exogenous variable is the Z-score of the mean forward rate,
        calculated over the past `window_size` 4-week periods.
        """
        # Here, we shift the signal by one period (4 weeks)
        # to avoid lookahead bias, so that the date index represents t,
        # the time when we're selling the position that we decided on a month ago.
        # This is consistent with our date indexing for portfolio returns data:
        # that date represents the day we _closed_ our 4-week long position.
        fwd = self.zcb.stack(1).swaplevel().loc['fwd'][self.tenors].shift(1)
        fwd_mean = fwd.mean(axis=1)

        # Calculate Z-score of curvature relative to its `window_size`-period
        # simple moving average
        sma = fwd_mean.rolling(window=self.window_size).agg(['mean', 'std'])
        z_score = (fwd_mean - sma['mean']) / sma['std']
        z_score.name = 'z_score'
        sma['z_score'] = z_score

        # Get a regression coefficient (with CI bounds) by fitting
        # a rolling, exponentially weighted, univariate least squares
        self.ew_model = self.fit_ewls(
            exog=z_score,
            equation="pnl ~ z_score + 0",
            half_life=self.half_life,
        )
        coeff = self.ew_model.conf_int(alpha=self.ci_alpha)
        coeff.columns = ['ci_lower', 'ci_upper']
        coeff.loc[:, 'ci_mid'] = self.ew_model.params

        # Use the regression coefficient to predict today's PnL (without fees)
        # from last month's curvature Z-score, with confidence interval
        # at alpha equal to `self.ci_alpha`.
        df = sma.merge(
            coeff, how='inner', left_index=True,
            right_index=True, suffixes=(None, '_coeff'),
        )
        pred = coeff.multiply(df['z_score'], axis=0)
        df['pnl_no_fees_pred'] = pred['ci_mid']
        pred[['ci_lower', 'ci_mid', 'ci_upper']] = (
            pred[['ci_lower', 'ci_mid', 'ci_upper']]
            .apply(lambda row: row.sort_values(ascending=row.ci_mid >= 0, ignore_index=True), axis=1)
        )

        # Get the total fees paid for long and short portfolios.
        # They _should_ be the same.
        long_fees  = abs(self.strat_returns(is_long=True )['total_fees'])
        short_fees = abs(self.strat_returns(is_long=False)['total_fees'])
        assert long_fees == short_fees, (long_fees, short_fees)

        # Use the CI upper or CI lower (whichever is optimistic)
        # to calculate PnL with fees, depending on whether the suggested
        # position (CI midpoint) is likely to be long or short.
        df['pnl_pred'] = pred.apply(
            lambda row:
                # PnL minus fees if long
                max(row.ci_upper - long_fees, 0.) if row.ci_mid >= 0 else
                # PnL (negative) plus fees if short
                min(row.ci_upper + long_fees, 0.),
        axis=1)

        # Generate our signal: buy (sell) the portfolio if the predicted
        # PnL plus fees is positive (negative).
        df['signal'] = 0
        df.loc[df['pnl_pred'] > 0, 'signal'] = 1
        df.loc[df['pnl_pred'] < 0, 'signal'] = -1
        return df

    def pred_with_fees(self, without: pd.Series, fees: float):
        """
        Get PnL with fees from PnL `without` the added fee `fees`.
        This function sets PnL to a minimum of zero if fees are larger
        than the PnL without.
        """
        assert fees > 0, fees
        return without.apply(
            lambda x:
                max(x - fees, 0) if x >= 0 else
                min(x + fees, 0)
        )

class Strat1D(Strat1C):
    """
    Strat1C, but we don't close our position if the signal is the same
    as last month, saving on transaction fees.
    """

    def __init__(
        self,
        *args,
        **kw,
    ):
        super(Strat1D, self).__init__(*args, **kw)

    def get_pnl(self):
        """
        Overrides parent method `StrategyBase.get_pnl`.

        If we have an open position and the signal is the same, use the
        fee-free PnL, since this represents our profits when we avoid closing
        and reopening the same position.
        """
        # Reuse the parent's method
        pnl = StrategyBase.get_pnl(self)

        # `pnl` column includes the fee for buying and selling the position
        # once each. This means that we can maintain our position if the signal
        # is the same, and save the transaction fee amount by avoiding closing
        # and reopening the same position.
        pnl['same_sig'] = pnl.signal == pnl.signal.shift(1)
        pnl['pnl'] = pnl.apply(self._get_pnl_if_same_sig, axis=1)

        # We need to recalculate `pnl_pct`
        collateral = self.capital / self.leverage
        pnl['pnl_pct'] = 100 * pnl['pnl'] / collateral

        self.pnl = pnl
        return pnl

class Strat2A(StrategyBase):
    """
    Adapted from Strategy 2-A in Chua et al 2005.
    """

    def __init__(
        self,
        *args,
        sigma_thresh=0.8,
        window_size=102,
        **kw,
    ):
        super(Strat2A, self).__init__(*args, **kw)
        self.sigma_thresh = sigma_thresh
        self.window_size = window_size
        assert len(self.tenors) == 2, self.tenors
        assert self.tenors[0] < self.tenors[1], self.tenors
        self._param_names.extend([
            'sigma_thresh', 'window_size',
        ])

    def get_hedge_factors(self, val, long_mult):
        """
        Assuming duration and cash-weighted
        positions for Strategy 2-A as described in Chua et al 2005.
        """
        # Calculate the hedge factors for each position as described in Chua et al 2005.
        hedge_factors = pd.Series({
            self.tenors[0]: 1.,
            self.tenors[1]: k_div_x1(self.tenors[1]),
        })

        # Multiply the long (short) tenor's factor by -1
        # if we're buying (selling) the spread.
        if long_mult == -1:
            hedge_factors.loc[self.tenors[0]] *= -1
        elif long_mult == 1:
            hedge_factors.loc[self.tenors[1]] *= -1
        else:
            raise NotImplementedError(f'Invalid {long_mult=}')

        # Borrow (deposit) remaining cash at the 4-week rate
        # to ensure cash neutrality (58k/59)
        hedge_factors.loc[4] = -hedge_factors[[col for col in hedge_factors.index if col != 4]].sum()
        self.hedge_factors_raw = hedge_factors
        return self.hedge_factors_raw

    def get_signal(self) -> pd.Series:
        """
        Generate the trading signal for naive Strategy 2-A. The trading signal
        will be 1 if we should take a long position on the 2-A portfolio,
        0 if we should be flat, and -1 if we should short. A non-zero signal
        is emitted if the slope of the 4-week forward rate curve
        (spread between low maturity rate and high maturity)
        is >= `sigma_thresh` (<= -`sigma_thresh` for short) standard
        deviations from the mean historical slope, where mean and STD
        are calculated over the last `window_size` 4-week periods.
        """
        # Here, we shift the signal by one period (4 weeks)
        # to avoid lookahead bias, so that the date index represents t,
        # the time when we're selling the position that we decided on a month ago.
        # This is consistent with our date indexing for portfolio returns data:
        # that date represents the day we _closed_ our 4-week long position.
        fwd = self.zcb.stack(1).swaplevel().loc['fwd'][self.tenors].shift(1)
        fwd_slope = fwd.iloc[:, 0] - fwd.iloc[:, 1]
        df = fwd_slope.rolling(window=self.window_size).agg(['mean', 'std'])
        df['low_thresh']  = df['mean'] - df['std'] * self.sigma_thresh
        df['high_thresh'] = df['mean'] + df['std'] * self.sigma_thresh
        df['fwd_slope'] = fwd_slope
        df['signal'] = 0
        df.loc[df['fwd_slope'] >= df['high_thresh'], 'signal'] = -1
        df.loc[df['fwd_slope'] <= df['low_thresh'], 'signal']  = 1
        return df


class Strat2C(Strat2A):
    """
    Strat2A, but the signal is the predicted
    PnL (with fees) from the rolling exponentially weighted linear regression on
    the slope Z-score.
    """

    def __init__(
        self,
        *args,
        window_size=102,
        half_life=12,
        ci_alpha=0.2,
        **kw,
    ):
        super(Strat2C, self).__init__(*args, **kw)
        self.window_size = window_size
        self.half_life = half_life
        self.ci_alpha = ci_alpha
        assert len(self.tenors) == 2, self.tenors
        assert self.tenors[0] < self.tenors[1], self.tenors
        self._param_names.extend([
            'half_life', 'window_size', 'ci_alpha',
        ])

    def get_signal(self) -> pd.Series:
        """
        Generate the trading signal for Strategy 3-C. The trading signal
        will be 1 if we should take a long position on the 3-C portfolio,
        0 if we should be flat, and -1 if we should short. A non-zero signal
        is emitted if any value in the confidence interval (alpha equal to
        `ci_alpha`)of predicted PnL is greater than the trading fees.
        This prediction is generated by a univariate rolling linear regression
        with exponentially decaying weights. The endogenous variable is PnL and
        the exogenous variable is the slope Z-score, calculated over the
        pas `window_size` 4-week periods.
        """
        # Here, we shift the signal by one period (4 weeks)
        # to avoid lookahead bias, so that the date index represents t,
        # the time when we're selling the position that we decided on a month ago.
        # This is consistent with our date indexing for portfolio returns data:
        # that date represents the day we _closed_ our 4-week long position.
        fwd = self.zcb.stack(1).swaplevel().loc['fwd'][self.tenors].shift(1)
        fwd_slope = fwd.iloc[:, 0] - fwd.iloc[:, 1]

        # Calculate Z-score of curvature relative to its `window_size`-period
        # simple moving average
        sma = fwd_slope.rolling(window=self.window_size).agg(['mean', 'std'])
        z_score = (fwd_slope - sma['mean']) / sma['std']
        z_score.name = 'z_score'
        sma['z_score'] = z_score

        # Get a regression coefficient (with CI bounds) by fitting
        # a rolling, exponentially weighted, univariate least squares
        self.ew_model = self.fit_ewls(
            exog=z_score,
            equation="pnl ~ z_score + 0",
            half_life=self.half_life,
        )
        coeff = self.ew_model.conf_int(alpha=self.ci_alpha)
        coeff.columns = ['ci_lower', 'ci_upper']
        coeff.loc[:, 'ci_mid'] = self.ew_model.params

        # Use the regression coefficient to predict today's PnL (without fees)
        # from last month's curvature Z-score, with confidence interval
        # at alpha equal to `self.ci_alpha`.
        df = sma.merge(
            coeff, how='inner', left_index=True,
            right_index=True, suffixes=(None, '_coeff'),
        )
        pred = coeff.multiply(df['z_score'], axis=0)
        df['pnl_no_fees_pred'] = pred['ci_mid']
        pred[['ci_lower', 'ci_mid', 'ci_upper']] = (
            pred[['ci_lower', 'ci_mid', 'ci_upper']]
            .apply(lambda row: row.sort_values(ascending=row.ci_mid >= 0, ignore_index=True), axis=1)
        )

        # Get the total fees paid for long and short portfolios.
        # They _should_ be the same.
        long_fees  = abs(self.strat_returns(is_long=True )['total_fees'])
        short_fees = abs(self.strat_returns(is_long=False)['total_fees'])
        assert long_fees == short_fees, (long_fees, short_fees)

        # Use the CI upper or CI lower (whichever is optimistic)
        # to calculate PnL with fees, depending on whether the suggested
        # position (CI midpoint) is likely to be long or short.
        df['pnl_pred'] = pred.apply(
            lambda row:
                # PnL minus fees if long
                max(row.ci_upper - long_fees, 0.) if row.ci_mid >= 0 else
                # PnL (negative) plus fees if short
                min(row.ci_upper + long_fees, 0.),
        axis=1)

        # Generate our signal: buy (sell) the portfolio if the predicted
        # PnL plus fees is positive (negative).
        df['signal'] = 0
        df.loc[df['pnl_pred'] > 0, 'signal'] = 1
        df.loc[df['pnl_pred'] < 0, 'signal'] = -1
        return df

    def pred_with_fees(self, without: pd.Series, fees: float):
        """
        Get PnL with fees from PnL `without` the added fee `fees`.
        This function sets PnL to a minimum of zero if fees are larger
        than the PnL without.
        """
        assert fees > 0, fees
        return without.apply(
            lambda x:
                max(x - fees, 0) if x >= 0 else
                min(x + fees, 0)
        )

class Strat2D(Strat2C):
    """
    Strat2C, but we don't close our position if the signal is the same
    as last month, saving on transaction fees.
    """

    def __init__(
        self,
        *args,
        **kw,
    ):
        super(Strat2D, self).__init__(*args, **kw)

    def get_pnl(self):
        """
        Overrides parent method `StrategyBase.get_pnl`.

        If we have an open position and the signal is the same, use the
        fee-free PnL, since this represents our profits when we avoid closing
        and reopening the same position.
        """
        # Reuse the parent's method
        pnl = StrategyBase.get_pnl(self)

        # `pnl` column includes the fee for buying and selling the position
        # once each. This means that we can maintain our position if the signal
        # is the same, and save the transaction fee amount by avoiding closing
        # and reopening the same position.
        pnl['same_sig'] = pnl.signal == pnl.signal.shift(1)
        pnl['pnl'] = pnl.apply(self._get_pnl_if_same_sig, axis=1)

        # We need to recalculate `pnl_pct`
        collateral = self.capital / self.leverage
        pnl['pnl_pct'] = 100 * pnl['pnl'] / collateral

        self.pnl = pnl
        return pnl

class Strat3A(StrategyBase):
    """
    Adapted from Strategy 3-A in Chua et al 2005.
    """

    def __init__(
        self,
        *args,
        sigma_thresh=0.8,
        window_size=102,
        **kw,
    ):
        super(Strat3A, self).__init__(*args, **kw)
        self.sigma_thresh = sigma_thresh
        self.window_size = window_size
        assert len(self.tenors) == 3, self.tenors
        assert self.tenors[0] < self.tenors[1] < self.tenors[2], self.tenors
        self._param_names.extend([
            'sigma_thresh', 'window_size',
        ])

    def get_hedge_factors(self, val, long_mult):
        """
        Assuming duration and cash-weighted
        positions for Strategy 3-A as described in Chua et al 2005.
        """
        # Calculate the hedge factors for each position as described in Chua et al 2005.
        hedge_factors = pd.Series({
            tenor_wk: float('nan')
            for tenor_wk in self.tenors
        })

        sh, mid, lon = self.tenors
        hedge_factors[lon] = k_div_x1(lon)
        hedge_factors[sh] = 1.
        hedge_factors[mid] = 2 * k_div_x1(mid)

        if long_mult == 1:
            hedge_factors[mid] *= -1
        elif long_mult == -1:
            hedge_factors[sh] *= -1
            hedge_factors[lon] *= -1
        else:
            raise NotImplementedError(f'Invalid {long_mult=}')

        # Borrow (deposit) remaining cash at the 4-week rate
        # to ensure cash neutrality.
        hedge_factors.loc[4] = -hedge_factors[[col for col in hedge_factors.index if col != 4]].sum()
        self.hedge_factors_raw = hedge_factors
        return self.hedge_factors_raw

    def get_signal(self) -> pd.Series:
        """
        Generate the trading signal for naive Strategy 3-A. The trading signal
        will be 1 if we should take a long position on the 3-A portfolio,
        0 if we should be flat, and -1 if we should short. A non-zero signal
        is emitted if the curvature of the 4-week forward rate curve
        is >= `sigma_thresh` (<= -`sigma_thresh` for short) standard
        deviations from the mean historical curvature, where mean and STD
        are calculated over the last `window_size` 4-week periods.
        """
        # Here, we shift the signal by one period (4 weeks)
        # to avoid lookahead bias, so that the date index represents t,
        # the time when we're selling the position that we decided on a month ago.
        # This is consistent with our date indexing for portfolio returns data:
        # that date represents the day we _closed_ our 4-week long position.
        fwd = self.zcb.stack(1).swaplevel().loc['fwd'][self.tenors].shift(1)

        # Calculate curvature (Eq. 2 in Chua et al 2005)
        sh, mid, lon = self.tenors
        fwd_curv = (
            ((fwd[mid] - fwd[sh]) / (mid - sh)) -
            ((fwd[lon] - fwd[mid]) / (lon - mid))
        )

        df = fwd_curv.rolling(window=self.window_size).agg(['mean', 'std'])
        df['low_thresh']  = df['mean'] - df['std'] * self.sigma_thresh
        df['high_thresh'] = df['mean'] + df['std'] * self.sigma_thresh
        df['fwd_curv'] = fwd_curv
        df['signal'] = 0
        df.loc[df['fwd_curv'] >= df['high_thresh'], 'signal'] = -1
        df.loc[df['fwd_curv'] <= df['low_thresh'], 'signal']  = 1
        return df

class Strat3C(StrategyBase):
    """
    Strat3A, but the signal is based on the predictive regression
    developed in 20230307_p_model.ipynb. The signal is the predicted
    PnL (with fees) from the rolling exponentially weighted linear regression on
    the curvature Z-score.
    """

    def __init__(
        self,
        *args,
        window_size=102,
        half_life=12,
        ci_alpha=0.2,
        **kw,
    ):
        super(Strat3C, self).__init__(*args, **kw)
        self.window_size = window_size
        self.half_life = half_life
        self.ci_alpha = ci_alpha
        assert len(self.tenors) == 3, self.tenors
        assert self.tenors[0] < self.tenors[1] < self.tenors[2], self.tenors
        self._param_names.extend([
            'half_life', 'window_size', 'ci_alpha',
        ])

    def get_hedge_factors(self, val, long_mult):
        """
        Same as Strat3A.

        Assuming duration and cash-weighted
        positions for Strategy 3-A as described in Chua et al 2005.
        """
        # Calculate the hedge factors for each position as described in Chua et al 2005.
        hedge_factors = pd.Series({
            tenor_wk: float('nan')
            for tenor_wk in self.tenors
        })

        sh, mid, lon = self.tenors
        hedge_factors[lon] = k_div_x1(lon)
        hedge_factors[sh] = 1.
        hedge_factors[mid] = 2 * k_div_x1(mid)

        if long_mult == 1:
            hedge_factors[mid] *= -1
        elif long_mult == -1:
            hedge_factors[sh] *= -1
            hedge_factors[lon] *= -1
        else:
            raise NotImplementedError(f'Invalid {long_mult=}')

        # Borrow (deposit) remaining cash at the 4-week rate
        # to ensure cash neutrality.
        hedge_factors.loc[4] = -hedge_factors[[col for col in hedge_factors.index if col != 4]].sum()
        self.hedge_factors_raw = hedge_factors
        return self.hedge_factors_raw

    def get_signal(self) -> pd.Series:
        """
        Generate the trading signal for Strategy 3-C. The trading signal
        will be 1 if we should take a long position on the 3-C portfolio,
        0 if we should be flat, and -1 if we should short. A non-zero signal
        is emitted if any value in the confidence interval (alpha equal to
        `ci_alpha`)of predicted PnL is greater than the trading fees.
        This prediction is generated by a univariate rolling linear regression
        with exponentially decaying weights. The endogenous variable is PnL and
        the exogenous variable is the curvature Z-score, calculated over the
        pas `window_size` 4-week periods.
        """
        # Here, we shift the signal by one period (4 weeks)
        # to avoid lookahead bias, so that the date index represents t,
        # the time when we're selling the position that we decided on a month ago.
        # This is consistent with our date indexing for portfolio returns data:
        # that date represents the day we _closed_ our 4-week long position.
        fwd = self.zcb.stack(1).swaplevel().loc['fwd'][self.tenors].shift(1)

        # Calculate curvature (Eq. 2 in Chua et al 2005)
        sh, mid, lon = self.tenors
        fwd_curv = (
            ((fwd[mid] - fwd[sh]) / (mid - sh)) -
            ((fwd[lon] - fwd[mid]) / (lon - mid))
        )

        # Calculate Z-score of curvature relative to its `window_size`-period
        # simple moving average
        curv_ma = fwd_curv.rolling(window=self.window_size).agg(['mean', 'std'])
        curv_z = (fwd_curv - curv_ma['mean']) / curv_ma['std']
        curv_z.name = 'curv_z'
        curv_ma['curv_z'] = curv_z

        # Get a regression coefficient (with CI bounds) by fitting
        # a rolling, exponentially weighted, univariate least squares
        self.ew_model = self.fit_ewls(
            exog=curv_z,
            equation="pnl ~ curv_z + 0",
            half_life=self.half_life,
        )
        coeff = self.ew_model.conf_int(alpha=self.ci_alpha)
        coeff.columns = ['ci_lower', 'ci_upper']
        coeff.loc[:, 'ci_mid'] = self.ew_model.params

        # Use the regression coefficient to predict today's PnL (without fees)
        # from last month's curvature Z-score, with confidence interval
        # at alpha equal to `self.ci_alpha`.
        df = curv_ma.merge(
            coeff, how='inner', left_index=True,
            right_index=True, suffixes=(None, '_coeff'),
        )
        pred = coeff.multiply(df['curv_z'], axis=0)
        df['pnl_no_fees_pred'] = pred['ci_mid']
        pred[['ci_lower', 'ci_mid', 'ci_upper']] = (
            pred[['ci_lower', 'ci_mid', 'ci_upper']]
            .apply(lambda row: row.sort_values(ascending=row.ci_mid >= 0, ignore_index=True), axis=1)
        )

        # Get the total fees paid for long and short portfolios.
        # They _should_ be the same.
        long_fees  = abs(self.strat_returns(is_long=True )['total_fees'])
        short_fees = abs(self.strat_returns(is_long=False)['total_fees'])
        assert long_fees == short_fees, (long_fees, short_fees)

        # Use the CI upper or CI lower (whichever is optimistic)
        # to calculate PnL with fees, depending on whether the suggested
        # position (CI midpoint) is likely to be long or short.
        df['pnl_pred'] = pred.apply(
            lambda row:
                # PnL minus fees if long
                max(row.ci_upper - long_fees, 0.) if row.ci_mid >= 0 else
                # PnL (negative) plus fees if short
                min(row.ci_upper + long_fees, 0.),
        axis=1)

        # Generate our signal: buy (sell) the portfolio if the predicted
        # PnL plus fees is positive (negative).
        df['signal'] = 0
        df.loc[df['pnl_pred'] > 0, 'signal'] = 1
        df.loc[df['pnl_pred'] < 0, 'signal'] = -1
        return df

    def pred_with_fees(self, without: pd.Series, fees: float):
        """
        Get PnL with fees from PnL `without` the added fee `fees`.
        This function sets PnL to a minimum of zero if fees are larger
        than the PnL without.
        """
        assert fees > 0, fees
        return without.apply(
            lambda x:
                max(x - fees, 0) if x >= 0 else
                min(x + fees, 0)
        )

class Strat3D(Strat3C):
    """
    Strat3C, but we don't close our position if the signal is the same
    as last month, saving on transaction fees.
    """

    def __init__(
        self,
        *args,
        **kw,
    ):
        super(Strat3D, self).__init__(*args, **kw)

    def get_pnl(self):
        """
        Overrides parent method `StrategyBase.get_pnl`.

        If we have an open position and the signal is the same, use the
        fee-free PnL, since this represents our profits when we avoid closing
        and reopening the same position.
        """
        # Reuse the parent's method
        pnl = StrategyBase.get_pnl(self)

        # `pnl` column includes the fee for buying and selling the position
        # once each. This means that we can maintain our position if the signal
        # is the same, and save the transaction fee amount by avoiding closing
        # and reopening the same position.
        pnl['same_sig'] = pnl.signal == pnl.signal.shift(1)
        pnl['pnl'] = pnl.apply(self._get_pnl_if_same_sig, axis=1)

        # We need to recalculate `pnl_pct`
        collateral = self.capital / self.leverage
        pnl['pnl_pct'] = 100 * pnl['pnl'] / collateral

        self.pnl = pnl
        return pnl


def grid_search_params(
    strat_class: type,
    file_stub: str = './data/final_proj/strat_n3A_02515',
    search_params: Dict[str, List] = None,
    **constant_params,
):
    """
    Search parameter space by grid search. `strat_class` should be a subclass
    of `BaseStrategy`. `search_params` should be a str -> list mapping of
    all possible values to try for that parameter. Extra kwargs are passed
    as `constant_params` to the strategy and are held constant.
    """
    if not search_params:
        search_params = dict()

    print(f"\n{'-' * 70}")
    print(f"Running grid search on {strat_class.__name__} using the params:")
    pprint(search_params)

    varnames = list(search_params.keys())
    for paramset_tuple in itertools.product(*search_params.values()):
        paramset = {
            varname: value for varname, value
            in zip(varnames, paramset_tuple)
        }
        paramset_str = "_".join(f"{name}{value}" for name, value in paramset.items())
        strat = strat_class(
            file_stub=f"{file_stub}_{paramset_str}",
            **constant_params,
            **paramset,
        )
        strat.get_pnl()
        strat.write_all()

def main(
    zcb_fp='./data/final_proj/uszcb.csv',
):
    zcb = read_uszcb(zcb_fp)

    # # Strategy naive 1-A 135
    # strat_n1A_135 = Strat1A(
    #     zcb,
    #     tenors=[52., 156., 260.],
    #     capital=10_000_000,
    #     leverage=5.,
    #     sigma_thresh=0.,
    #     window_size=102,
    #     file_stub='./data/final_proj/strat_n1A_135',
    # )
    # strat_n1A_135.get_pnl()
    # strat_n1A_135.write_all()

    # # Strategy naive 2-A 0510
    # strat_n2A_0510 = Strat2A(
    #     zcb,
    #     tenors=[26., 520.],
    #     capital=10_000_000,
    #     leverage=5.,
    #     sigma_thresh=0.,
    #     window_size=102,
    #     file_stub='./data/final_proj/strat_n2A_0510',
    # )
    # strat_n2A_0510.get_pnl()
    # strat_n2A_0510.write_all()

    # # Strategy naive 2-A 220
    # strat_n2A_220= Strat2A(
    #     zcb,
    #     tenors=[104., 1040.],
    #     capital=10_000_000,
    #     leverage=5.,
    #     sigma_thresh=0.,
    #     window_size=102,
    #     file_stub='./data/final_proj/strat_n2A_220',
    # )
    # strat_n2A_220.get_pnl()
    # strat_n2A_220.write_all()

    # # Strategy naive 3-A 135
    # strat_n3A_135= Strat3A(
    #     zcb,
    #     tenors=[52., 156., 260.],
    #     capital=10_000_000,
    #     leverage=5.,
    #     sigma_thresh=0.,
    #     window_size=102,
    #     file_stub='./data/final_proj/strat_n3A_135',
    # )
    # strat_n3A_135.get_pnl()
    # strat_n3A_135.write_all()

    # # Strategy naive 3-A 02515
    # strat_n3A_02515 = Strat3A(
    #     zcb,
    #     tenors=[13., 52., 260.],
    #     capital=10_000_000,
    #     leverage=5.,
    #     sigma_thresh=0.,
    #     window_size=102,
    #     file_stub='./data/final_proj/strat_n3A_02515',
    # )
    # strat_n3A_02515.get_pnl()
    # strat_n3A_02515.write_all()

    # # Grid searching n3A_02515 on parameters sigma_thresh and window_size
    # grid_search_params(
    #     strat_class=Strat3A,
    #     file_stub='./data/final_proj/strat_n3A_02515',

    #     # Grid search these parameters
    #     search_params=dict(
    #         sigma_thresh=[-0.5, 0, 0.25, 0.5, 1.],
    #         window_size=[13, 26, 52, 102]
    #     ),

    #     # These parameters are held constant
    #     zcb=zcb,
    #     tenors=[13., 52., 260.],
    #     capital=10_000_000,
    #     leverage=5.,
    # )

    # # Grid search n3A_135
    # grid_search_params(
    #     strat_class=Strat3A,
    #     file_stub='./data/final_proj/strat_n3A_135',
    #     search_params=dict(
    #         sigma_thresh=[-0.5, 0, 0.25, 0.5, 1.],
    #         window_size=[13, 26, 52, 102]
    #     ),
    #     zcb=zcb,
    #     tenors=[52., 156., 260.],
    #     capital=10_000_000,
    #     leverage=5.,
    # )

    # # Grid search 3C_135
    # grid_search_params(
    #     strat_class=Strat3C,
    #     file_stub='./data/final_proj/strat_3C_135',
    #     search_params=dict(
    #         half_life=[6, 12],
    #         window_size=[13, 26, 52, 102],
    #         ci_alpha=[0.05, 0.2, 0.4, 0.6],
    #     ),
    #     zcb=zcb,
    #     tenors=[52., 156., 260.],
    #     capital=10_000_000,
    #     leverage=5.,
    # )

    # # Grid search 3D_135
    # grid_search_params(
    #     strat_class=Strat3D,
    #     file_stub='./data/final_proj/strat_3D_135',
    #     search_params=dict(
    #         half_life=[6, 12],
    #         window_size=[13, 26, 52, 102],
    #         ci_alpha=[0.05, 0.2, 0.4, 0.6],
    #     ),
    #     zcb=zcb,
    #     tenors=[52., 156., 260.],
    #     capital=10_000_000,
    #     leverage=5.,
    # )

    # Grid searching Strat1C
    grid_search_params(
        strat_class=Strat1C,
        file_stub='./data/final_proj/strat_1C_135',
        search_params=dict(
            half_life=[6, 12],
            window_size=[13, 26, 52, 102],
            ci_alpha=[0.05, 0.2, 0.4, 0.6],
        ),
        zcb=zcb,
        tenors=[52., 156., 260.],
        capital=10_000_000,
        leverage=5.,
    )

    # Grid searching Strat1D
    grid_search_params(
        strat_class=Strat1D,
        file_stub='./data/final_proj/strat_1D_135',
        search_params=dict(
            half_life=[6, 12],
            window_size=[13, 26, 52, 102],
            ci_alpha=[0.05, 0.2, 0.4, 0.6],
        ),
        zcb=zcb,
        tenors=[52., 156., 260.],
        capital=10_000_000,
        leverage=5.,
    )


    # Grid searching Strat2C
    grid_search_params(
        strat_class=Strat2C,
        file_stub='./data/final_proj/strat_2C_220',
        search_params=dict(
            half_life=[6, 12],
            window_size=[13, 26, 52, 102],
            ci_alpha=[0.05, 0.2, 0.4, 0.6],
        ),
        zcb=zcb,
        tenors=[104., 1040.],
        capital=10_000_000,
        leverage=5.,
    )

    # Grid searching Strat2D
    grid_search_params(
        strat_class=Strat2D,
        file_stub='./data/final_proj/strat_2D_220',
        search_params=dict(
            half_life=[6, 12],
            window_size=[13, 26, 52, 102],
            ci_alpha=[0.05, 0.2, 0.4, 0.6],
        ),
        zcb=zcb,
        tenors=[104., 1040.],
        capital=10_000_000,
        leverage=5.,
    )

if __name__ == '__main__':
    main(*sys.argv[1:])