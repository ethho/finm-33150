import ast
import pandas as pd
import os
import sys

sys.path.append(os.path.realpath('src'))

from ubacktester import (
    PriceFeed, BacktestEngine, BasicStrategy, px_plot, BuyAndHold,
    NaiveQuantileStrat
)

HW3_QUANTILES_CSV = 'tests/data/hw3_quantiles.csv'
HW3_PRICES_CSV = 'tests/data/hw3_prices.csv'


def _get_quantiles():
    df = pd.read_csv(HW3_QUANTILES_CSV).set_index('date').sort_index()
    for col in df.columns:
        df[col] = df[col].apply(ast.literal_eval)
    return df

def _get_prices():
    df = pd.read_csv(HW3_PRICES_CSV).set_index('date').sort_index()
    return df
        
def run_quantile_strat(ratio, start_date, end_date) -> pd.DataFrame:
    be = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
    )
    quantiles_feed = PriceFeed.from_df(_get_quantiles().loc[start_date:end_date])
    prices_feed = PriceFeed.from_df(_get_prices().loc[start_date:end_date].reset_index())
    be.add_feed(quantiles_feed, name='quantiles')
    be.add_feed(prices_feed, name='prices')
    strat = NaiveQuantileStrat(cash_equity=1e4, ratio=ratio)
    strat.name = ratio
    be.add_strategy(strat)
    be.run()

if __name__ == '__main__':
    params = [
        ('pe', '2015-01-01', '2015-02-01'),
        # ('pe', '2015-01-01', '2015-03-01'),
        # ('pe', '2015-01-01', '2015-05-01'),
    ]
    for param_set in params:
        _ = run_quantile_strat(*param_set)