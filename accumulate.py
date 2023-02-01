import os
import pandas as pd
import sys

sys.path.append(os.path.realpath('src'))

from ubacktester import (
    PriceFeed, BacktestEngine, BasicStrategy, px_plot, BuyAndHold,
    NaiveQuantileStrat, ClockBase, AccumulationStratBase, TradesFeed, BookFeed,
)
from profiler import profiler

class AccumulateRunner:

    HW4_TRADES_CSV = 'tests/data/mini_trades_narrow_BTC-USD_2021.delim'

    def _get_trades(self):
        fp = self.HW4_TRADES_CSV
        df = pd.read_csv(fp, delim_whitespace=True)
        df.rename(columns={
            'timestamp_utc_nanoseconds': 'dt',
        }, inplace=True)
        return df

    @profiler()
    def test_accumulate_strat(self):
        # book = self._get_book()
        trades = self._get_trades()
        trades_feed = TradesFeed.from_df(trades)
        dti = pd.concat([None, trades['dt']]).sort_values().unique()
        as_dt = pd.to_datetime(dti)
        # breakpoint()
        clock = ClockBase(dti)

        be = BacktestEngine(clock=clock)
        be.add_feed(trades_feed, name='trades')
        strat1 = AccumulationStratBase(cash_equity=1e4)
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
    runner = AccumulateRunner()
    runner.test_accumulate_strat()