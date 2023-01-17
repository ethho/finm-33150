import pytest
import pandas as pd
import plotly.express as px
from backtest import PriceFeed, BacktestEngine, BasicStrategy


class TestPriceFeed:
    
    def test_construct(self):
        # Construct and populate with values
        ps = PriceFeed('2020-01-01', float('nan'))
        ps.dt = '2021-01-01'
        ps.price = 3.
        ps.record()
        ps.dt = '2021-01-02'
        ps.price = 4.
        ps.record()
        assert ps.get_prev()['price'] == 4.
        ps.dt = '2021-12-02'
        ps.price = 5.
        ps.set_from_prev()
        assert ps.price == 4.
        assert ps.get_prev()['price'] == 4.
        ps.record()
        df = ps.df
        
        # Can plot using plotly express
        ps.plot(show=False)

    def test_set_from_dataframe(self):
        df = px.data.stocks()
        ps = PriceFeed('2020-01-01', float('nan'))
        ps.record_from_df(df)
        out_df = ps.df
        out_df.reset_index(inplace=True)
        out_df.rename(columns=dict(dt='date'), inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        pd.testing.assert_frame_equal(df, out_df, check_dtype=False, check_index_type=False)
    
    def test_construct_from_csv(self):
        df = px.data.stocks()
        fp = '/tmp/foobarbaz.csv'
        df.to_csv(fp, index=False)
        ps = PriceFeed.from_csv(fp, price=None)
        ps.get_prev()


class TestRunStrategy:

    def test_run(self):
        be = BacktestEngine(
            start_date='2020-01-01',
            end_date='2022-12-01',
        )
        feed1 = PriceFeed.from_df(px.data.stocks())
        be.add_feed(feed1, name='prices')
        breakpoint()
        strat1 = BasicStrategy()
        be.add_strategy(strat1)
        be.run()
