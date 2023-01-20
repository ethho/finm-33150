import pytest
import pandas as pd
import plotly.express as px
from ubacktester import PriceFeed, BacktestEngine, BasicStrategy, px_plot, BuyAndHold


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
        df = ps[-2:]
        assert len(df) == 2

        # Can plot using plotly express
        ps.plot(show=False)

    @pytest.mark.skip
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

    def test_run_basic_strategy(self):
        be = BacktestEngine(
            start_date='2018-01-01',
            end_date='2019-12-01',
        )
        feed1 = PriceFeed.from_df(px.data.stocks())
        be.add_feed(feed1, name='price')
        assert len(be._feeds) == 1
        strat1 = BasicStrategy(cash_equity=200.)
        be.add_strategy(strat1)
        be.run()

        feed1.plot(show=False)
        strat1.positions[0].plot(show=False, include_cols=['price', 'returns', 'is_open'])

        strat1.plot(
            show=False,
            include_cols=['value', 'returns', 'nshort', 'nlong'],
            scale_cols={'nshort': 40, 'nlong': 40}
        )

        # Merge with price data and plot
        df = strat1.df.merge(
            feed1.df, how='outer', left_index=True, right_index=True,
        ).ffill()
        px_plot(
            df,
            show=True,
            include_cols=['value', 'returns', 'nshort', 'nlong', 'AAPL'],
            scale_cols={'nshort': 40, 'nlong': 40, 'AAPL': 100.}
        )

    def test_run_buy_and_hold(self):
        be = BacktestEngine(
            start_date='2018-01-01',
            end_date='2019-12-01',
        )
        feed1 = PriceFeed.from_df(px.data.stocks())
        be.add_feed(feed1, name='price')
        assert len(be._feeds) == 1
        strat1 = BuyAndHold(cash_equity=1e4, symbol='AAPL', pos_size=100.)
        be.add_strategy(strat1)
        be.run()

        strat1.plot(
            show=False,
            include_cols=['daily_pct_returns'],
            # scale_cols={'nshort': 40, 'nlong': 40}
        )