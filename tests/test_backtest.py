import pytest
import pandas as pd
import plotly.express as px
from backtest import PriceSeries

class TestPriceSeries:
    
    def test_construct(self):
        # Construct and populate with values
        ps = PriceSeries('2020-01-01', float('nan'))
        ps.dt = '2021-01-01'
        ps.price = 3.
        ps.record()
        ps.dt = '2021-01-02'
        ps.price = 4.
        ps.record()
        ps.dt = '2020-12-02'
        ps.price = 5.
        ps.record()
        df = ps.df
        
        # Can plot using plotly express
        ps.plot(show=False)

    def test_set_from_dataframe(self):
        df = px.data.stocks()
        ps = PriceSeries('2020-01-01', float('nan'))
        ps.in_df_from_df(df)
        out_df = ps.df
        out_df.reset_index(inplace=True)
        out_df.rename(columns=dict(dt='date'), inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        pd.testing.assert_frame_equal(df, out_df, check_dtype=False, check_index_type=False)