import pytest
from backtest import PriceSeries

class TestPriceSeries:
    
    def test_construct(self):
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
