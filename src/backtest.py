from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field, asdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

DATE_COLS = ('date',)


def infer_date_col(cols: List[str], matches=DATE_COLS) -> Union[str, None]:
    for col_raw in cols:
        col = col_raw.lower().replace(' ', '').strip()
        if col in matches:
            return col_raw
    return None


@dataclass
class TimeSeriesBase():
    dt: datetime
    
    @property
    def df(self):
        return self._get_df()
    
    def _get_df(self):
        assert getattr(self, '_records'), (
            f"'{self.__class__.__name__}' has no recorded data"
        )
        df = pd.DataFrame(
            data=self._records
        )
        assert 'dt' in df.columns, f"'{self.__class__.__name__}' has no attribute 'dt'"
        df['dt'] = pd.to_datetime(df['dt'])
        df.set_index('dt', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def record(self):
        if not hasattr(self, '_records'):
            self._records = list()
        self._records.append(asdict(self))

    def _records_from_df(self, df):
        if not hasattr(self, '_records'):
            self._records = list()
        self._records.extend(
            df.to_dict(orient='records')
        )

    # def load(self):
    #     df = self.in_df
    #     as_dict = df.loc[self.dt, :].to_dict(orient='records')
    #     assert as_dict, f"no rows exist in 'in_df' with index dt={self.dt}"
    #     for k, v in as_dict.items():
    #         setattr(self, k, v)
    #     return self

    def _set_in_df(self, df: pd.DataFrame):
        date_col = infer_date_col(df.columns)
        if date_col is None:
            logger.warning(f"could not find a date-like column in columns={df.columns}")
        else:
            df.rename(columns={date_col: 'dt'}, inplace=True)
        self._records_from_df(df)
        self.in_df = df

    def in_df_from_df(self, df):
        self._set_in_df(df.copy())

    def in_df_from_file(self, fp: str, **kw):
        df = pd.read_csv(fp, **kw)
        self._set_in_df(df)

    def in_df_to_now(self) -> pd.DataFrame:
        return self.in_df.loc[:self.dt, :]


class PlotlyPlotter:
    PX_RANGESELECTOR = dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )

    PX_TICKFORMATSTOPS = [
        dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"),
        dict(dtickrange=[1000, 60000], value="%H:%M:%S s"),
        dict(dtickrange=[60000, 3600000], value="%H:%M m"),
        dict(dtickrange=[3600000, 86400000], value="%H:%M h"),
        dict(dtickrange=[86400000, 604800000], value="%e. %b d"),
        dict(dtickrange=[604800000, "M1"], value="%e. %b w"),
        dict(dtickrange=["M1", "M12"], value="%b '%y M"),
        dict(dtickrange=["M12", None], value="%Y Y")
    ]
    
    def __init__(self):
        # super(FooBar, self).__init__()
        pass

    def plot(self, *args, **kw):
        return self._plot(in_df=self.df, *args, **kw)

    def _plot(
        self, in_df: pd.DataFrame,
        date_col="dt",
        title=None,
        height=600, width=800,
        labels: Dict = None,
        show: bool = True,
    ):
        df = in_df.reset_index()
        fig = px.line(
            df, x=date_col, y=df.columns,
            hover_data={date_col: "|%B %d, %Y"},
            title=title,
            height=height, width=width,
            labels=labels,
        )
        fig.update_xaxes(
            tickformatstops = self.PX_TICKFORMATSTOPS,
            rangeslider_visible=True,
            rangeselector=self.PX_RANGESELECTOR,
        )
        if show:
            fig.show()
        return fig


@dataclass
class PriceSeries(TimeSeriesBase, PlotlyPlotter):
    price: float


class BacktestEngine(object):

    def __init__(self):
        # super(BacktestEngine, self).__init__()
        pass

    def next_t(self):
        pass


class Opportunity(object):

    def __init__(self):
        # super(Opportunity, self).__init__()
        pass