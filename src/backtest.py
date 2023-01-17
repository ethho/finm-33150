import os
import json
import hashlib
from collections import namedtuple
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field, asdict, make_dataclass
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATE_COLS = ('date',)
FeedID = namedtuple('FeedID', ('name', 'field'))

def cls_name(self):
    return self.__class__.__name__


def infer_price_feed_id(symbol: str, feeds: Dict[str, Any]) -> Union[FeedID, None]:
    # TODO
    for feed_name, feed in feeds.items():
        if feed_name == 'price':
            if hasattr(feed, symbol):
                return FeedID(feed_name, symbol)
    return None


def sha1(d: Dict, maxlen=7) -> str:
    return hashlib.sha1(
        json.dumps(d, sort_keys=True).encode('utf-8')
    ).hexdigest()[:maxlen]


def pd_to_native_dtype(dtype):
    """Given a pandas type `dtype`, returns a Python built-in type."""
    if pd.api.types.is_float_dtype(dtype):
        return float
    elif pd.api.types.is_string_dtype(dtype):
        return str
    elif pd.api.types.is_object_dtype(dtype):
        return str
    elif pd.api.types.is_integer_dtype(dtype):
        return int
    elif pd.api.types.is_datetime64_dtype(dtype):
        return np.datetime64
    return None


def infer_date_col(cols: List[str], matches=DATE_COLS) -> Union[str, None]:
    for col_raw in cols:
        col = col_raw.lower().replace(' ', '').strip()
        if col in matches:
            return col_raw
    return None


@dataclass
class FeedBase():
    dt: datetime

    @property
    def df(self):
        return self._get_df()

    def _get_df(self):
        assert getattr(self, '_records', None), (
            f"'{cls_name(self)}' has no recorded data"
        )
        df = pd.DataFrame(data=self._records)
        assert 'dt' in df.columns, f"'{cls_name(self)}' has no attribute 'dt'"
        df['dt'] = pd.to_datetime(df['dt'])
        df.set_index('dt', inplace=True)
        df.sort_index(inplace=True)
        return df

    def record(self):
        if not hasattr(self, '_records'):
            self._records = list()
        self._records.append(asdict(self))

    def in_df_bounds(self) -> bool:
        if not hasattr(self, '_in_df_last_dt'):
            return True
        elif not self.dt:
            return True
        elif self.dt > self._in_df_last_dt:
            return False
        else:
            return True

    def get_prev(self) -> Dict:
        as_dict = self.df.loc[:self.dt, :].to_dict(orient='records')
        assert as_dict, (
            f"no records before {self.dt} exist in instance "
            f"of {cls_name(self)} (first is {self.df.index[0]})"
        )
        last_dict = as_dict[-1]
        if not self.in_df_bounds():
            for k in last_dict:
                last_dict[k] = None
        return last_dict

    def get_next(self) -> Dict:
        as_dict = self.df.loc[self.dt:, :].to_dict(orient='records')
        assert as_dict, (
            f"no records after {self.dt} exist in instance "
            f"of {cls_name(self)} (latest is {self.df.index[-1]})"
        )
        first_dict = as_dict[0]
        if not self.in_df_bounds():
            for k in first_dict:
                first_dict[k] = None
        return first_dict

    def get_first(self) -> Dict:
        as_ser = self.df.iloc[0, :]
        as_dict = as_ser.to_dict()
        assert as_dict, (f"no records exist in instance of {cls_name(self)}")
        as_dict[self.df.index.name] = as_ser.name
        return as_dict

    def set_from_prev(self):
        self._set_from_dict(self.get_prev())

    def set_from_next(self):
        self._set_from_dict(self.get_next())

    def set_from_first(self):
        self._set_from_dict(self.get_first())

    def _set_from_dict(self, d: Dict):
        for k, v in d.items():
            if not hasattr(self, k):
                logger.warning(f"setting value of attribute {k=} which does "
                               f"not exist in instance of {cls_name(self)}")
            setattr(self, k, v)

    def _record_from_df(self, df: pd.DataFrame):
        date_col = infer_date_col(df.columns)
        if date_col is None:
            logger.warning(f"could not find a date-like column in columns={df.columns}")
        else:
            df.rename(columns={date_col: 'dt'}, inplace=True)

        # Add fields that are not in the current dataclass
        fields_to_add = list()
        for col in df.columns:
            if col not in self.__dataclass_fields__.keys():
                dtype = pd_to_native_dtype(df.dtypes[col])
                fields_to_add.append((col, dtype, field(default=dtype())))
        if fields_to_add:
            field_names = [col[0] for col in fields_to_add]
            cls_name = f"Feed_{sha1(field_names)}"
            self.__class__ = make_dataclass(
                cls_name, fields=fields_to_add, bases=(FeedBase, PlotlyPlotter))

        # Set records from df
        if not hasattr(self, '_records'):
            self._records = list()
        as_dict: List[Dict] = df.to_dict(orient='records')
        if isinstance(as_dict, Dict):
            as_dict = [as_dict]
        self._records.extend(as_dict)
        self._in_df = df
        self._in_df_last_dt = pd.to_datetime(df['dt'].max())
        self.set_from_first()

    def record_from_df(self, df):
        self._record_from_df(df.copy())

    def record_from_csv(self, fp: str, rename: Union[Dict, None] = None, **kw):
        df = pd.read_csv(fp, **kw)
        if rename:
            df.rename(columns=rename, inplace=True)
        self._record_from_df(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame, *args, name=None, **kw):
        if name is None and hasattr(df, 'name'):
            name = df.name
        feed = cls(*args, dt=np.datetime64(None), name=name, **kw)
        feed.record_from_df(df)
        return feed

    @classmethod
    def from_csv(cls, fp: str, *args, name=None, **kw):
        if name is None:
            name = os.path.basename(os.path.splitext(fp)[0])
        feed = cls(*args, dt=np.datetime64(None), name=name, **kw)
        feed.record_from_csv(fp)
        return feed


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

    def plot(self, *args, exclude_cols=('name', ), include_cols=None, **kw):
        try:
            return self._plot(
                in_df=self.df, *args, exclude_cols=exclude_cols,
                include_cols=include_cols, **kw
            )
        except ValueError as err:
            if 'columns of different type' in str(err):
                raise Exception(
                    f"Plotly Express could not plot DataFrame with "
                    f"columns={self.df.columns.tolist()}. Try passing "
                    f"`exclude_cols`, especially for string or object dtypes."
                )

    def _plot(
        self, in_df: pd.DataFrame,
        date_col="dt",
        title=None,
        exclude_cols=('name',),
        include_cols=None,
        height=600, width=800,
        labels: Dict = None,
        show: bool = True,
    ):
        df = in_df.reset_index()
        df.drop(columns=list(exclude_cols), errors='ignore', inplace=True)
        if include_cols:
            df = df.loc[:, list(include_cols) + [date_col]]
            assert not df.empty
            if len(include_cols) == 1 and isinstance(df, pd.Series):
                df = df.to_frame(name=include_cols[0])
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
class PriceFeed(FeedBase, PlotlyPlotter):
    price: float = float('nan')
    name: str = 'price'


@dataclass
class PositionBase(FeedBase, PlotlyPlotter):
    nshares: float
    feed: PriceFeed
    feed_id: FeedID
    symbol: str
    is_open: int = 1
    price: float = None
    value: float = 0.
    returns: float = 0.
    price_at_open: float = field(init=False)
    open_dt: np.datetime64 = field(init=False)
    dt: np.datetime64 = field(init=False) # do not use

    def __post_init__(self):
        self.price_at_open = self.get_price()
        self.open_dt = self.feed.dt
        self.get_dt()

    def close(self):
        self.is_open = 0

    def get_dt(self):
        self.dt = self.feed.dt
        return self.dt

    def get_price(self):
        self.price = getattr(self.feed, self.feed_id.field)
        return self.price

    def get_value(self):
        # TODO: prevent change after close
        self.value = self.price * self.nshares * self.is_open
        return self.value

    def get_returns(self):
        if not self.is_open:
            logger.warning(f"refusing to update returns of a closed position")
            return
        self.returns = self.value - (self.price_at_open * self.nshares)
        return self.value

    @property
    def days_open(self):
        return (self.dt - self.open_dt).days

    def update(self):
        self.get_dt()
        self.get_price()
        self.get_value()
        if self.is_open:
            self.get_returns()
            self.record()


class StrategyBase(object):

    def __init__(self):
        self.feeds: Dict[str, FeedBase] = dict()
        self.positions: List[PositionBase] = list()

    def _pre_step(self):
        for pos in self.positions:
            pos.update()

    def step(self):
        raise NotImplementedError(
            f"method 'step' is a virtual method and should be implemented "
            f"in subclass"
        )

    @property
    def long_positions(self) -> List[PositionBase]:
        return [pos for pos in self.positions if pos.nshares > 0. and pos.is_open]

    @property
    def short_positions(self) -> List[PositionBase]:
        return [pos for pos in self.positions if pos.nshares < 0. and pos.is_open]

    def _transact(
        self, symbol: str, nshares: float, feed_id: Union[FeedID, None] = None
    ):
        # Get the data feed associated with this trade
        if feed_id is None:
            feed_id: Union[FeedID, None] = infer_price_feed_id(
                symbol=symbol, feeds=self.feeds)
        assert feed_id is not None

        # TODO: determine if opposite position exists and close the
        # existing before opening

        pos = PositionBase(
            symbol=symbol,
            nshares=nshares,
            feed=self.feeds[feed_id.name],
            feed_id=feed_id,
        )
        self.positions.append(pos)
        logger.info(f"Opened position {pos}")
        # breakpoint()

    def buy(self, *args, **kw):
        return self._transact(*args, **kw)

    def sell(self, nshares: float, *args, **kw):
        return self._transact(*args, nshares=-nshares, **kw)

    def is_long(self):
        return any(self.long_positions)

    def is_short(self):
        return any(self.short_positions)

    def exit_all(self):
        raise NotImplementedError()


class ClockBase(object):

    def __init__(self, dti: pd.DatetimeIndex):
        self.dti = dti
        self.i = 0
        self.dt = self.dti[self.i]

    @property
    def name(self) -> Union[str, None]:
        return getattr(self.dti, 'name', None)

    def step(self):
        self.i += 1
        if self.i >= len(self.dti):
            # return None out of bounds
            return None
        self.dt = self.dti[self.i]
        return self.dt


class BacktestEngine(object):

    def __init__(
        self, start_date: str, end_date: str, step_size: str = '1D',
        output: Dict[str, str] = None,
        normalize_to_midnight: bool = True,
    ):
        # Containers for feeds and strategies
        self._feeds: Dict[str, FeedBase] = dict()
        self._strats: Dict[str, StrategyBase] = dict()

        # Clocks
        self._clocks: Dict[str, ClockBase] = dict()

        # Define first clock
        self.add_clock(ClockBase(pd.date_range(
            start=start_date, end=end_date,
            normalize=normalize_to_midnight,
            freq=step_size,
        )), name='main')

        # TODO: define output feed

        self.dt = None

    @property
    def positions(self) -> List[PositionBase]:
        positions = list()
        for strat in getattr(self, '_strats', list()):
            positions.extend(strat.positions)
        return positions

    def add_feed(self, feed: FeedBase, name=None):
        if name is None:
            name = getattr(feed, 'name', None)
        if name is None:
            name = cls_name(feed)
        if name in self._feeds:
            raise Exception(f"{cls_name(self)} already has a feed named {name}")
        self._feeds[name] = feed

    def add_strategy(self, strat: StrategyBase, name=None):
        if name is None:
            name = getattr(strat, 'name', None)
        if name is None:
            name = cls_name(strat)
        if name in self._strats:
            raise Exception(f"{cls_name(self)} already has a strategy named {name}")
        self._strats[name] = strat

    def add_clock(self, clock: ClockBase, name=None):
        if name is None:
            name = getattr(clock, 'name', None)
        if name in self._clocks:
            raise Exception(f"{cls_name(self)} already has a clock named {name}")
        self._clocks[name] = clock

    def run(self):
        # Pass all feeds to all strats
        for strat in self._strats.values():
            strat.feeds.update(self._feeds)

        self.step()
        while not pd.isnull(self.dt):
            self.step()

    def step(self):
        # Tick main clock
        for clockn, clock in self._clocks.items():
            if clockn == 'main':
                self.dt = clock.step()
            else:
                raise NotImplementedError()
        if pd.isnull(self.dt):
            return

        # Update all feeds
        for feedn, feed in self._feeds.items():
            feed.dt = self.dt
            feed.set_from_prev()

        # Iterate over strategies
        for stratn, strat in self._strats.items():
            strat._pre_step()
            if isinstance(getattr(strat, 'pre_step', None), Callable):
                strat.pre_step()
            strat.step()
            if isinstance(getattr(strat, 'post_step', None), Callable):
                strat.post_step()

        # TODO: record stuff using FeedBase


class BasicStrategy(StrategyBase):

    def step(self):
        aapl = self.feeds['price'].AAPL
        if aapl is None:
            return
        elif aapl < 1.0 and not self.is_long():
            self.buy(nshares=1., symbol='AAPL')
        elif aapl > 1.2 and not self.is_short():
            self.sell(nshares=1., symbol='AAPL')

        if self.is_long():
            pos = self.long_positions[0]
            if pos.days_open >= 30:
                pos.close()
