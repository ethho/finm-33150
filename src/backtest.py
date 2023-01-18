import os
from pprint import pformat as pf
import json
import hashlib
from collections import namedtuple
import logging
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict, make_dataclass
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Columns in DataFrames that, when lower-cased,
# are set as the DateIndex if they match one of these
DATE_COLS = ('date',)
# FeedID is a container that identifies a value that changes
# over time. It's a two-length tuple that contains the name
# of the data feed and the field in that data feed that contains the value.
FeedID = namedtuple('FeedID', ('name', 'field'))


def cls_name(self) -> str:
    """Get the name of `self`'s class."""
    return self.__class__.__name__


def infer_price_feed_id(symbol: str, feeds: Dict[str, Any]) -> Union[FeedID, None]:
    """
    Given a dict of `feeds`, attempts to find which feed contains price data.
    Returns a `FeedID` identifying the feed if found, None otherwise.
    """
    # TODO: improve inference
    for feed_name, feed in feeds.items():
        if feed_name == 'price':
            if hasattr(feed, symbol):
                return FeedID(feed_name, symbol)
    return None


def sha1(d: Dict, maxlen=7) -> str:
    """Wrapper for sha1 hashing. Will truncate output string at `maxlen` chars."""
    return hashlib.sha1(
        json.dumps(d, sort_keys=True).encode('utf-8')
    ).hexdigest()[:maxlen]


def pd_to_native_dtype(dtype):
    """
    Given a pandas type `dtype`, returns the corresponding Python built-in type.
    """
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
    """
    Given a list of columns `cols`, try to infer which column contains
    date or datetime data. Returns the column name if found, None otherwise.
    """
    for col_raw in cols:
        col = col_raw.lower().replace(' ', '').strip()
        if col in matches:
            return col_raw
    return None


@dataclass
class FeedBase():
    """
    Defines a Feed, which represents one or more values that
    change over time. A Feed is able to append it's own dataclass attributes
    as a dict-type record to an internal list of records (attribute `_records`).
    This internal list of records can also be pre-populated from a CSV or
    DataFrame. This class also provides data access utilities, such as
    checking that the current datetime `dt` is in bounds with pre-populated
    records. It an also export these records as a DataFrame.
    """
    dt: datetime

    @property
    def df(self) -> pd.DataFrame:
        """Wrapper method."""
        return self._get_df()

    def _get_df(self) -> pd.DataFrame:
        """
        Return the list of records as a pandas DataFrame, indexed and sorted
        by datetime (`dt`).
        """
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
        """Record `self`'s attributes as a record and append it to `_records`."""
        if not hasattr(self, '_records'):
            self._records = list()
        self._records.append(asdict(self))

    def in_df_bounds(self) -> bool:
        """
        Checks whether current datetime `dt` is within the bounds of
        the dates in the input DataFrame, if `self` contains such a DataFrame.
        Returns True if in bounds or if no input DataFrame exists. Returns
        False if current datetime `dt` is out of bounds.
        """
        if not hasattr(self, '_in_df_last_dt'):
            return True
        elif not self.dt:
            return True
        elif self.dt > self._in_df_last_dt:
            return False
        else:
            return True

    def get_prev(self) -> Dict:
        """
        Get the last record where the datetime is less than or equal to
        the current `dt`.
        """
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
        """
        Get the next record where the datetime is greater than or equal to
        the current `dt`.
        """
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
        """
        Get the first record as a dict.
        """
        as_ser = self.df.iloc[0, :]
        as_dict = as_ser.to_dict()
        assert as_dict, (f"no records exist in instance of {cls_name(self)}")
        as_dict[self.df.index.name] = as_ser.name
        return as_dict

    def set_from_prev(self):
        """
        Set `self`'s attributes to the values of the next record where the
        datetime is less than or equal to the current `dt`.
        """
        self._set_from_dict(self.get_prev())

    def set_from_next(self):
        """
        Set `self`'s attributes to the values of the next record where the
        datetime is greater than or equal to the current `dt`.
        """
        self._set_from_dict(self.get_next())

    def set_from_first(self):
        """
        Set `self`'s attributes to the values of the first record.
        """
        self._set_from_dict(self.get_first())

    def _set_from_dict(self, d: Dict):
        """Given a dict `d`, set an attribute in `self` for each item in `d`."""
        for k, v in d.items():
            if not hasattr(self, k):
                logger.warning(f"setting value of attribute {k=} which does "
                               f"not exist in instance of {cls_name(self)}")
            setattr(self, k, v)

    def _record_from_df(self, df: pd.DataFrame):
        """
        Load records from a pandas DataFrame, and sets `self`'s attributes to
        the values in the first row (chronologically) of the DataFrame.
        """
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
        """Wrapper method."""
        self._record_from_df(df.copy())

    def record_from_csv(self, fp: str, rename: Union[Dict, None] = None, **kw):
        """Wrapper method."""
        df = pd.read_csv(fp, **kw)
        if rename:
            df.rename(columns=rename, inplace=True)
        self._record_from_df(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame, *args, name=None, **kw):
        """
        Construct a Feed instance and load its records with the values of
        pandas DataFrame `df`.
        """
        if name is None and hasattr(df, 'name'):
            name = df.name
        feed = cls(*args, dt=np.datetime64(None), name=name, **kw)
        feed.record_from_df(df)
        return feed

    @classmethod
    def from_csv(cls, fp: str, *args, name=None, **kw):
        """
        Construct a Feed instance and load its records with the values from
        a CSV file at path `fp`.
        """
        if name is None:
            name = os.path.basename(os.path.splitext(fp)[0])
        feed = cls(*args, dt=np.datetime64(None), name=name, **kw)
        feed.record_from_csv(fp)
        return feed


class PlotlyPlotter:
    """
    Mixin class that provides plotting functionality using Plotly Express.
    Requires any subclass to have a `df` attribute that returns a pandas
    DataFrame, which will be plotted when `self.plot()` is invoked.
    Intended to be inherited by a `FeedBase` subclass.
    """
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
        """Wrapper method."""
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
        scale_cols: Dict[str, float] = None,
        height=600, width=800,
        labels: Dict = None,
        show: bool = True,
    ):
        """
        Plot a pandas DataFrame `df` using Plotly Express, with x-axis
        range sliders.
        `date_col` can be used to override which column in `in_df`
        is used as the x-axis DateIndex ('dt' by default).
        Columns in `exclude_cols` will not be plotted.
        If `include_cols` is non-empty, only those columns will be plotted.
        `scale_cols` is a dictionary of string column names to floats, where
        the float is the scalar that will be applied to the value of the column
        before plotting.
        Passing `show=False` will not show the figure.
        """
        df = in_df.reset_index()
        df.drop(columns=list(exclude_cols), errors='ignore', inplace=True)
        if include_cols:
            df = df.loc[:, list(include_cols) + [date_col]]
            assert not df.empty
            if len(include_cols) == 1 and isinstance(df, pd.Series):
                df = df.to_frame(name=include_cols[0])

        if not labels:
            labels = dict()

        if scale_cols:
            for k, v in scale_cols.items():
                df[k] = df[k] * v
                labels[k] = labels.get(k, k) + f" (scaled {v:0.1f}X)"

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
    """Represents a price value that changes over datetime index `dt`."""
    price: float = float('nan')
    name: str = 'price'


@dataclass
class PositionBase(FeedBase, PlotlyPlotter):
    """
    Base class that represents a single position, long or short.
    """
    # Number of shares. Immutable. Negative for short positions.
    nshares: float
    # Pointer to the `Feed` object that this position tracks.
    feed: PriceFeed = field(repr=False)
    # `feed_id.field` defines where to find the price data in `self.feed`
    feed_id: FeedID
    # Symbol of the traded asset
    symbol: str
    # Whether the position is open or not.
    is_open: int = 1
    # Current price of the tracked asset. This will continue to change
    # even after position has closed.
    price: float = None
    # Current value of the position. Zero if position is closed.
    value: float = 0.
    # Return from the position. Will not reset to zero when position is closed.
    # Will be positive for a profitable short position.
    returns: float = 0.
    # Immutable. Price of the asset when position was opened.
    price_at_open: float = field(init=False)
    # Immutable. Price of the asset when position was closed.
    price_at_close: Optional[float] = None
    # Immutable. Total value of the position when it was closed.
    value_at_close: Optional[float] = None
    # Immutable. datetime when the position was opened
    open_dt: np.datetime64 = field(init=False)
    # Number of days that the position was open.
    days_open: int = 0
    # The current datetime, see `FeedBase`.
    dt: Optional[np.datetime64] = None

    def __post_init__(self):
        """
        Extra initialization steps that are run after the dataclass
        `__init__`.
        """
        self.price_at_open = self.get_price()
        self.open_dt = self.feed.dt
        self.get_dt()
        logger.info(f"Opened position {pf(asdict(self))}")

    def cost_to_open(self) -> float:
        """
        The cost to open this position. Will always be positive even for short
        positions.
        """
        short_mult = 1. if self.is_long else -1.
        return short_mult * self.price_at_open * self.nshares

    def close_value(self) -> float:
        """
        The value of the position if it were to be closed. Zero if the position
        is already closed.
        """
        return (self.cost_to_open() + self.returns) * self.is_open

    def close(self) -> float:
        """
        Closes the position, setting `is_open` to zero.
        """
        self.price_at_close = self.get_price()
        self.value_at_close = self.close_value()
        # if self.is_long:
        #     breakpoint()
        self.is_open = 0
        logger.info(f"Closed position {pf(asdict(self))}")
        return self.value_at_close

    def get_dt(self):
        """Update the `dt` attribute."""
        self.dt = self.feed.dt
        return self.dt

    def get_price(self):
        """Update the `price` attribute."""
        self.price = getattr(self.feed, self.feed_id.field)
        return self.price

    def get_value(self):
        """Update the `value` attribute."""
        self.value = self.get_price() * self.nshares * self.is_open
        return self.value

    def get_returns(self):
        """Update the `returns` attribute if the position is open."""
        if not self.is_open:
            logger.warning(f"refusing to update returns of a closed position")
            return
        self.returns = self.get_value() - (self.price_at_open * self.nshares)
        return self.value

    def get_days_open(self) -> int:
        """Update the `days_open` attribute."""
        self.days_open = (self.get_dt() - self.open_dt).days
        return self.days_open

    @property
    def is_long(self) -> bool:
        """Returns whether this position is long or not."""
        return bool(self.nshares >= 0)

    def update(self):
        """
        Update all attributes that should be updated at every step.
        """
        self.get_dt()
        self.get_price()
        self.get_value()
        if self.is_open:
            self.get_days_open()
            self.get_returns()
            self.record()


@dataclass
class StrategyBase(FeedBase, PlotlyPlotter):
    """
    Base class that represents a trading strategy/opportunity.
    """
    # Dictionary of data feeds that are visible to this strategy.
    # Keyed by string-type name.
    feeds: Dict[str, FeedBase] = field(default_factory=dict)
    # List of positions (open or closed) that this strategy has initialized.
    positions: List[PositionBase] = field(default_factory=list)
    # Current value of all positions to date, plus cash equity.
    value: float = 0.
    # Total return of the strategy to date.
    returns: float = 0.
    # Whether the strategy is active or not.
    is_active: int = 1
    # Amount of cash equity to start with
    cash_equity: float = 10000.
    # Current datetime, see `FeedBase`
    dt: Optional[np.datetime64] = None
    # Immutable. Number of open positions
    npositions: int = 0
    # Immutable. Number of open short positions
    nshort: int = 0
    # Immutable. Number of open long positions
    nlong: int = 0

    def get_dt(self):
        """
        Update `dt` attribute, setting it to the minimal `dt` across all
        data feeds.
        """
        pos_dt = [pos.dt for pos in getattr(self, 'positions', list())]
        if pos_dt:
            self.dt = min(pos_dt)
            return self.dt
        return None

    def update(self):
        """Update all attributes that should be updated every step."""
        self.get_npositions()
        if self.is_active:
            self.get_dt()
            self.get_value()
            self.get_returns()
            self.record()

    def get_npositions(self) -> float:
        """Update attributes `npositions`, `nshort`, and `nlong`."""
        self.npositions = len(self.get_positions())
        self.nshort = len(self.short_positions())
        self.nlong = len(self.long_positions())
        return self.npositions

    def get_value(self) -> float:
        """
        Update the `value` attribute. Calculated as the sum of value of all open
        positions' values, plus cash equity.
        """
        self.value = (sum(
            pos.close_value() for pos in self.positions
        ) * self.is_active) + self.cash_equity
        return self.value

    def get_returns(self) -> float:
        """Update the `returns` attribute."""
        if not self.is_active:
            logger.warning(f"refusing to update returns of a inactive strategy")
            return float('nan')
        rets = 0.
        for pos in self.positions:
            pos.update()
            rets += pos.returns
        self.returns = rets
        return self.returns

    def _pre_step(self):
        """Runs before `pre_step` method at every step."""
        for pos in self.positions:
            pos.update()

    def pre_step(self):
        """Runs before `step` method at every step."""
        pass

    def step(self):
        """Method that runs at every step."""
        raise NotImplementedError(
            f"method 'step' is a virtual method and should be implemented "
            f"in subclass"
        )

    def post_step(self):
        """Runs after `step` method at every step."""
        pass

    def _post_step(self):
        """Runs after `post_step` method at every step."""
        self.update()

    def finish(self):
        """
        Runs after all steps in the backtest simulation are completed.
        Exits all positions by default.
        """
        self.exit_all()

    def get_positions(self, symbol: str = None) -> List[PositionBase]:
        """
        Get list of open positions. Will only return positions
        on `symbol` if it is provided.
        """
        return sorted([
            pos for pos in self.positions
            if pos.is_open and symbol in (None, pos.symbol)
        ], key=lambda x: x.get_value(), reverse=True)

    def long_positions(self, symbol: str = None) -> List[PositionBase]:
        """
        Get list of open long positions. Will only return positions
        on `symbol` if it is provided.
        """
        return [
            pos for pos in self.get_positions(symbol)
            if pos.is_long
        ]

    def short_positions(self, symbol: str = None) -> List[PositionBase]:
        """
        Get list of open short positions. Will only return positions
        on `symbol` if it is provided.
        """
        return [
            pos for pos in self.get_positions(symbol)
            if not pos.is_long
        ]

    def open_pos(self, pos: PositionBase):
        """
        Opens a new position `pos`, subtracting the cost of opening the position
        from cash equity.
        """
        og_val = self.get_value()
        self.cash_equity -= pos.cost_to_open()
        self.positions.append(pos)
        if not pos.is_long:
            # breakpoint()
            pass

    def close(self, pos: PositionBase):
        """
        Closes the position `pos`, adding the value of the position back to
        cash equity.
        """
        self.cash_equity += pos.close()

    def _get_counter_pos(self, symbol: str, going_long: bool) -> Union[PositionBase, None]:
        """
        Returns an open position opposite to a position defined by `symbol` and
        `going` long. e.g. if `symbol='AAPL'` and `going_long=True`, this function
        will return a short position on 'AAPL', if such a position is open.
        Returns None if no such position can be found.
        """
        if going_long:
            pos_sorted = self.short_positions(symbol)
        else:
            pos_sorted = self.long_positions(symbol)
        if not pos_sorted:
            return None
        return pos_sorted[-1] if going_long else pos_sorted[0]

    def _transact(
        self, symbol: str, nshares: float, feed_id: Union[FeedID, None] = None,
        close_opposite: bool = True,
    ):
        """
        Enters a position of `nshares` on asset `symbol`, using funds from
        `cash_equity`. If `close_opposite` is `True`, then this method will
        look for a position opposite to the soon-to-be-opened position, and
        close the opposite, returning the share differential for a smaller
        forward position.
        """
        # Get the data feed associated with this trade
        if feed_id is None:
            feed_id: Union[FeedID, None] = infer_price_feed_id(
                symbol=symbol, feeds=self.feeds)
        assert feed_id is not None

        # Determine if opposite position exists and close the
        # existing before opening
        counter_pos = self._get_counter_pos(
            going_long=bool(nshares >= 0),
            symbol=symbol,
        )
        if close_opposite and counter_pos:
            nshares += counter_pos.nshares
            self.close(counter_pos)

        # Create the new position
        pos = PositionBase(
            symbol=symbol,
            nshares=nshares,
            feed=self.feeds[feed_id.name],
            feed_id=feed_id,
        )
        self.open_pos(pos)

    def buy(self, *args, **kw):
        """Enters a long position. See `_transact` docs."""
        return self._transact(*args, **kw)

    def sell(self, nshares: float, *args, **kw):
        """Enters a short position with `nshares` > 0. See `_transact` docs."""
        return self._transact(*args, nshares=-nshares, **kw)

    def any_long(self) -> bool:
        """Returns True if there are any open long positions."""
        return any(self.long_positions())

    def any_short(self):
        """Returns True if there are any open short positions."""
        return any(self.short_positions())

    def exit_all(self):
        """Exit all open positions"""
        for pos in self.positions:
            self.close(pos)


class ClockBase(object):
    """
    Base class for a Clock, which manages start, end, and step
    size of the backtest based on a pandas DatetimeIndex.
    """

    def __init__(self, dti: pd.DatetimeIndex):
        self.dti = dti
        self.i = 0
        self.dt = self.dti[self.i]

    @property
    def name(self) -> Union[str, None]:
        return getattr(self.dti, 'name', None)

    def step(self):
        """
        Set attribute `dt` to the next value in the DatetimeIndex.
        """
        self.i += 1
        if self.i >= len(self.dti):
            # return None out of bounds
            return None
        self.dt = self.dti[self.i]
        return self.dt


class BacktestEngine(object):
    """Singleton that manages a backtest."""

    def __init__(
        self, start_date: str, end_date: str, step_size: str = '1D',
        normalize_to_midnight: bool = True,
    ):
        """
        Initialize a backtest starting at `start_date` and ending at
        `end_date`. The `step` method of all feeds and strategies is called
        at every interval of `step_size`.
        """
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

        self.dt = None

    @property
    def positions(self) -> List[PositionBase]:
        """
        List of all open positions in all strategies in descending order by
        current position value.
        """
        positions = list()
        for strat in getattr(self, '_strats', list()):
            positions.extend(strat.get_positions())
        return sorted(positions, lambda x: x.get_value(), reverse=True)

    def add_feed(self, feed: FeedBase, name=None):
        """
        Add a data `feed` to the backtest. This feed will be available to all
        strategies in the backtest at `self.feeds[name]`. Name is set
        from the feed's `name` attribute, if `name` is not passed to this method.
        """
        if name is None:
            name = getattr(feed, 'name', None)
        if name is None:
            name = cls_name(feed)
        if name in self._feeds:
            raise Exception(f"{cls_name(self)} already has a feed named {name}")
        self._feeds[name] = feed

    def add_strategy(self, strat: StrategyBase, name=None):
        """
        Add a strategy `strat` to the backtest. `strat.step()` will be called
        at every datetime `dt` value of the main clock.  Name is set
        from the strategy's `name` attribute, if `name` is not passed to this method.
        """
        if name is None:
            name = getattr(strat, 'name', None)
        if name is None:
            name = cls_name(strat)
        if name in self._strats:
            raise Exception(f"{cls_name(self)} already has a strategy named {name}")
        self._strats[name] = strat

    def add_clock(self, clock: ClockBase, name=None):
        """
        Add an additional Clock to the backtest. The main clock is constructed
        and added automatically when the BacktestEngine is initialized.
        """
        if name is None:
            name = getattr(clock, 'name', None)
        if name in self._clocks:
            raise Exception(f"{cls_name(self)} already has a clock named {name}")
        self._clocks[name] = clock

    def run(self):
        """
        Main event loop. Call this method to run the backtest simulation
        for all intervals of the main clock between `start_date` and `end_date`.
        """
        # Pass all feeds to all strats
        for strat in self._strats.values():
            strat.feeds.update(self._feeds)

        # Main do..while event loop
        self.step()
        while not pd.isnull(self.dt):
            self.step()

        # Run `finish` for all strategies
        for strat in self._strats.values():
            strat.finish()

    def step(self):
        """Run at every step of the simulation."""
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
        for strat in self._strats.values():
            strat._pre_step()
            if isinstance(getattr(strat, 'pre_step', None), Callable):
                strat.pre_step()
            strat.step()
            if isinstance(getattr(strat, 'post_step', None), Callable):
                strat.post_step()
            strat._post_step()


class BasicStrategy(StrategyBase):
    """
    An example of a basic strategy. Strategies should always inherit
    from StrategyBase, and have a method `step` that defines what the strategy
    should do at each timestep.

    All feeds added to the BacktestEngine are available in the `feeds` attribute.
    e.g. if we added a PriceFeed to the backtest with `name='foobar'`, then
    we can access the values of this feed using `self.feeds['foobar']`.

    This simple example buys 100 shares of AAPL when its price dips below $1.0,
    shorts 100 shares of AAPL when the price rises above $1.20, and closes any
    long positions after 30 days.
    """

    def step(self):
        aapl = self.feeds['price'].AAPL
        if aapl is None:
            return
        elif aapl < 1.0 and not self.any_long():
            self.buy(nshares=100., symbol='AAPL')
        elif aapl > 1.2 and not self.any_short():
            self.sell(nshares=100., symbol='AAPL')

        # Drop into an interactive debugging session at a particular timepoint
        # if self.dt and self.dt >= pd.to_datetime('2018-10-28'):
        #     breakpoint()

        if self.any_long():
            pos = self.long_positions()[0]
            if pos.days_open >= 30:
                self.close(pos)
