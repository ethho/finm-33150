import os
from pprint import pformat as pf
import json
import time
import hashlib
from collections import namedtuple, abc
import logging
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from datetime import datetime, date
from dataclasses import dataclass, field, asdict, make_dataclass
from math import sqrt, ceil
from numbers import Number
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

__all__ = [
    'BacktestEngine',
    'StrategyBase',
    'PositionBase',
    'FeedBase',
    'PlotlyPlotter'
    'FeedID',
    'px_plot',
]

# Columns in DataFrames that, when lower-cased,
# are set as the DateIndex if they match one of these
DATE_COLS = ('date', 'dt', )
# FeedID is a container that identifies a value that changes
# over time. It's a two-length tuple that contains the name
# of the data feed and the field in that data feed that contains the value.
FeedID = namedtuple('FeedID', ('name', 'field'))

# -------------------------- Helpers & Utilities -------------------------------

def downside_deviation(ser: pd.Series, rf: Union[float, pd.Series] = 0.) -> float:
    """
    Calculate the downside standard deviation of return series `ser`.
    Optionally, pass risk-free rate `rf` as a Series (defaults to zero).
    """
    if not isinstance(ser, abc.Sequence):
        return 0.
    ser_adj = ser - rf
    neg_returns = ser_adj.loc[ser_adj < 0]
    dev = sqrt(neg_returns.pow(2).sum() / len(ser_adj))
    return dev


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


def px_plot(df: pd.DataFrame, *args, **kw):
    """Plots DataFrame `df` using a PlotlyPlotter instance."""
    plotter = PlotlyPlotter()
    return plotter._plot(in_df=df, *args, **kw)


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

# ------------------------------- Base Classes ---------------------------------

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
    dt: Union[datetime, int]

    def has_records(self) -> bool:
        raise NotImplementedError('deprecated')
        return len(getattr(self, '_records', list())) > 0

    @property
    def df(self) -> pd.DataFrame:
        """Wrapper method."""
        return self._get_df()

    def _get_df(self, before_dt: bool = False) -> pd.DataFrame:
        """
        Return the list of records as a pandas DataFrame, indexed and sorted
        by datetime (`dt`).

        Note: this is an expensive operation if feed has many fields and records.
        """
        df = getattr(self, '_in_df')
        if before_dt:
            return getattr(self, '_in_df').loc[:self.dt, :]
        else:
            return getattr(self, '_in_df')

    def __getitem__(self, *args, **kw):
        return self._get_df(before_dt=True).iloc.__getitem__(*args, **kw)

    def asdict(self):
        # d = asdict(
        #     self,
        #     dict_factory=lambda x: {
        #         k: v for (k, v) in x if isinstance(v, allow_types)
        #     }
        d = {name: getattr(self, name) for name in self._get_builtin_dtype_fields()}
        return d

    def _get_builtin_dtype_fields(
        self, allow_types=(Number, bool, str, np.datetime64, datetime, date)
    ) -> List:
        if hasattr(self, '_builtin_dtype_fields'):
            return self._builtin_dtype_fields
        self._builtin_dtype_fields = [
            name for name, field in self.__dataclass_fields__.items() if
            isinstance(getattr(self, name, None), allow_types)
        ]
        return self._builtin_dtype_fields

    def record(self, allow_types=(Number, bool, str, np.datetime64, datetime, date)):
        """Record `self`'s attributes as a record and append it to `_records`."""
        d = self.asdict()
        self._append_to_in_df(d)

    def _append_to_in_df_slow(self, d: Dict):
        as_df = pd.DataFrame(
            data=d,
            columns=d.keys(),
            index=[self.dt],
        )
        self._in_df = pd.concat([getattr(self, '_in_df', None), as_df])
        return self._in_df

    def _append_to_in_df(self, d: Dict):
        for key in DATE_COLS:
            if key in d:
                del d[key]
        if not hasattr(self, '_in_df'):
            self._init_in_df(d)
        self._in_df.loc[self.dt, :] = d
        return self._in_df

    def _init_in_df(self, d: Dict):
        if hasattr(self, 'clock'):
            # Fix dtypes
            ser_dict = dict()
            dtypes = self._get_field_dtypes()
            for name, val in d.items():
                dtype = dtypes.get(name, 'object')
                if dtype is pd._libs.tslibs.timestamps.Timestamp:
                    dtype = 'datetime64[ns]'
                ser_dict[name] = pd.Series(
                    data=[val], dtype=dtype
                )
            self._in_df = pd.DataFrame(
                data=ser_dict, index=self.clock.dti,
            )
        else:
            self._in_df = self._append_to_in_df_slow(d)
        return self._in_df

    def _get_field_dtypes(self) -> Dict:
        dtypes = {
            name: type(getattr(self, name)) for name
            in self._get_builtin_dtype_fields()
        }
        # breakpoint()
        return dtypes

    def _old_record(self, allow_types=(Number, bool, str, np.datetime64, datetime, date)):
        """Record `self`'s attributes as a record and append it to `_records`."""
        if not hasattr(self, '_records'):
            self._records = list()
        d = asdict(
            self,
            dict_factory=lambda x: {
                k: v for (k, v) in x if isinstance(v, allow_types)
            }
        )
        self._records.append(d)

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
        # assert as_dict, (f"no records exist in instance of {cls_name(self)}")
        as_dict[self.df.index.name] = as_ser.name
        return as_dict

    def set_from_prev_in_df(self):
        # Get last row of _in_df
        # last_row = self._in_df[self._in_df.dt <= self.dt].iloc[-1, :].to_dict()
        last_row = self[-1, :].to_dict()
        if 'dt' in last_row:
            del last_row['dt']
        self._set_from_dict(last_row)

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

    @classmethod
    def to_datetime(cls, val: Any):
        dt = pd.to_datetime(val)
        if dt is None:
            return None
        elif getattr(cls, 'USE_NS_DT', False):
            dt = dt.astype(int)
        return dt

    def _record_from_df(self, in_df: pd.DataFrame):
        """
        Load records from a pandas DataFrame, and sets `self`'s attributes to
        the values in the first row (chronologically) of the DataFrame.
        """
        df = in_df.reset_index()
        date_col = infer_date_col(df.columns)
        if date_col is None:
            logger.warning(f"could not find a date-like column in columns={df.columns}")
        else:
            df.rename(columns={date_col: 'dt'}, inplace=True)
            df.sort_values(by='dt', inplace=True)

        # Add fields that are not in the current dataclass
        use_ns_dt = getattr(self, 'USE_NS_DT', False)
        fields_to_add = list()
        for col in df.columns:
            if col not in self.__dataclass_fields__.keys():
                dtype = pd_to_native_dtype(df.dtypes[col])
                fields_to_add.append((col, dtype, field(default=dtype())))
        if fields_to_add:
            field_names = [col[0] for col in fields_to_add]
            cls_name = f"Feed_{sha1(field_names)}"
            _class = make_dataclass(
                cls_name, fields=fields_to_add, bases=(FeedBase, PlotlyPlotter))
            if use_ns_dt:
                _class.USE_NS_DT = True
            self.__class__ = _class

        df['dt'] = self.to_datetime(df['dt'])
        self._in_df_last_dt = df['dt'].max()
        df.set_index('dt', inplace=True)
        df.sort_index(inplace=True)
        self._in_df = df
        self.set_from_first()

    def record_from_df(self, df):
        """Wrapper method."""
        self._record_from_df(df)

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
        feed = cls(*args, dt=cls.to_datetime(None), name=name, **kw)
        # feed = cls(*args, dt=np.datetime64(None), name=name, **kw)
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
        only_numeric=True,
        include_cols=None,
        scale_cols: Dict[str, float] = None,
        offset_cols: Dict[str, float] = None,
        height=600, width=800,
        labels: Optional[Dict] = None,
        names: Optional[Dict] = None,
        show: bool = False,
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
        If `only_numeric` is True, will only plot columns in the DataFrame
        with a numeric data type.
        Passing `show=False` will not show the figure.
        """
        if not in_df.index.name or in_df.index.name in ['index', ] + list(DATE_COLS):
            in_df.index.name = 'dt'
        df = in_df.reset_index()

        if only_numeric:
            non_numeric_cols = [
                col for col in df.columns
                if not pd.api.types.is_numeric_dtype(df[col])
                and col not in (date_col,)
            ]
            exclude_cols = list(exclude_cols) + non_numeric_cols

        df.drop(columns=list(exclude_cols), errors='ignore', inplace=True)
        if include_cols:
            df = df.loc[:, list(include_cols) + [date_col]]
            assert not df.empty
            if len(include_cols) == 1 and isinstance(df, pd.Series):
                df = df.to_frame(name=include_cols[0])

        if not labels:
            labels = dict()

        if not names:
            names = dict()

        if scale_cols:
            for k, v in scale_cols.items():
                df[k] = df[k] * v
                if v < 0.5:
                    vshow = f"1/{round(1/v)}"
                else:
                    vshow = f"{v:0.1f}"
                names[k] = names.get(k, k) + f" (scaled {vshow}X)"

        if offset_cols:
            for k, v in offset_cols.items():
                df[k] = df[k] + v
                names[k] = names.get(k, k) + f" (offset by {v:0.1f})"

        fig = px.line(
            df, x=date_col, y=df.columns,
            hover_data={
                date_col: "|%B %d, %Y",
            },
            title=title,
            height=height, width=width,
            labels=labels,
        )
        # fig.update_traces(
        #     hovertemplate='%{x}:<br> {y}=%{y}'
        # )
        fig.update_xaxes(
            tickformatstops = self.PX_TICKFORMATSTOPS,
            rangeslider_visible=True,
            rangeselector=self.PX_RANGESELECTOR,
        )

        # Update names of each trace using custom `names` dict
        fig.for_each_trace(
            lambda t: t.update(
                name = names.get(t.name, t.name),
                legendgroup = names.get(t.name, t.name),
                hovertemplate = t.hovertemplate.replace(t.name, names.get(t.name, t.name))
            )
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
    # Pointer to the `Feed` object that this position tracks.
    feed: PriceFeed = field(repr=False, metadata=dict(exclude_from_dict=True))
    # `feed_id.field` defines where to find the price data in `self.feed`
    feed_id: FeedID
    # Symbol of the traded asset
    symbol: str
    # Number of shares. Immutable. Negative for short positions.
    nshares: Optional[float] = float('nan')
    # Position size. Immutable. Negative for short positions.
    # Either nshares or pos_size must be passed. pos_size will be ignored
    # if both are passed
    pos_size: Optional[float] = float('nan')
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
    # Also track daily returns and daily % returns
    daily_returns: float = 0.
    daily_pct_returns: float = 0.
    # Immutable. Price of the asset when position was opened.
    price_at_open: float = field(init=False)
    # Immutable. Price of the asset when position was closed.
    price_at_close: Optional[float] = None
    # Immutable. Total value of the position when it was closed.
    value_at_close: Optional[float] = None
    # Immutable. datetime when the position was opened
    open_dt: Union[np.datetime64, int] = field(init=False)
    # Number of days that the position was open.
    days_open: int = 0
    # The current datetime, see `FeedBase`.
    dt: Union[np.datetime64, int] = field(init=False)
    # Logging level for opening and closing trades
    logging_level: int = logging.DEBUG
    # Whether to allow fractional shares when calculating nshares from pos_size
    # If False, rounds nshares to the nearest integer number of shares.
    allow_fractional: bool = False

    def __post_init__(self):
        """
        Extra initialization steps that are run after the dataclass
        `__init__`.
        """
        self.price_at_open = self.get_price()
        self.open_dt = self.get_dt()

        # Calculate nshares if only pos_size was passed
        if pd.isnull(self.nshares):
            assert not pd.isnull(self.pos_size), (
                f"must provide one of `pos_size` or `nshares`"
            )
            nshares = self.pos_size / self.price_at_open
            if not self.allow_fractional:
                nshares = round(nshares)
                if nshares == 0:
                    logger.warning(
                        f"After rounding (allow_fractional=False), number of "
                        f"shares is {nshares=}. Pass allow_fractional=True "
                        f"or provide `nshares` explicitly."
                    )
            self.nshares = nshares
        assert not pd.isnull(self.nshares)

        logger.log(self.logging_level, f"Opened position {pf(asdict(self))}")

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
        return (self.cost_to_open() + self.get_returns()) * self.is_open

    def close(self, log=False) -> float:
        """
        Closes the position, setting `is_open` to zero.
        """
        self.price_at_close = self.get_price()
        self.value_at_close = self.close_value()
        self.daily_returns = 0.
        self.daily_pct_returns = 0.
        self.is_open = 0
        if log:
            logger.log(self.logging_level, f"Closed position {pf(asdict(self))}")
        # breakpoint()
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
            self.daily_returns = 0.
            self.daily_pct_returns = 0.
            return
        self.returns = self.get_value() - (self.price_at_open * self.nshares)
        # print(f"nshares={self.nshares:0.2f} price={self.get_price():0.2f} "
        #       f"value={self.get_value():0.2f} "
        #       f"open_price={self.price_at_open:0.2f} returns={self.returns:0.2f}")
        return self.returns

    def get_daily_returns(self):
        raise NotImplementedError('deprecated due to performance issues')
        # Daily returns are today's position value minus yesterday's
        yest_value = self.get_yest_value()
        # assert isinstance(yest_value, float)
        self.daily_returns = self.get_value() - yest_value
        self.daily_pct_returns = 100 * self.daily_returns / yest_value
        return self.daily_returns

    def get_yest_value(self) -> float:
        """
        Get yesterday's value.
        Large source of latency due to call to _get_df, and since this
        is called frequently.
        """
        raise NotImplementedError('deprecated due to performance issues')
        if len(getattr(self, '_records', list())) < 2:
            return self.get_value()
        else:
            raise NotImplementedError('deprecated due to performance issues')
            return self[-2].loc['value']

    def get_days_open(self) -> int:
        """Update the `days_open` attribute."""
        diff = self.get_dt() - self.open_dt
        self.days_open = getattr(diff, 'days', -1)
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
            # self.get_daily_returns()


class ClockBase(object):
    """
    Base class for a Clock, which manages start, end, and step
    size of the backtest based on a pandas DatetimeIndex.
    """

    def __init__(self, dti: Union[pd.DatetimeIndex, pd.Index]):
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


@dataclass
class StrategyBase(FeedBase, PlotlyPlotter):
    """
    Base class that represents a trading strategy/opportunity.
    """
    # Clock used to set dt
    clock: Optional[ClockBase] = field(
        default=None,
        metadata=dict(exclude_from_dict=True)
    )
    # Dictionary of data feeds that are visible to this strategy.
    # Keyed by string-type name.
    feeds: Dict[str, FeedBase] = field(
        default_factory=dict,
        metadata=dict(exclude_from_dict=True)
    )
    # List of positions (open or closed) that this strategy has initialized.
    positions: List[PositionBase] = field(
        default_factory=list,
        metadata=dict(exclude_from_dict=True)
    )
    # Current value of all positions to date, plus cash equity.
    value: float = 0.
    # Total return of the strategy to date.
    returns: float = 0.
    # Daily returns and % returns, i.e. return since yesterday.
    daily_returns: float = 0.
    daily_pct_returns: float = 0.
    # Whether the strategy is active or not.
    is_active: int = 1
    # Amount of cash equity to start with
    cash_equity: float = 10000.
    # Current datetime, see `FeedBase`
    dt: Optional[Union[np.datetime64, int]] = None
    # Immutable. Number of open positions
    npositions: int = 0
    # Immutable. Number of open short positions
    nshort: int = 0
    # Immutable. Number of open long positions
    nlong: int = 0
    # Track the Sharpe and Sortino ratios
    sharpe: float = 0.
    sortino: float = 0.

    def _init_dt(self):
        """
        Initialize `dt` attribute, setting it to the minimal `dt` across all
        data feeds.
        """
        pos_dt = [pos.dt for pos in getattr(self, 'positions', list())]
        if pos_dt:
            self.dt = min(pos_dt)
            return self.dt
        return None

    def get_dt(self):
        if getattr(self, 'clock', None):
            self.dt = self.clock.dt
        else:
            raise Exception(f"could not set dt from clock")

    def update(self):
        """Update all attributes that should be updated every step."""
        for pos in self.get_open_positions():
            pos.update()
        self.get_npositions()
        if self.is_active:
            self.get_dt()
            self.get_value()
            self.get_returns()
            # self.get_sharpe()
            # self.get_sortino()

    def get_npositions(self) -> float:
        """Update attributes `npositions`, `nshort`, and `nlong`."""
        self.npositions = len(self.get_open_positions())
        self.nshort = len(self.short_positions())
        self.nlong = len(self.long_positions())
        return self.npositions

    def get_value(self) -> float:
        """
        Update the `value` attribute. Calculated as the sum of value of all open
        positions' values, plus cash equity.
        """
        self.value = (sum(
            pos.close_value() for pos in self.get_open_positions()
        ) * self.is_active) + self.cash_equity
        return self.value

    def get_yest_value(self) -> float:
        """Get yesterday's value"""
        raise NotImplementedError('deprecated due to performance issues')
        if len(getattr(self, '_records', list())) < 2:
            return self.get_value()
        else:
            raise NotImplementedError('deprecated due to performance issues')
            return self[-2].loc['value']

    def get_returns(self) -> float:
        """Update the `returns` attribute."""
        if not self.is_active:
            logger.warning(f"refusing to update returns of a inactive strategy")
            return float('nan')
        rets = 0.
        daily_rets = 0.
        # yest_value = self.get_yest_value()
        # assert isinstance(yest_value, float)
        for pos in self.positions:
            if pos.is_open:
                pos.update()
            rets += pos.returns
            # daily_rets += pos.daily_returns
        self.returns = rets
        # self.daily_returns = daily_rets
        # if self.daily_returns - (self.get_value() - yest_value) > 1e-4:
        #     breakpoint()
        # self.daily_pct_returns = 100 * self.daily_returns / yest_value
        return self.returns

    def get_sharpe(self, rf: Union[float, pd.Series] = 0.) -> float:
        """Update the `sharpe` attribute with the current Sharpe ratio."""
        raise NotImplementedError('deprecated due to performance issues')
        if self.has_records():
            daily_returns = self._get_df(before_dt=True).loc[:, 'daily_returns'].iloc[0]
        else:
            self.sharpe = 0.
            return self.sharpe
        if not daily_returns.std():
            self.sharpe = 0.
            return self.sharpe
        self.sharpe = (daily_returns - rf) / daily_returns.std()
        return self.sharpe

    def get_sortino(self, rf: Union[float, pd.Series] = 0.) -> float:
        """Update the `sortino` attribute with the current Sortino ratio. Deprecated due to performance issues."""
        raise NotImplementedError('deprecated due to performance issues')
        if self.has_records():
            daily_returns = self._get_df(before_dt=True).loc[:, 'daily_returns'].iloc[0]
        else:
            self.sortino = 0.
            return self.sortino
        if not downside_deviation(daily_returns):
            self.sortino = 0.
            return self.sortino
        self.sortino = (daily_returns - rf) / downside_deviation(daily_returns)
        return self.sortino

    def start(self):
        """Runs once before the first step in the simulation."""
        pass

    def _pre_step(self):
        """Runs before `pre_step` method at every step."""
        self.update()

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
        for pos in self.get_open_positions():
            pos.record()
        self.record()

    def finish(self):
        """
        Runs after all steps in the backtest simulation are completed.
        Exits all positions by default.
        """
        self.exit_all()
        assert not self.get_open_positions()

    def get_positions(self, *args, **kw):
        """Alias for `get_open_positions`."""
        return self.get_open_positions(*args, **kw)

    def get_open_positions(self, symbol: str = None) -> List[PositionBase]:
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
            pos for pos in self.get_open_positions(symbol)
            if pos.is_long
        ]

    def short_positions(self, symbol: str = None) -> List[PositionBase]:
        """
        Get list of open short positions. Will only return positions
        on `symbol` if it is provided.
        """
        return [
            pos for pos in self.get_open_positions(symbol)
            if not pos.is_long
        ]

    def open_pos(self, pos: PositionBase, fee: float = 0.):
        """
        Opens a new position `pos`, subtracting the cost of opening the position
        from cash equity.

        `fee` is added as a percentage of notional value, in basis points.
        """
        notional_value = pos.cost_to_open()
        fee_val = notional_value * fee / 1e4
        self.cash_equity -= notional_value + fee_val
        self.positions.append(pos)

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
        self, symbol: str,
        nshares: Optional[float] = None,
        pos_size: Optional[float] = None,
        feed_id: Union[FeedID, None] = None,
        close_opposite: bool = False,
        pos_cls: type = PositionBase,
        allow_fractional: bool = False,
        fee: float = 0.,
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
        assert feed_id is not None, f"could not infer price FeedID"
        if not isinstance(feed_id, FeedID):
            try:
                feed_id = FeedID(*feed_id)
            except:
                raise

        # Determine if opposite position exists and close the
        # existing before opening
        if close_opposite:
            if nshares is None:
                raise NotImplementedError()
            counter_pos = self._get_counter_pos(
                going_long=bool(nshares >= 0),
                symbol=symbol,
            )
            if counter_pos:
                nshares += counter_pos.nshares
                self.close(counter_pos)

        # Create the new position
        pos = pos_cls(
            symbol=symbol,
            nshares=nshares,
            pos_size=pos_size,
            feed=self.feeds[feed_id.name],
            feed_id=feed_id,
            allow_fractional=allow_fractional,
        )
        pos.clock = self.clock
        self.open_pos(pos, fee=fee)

    def buy(self, *args, **kw):
        """Enters a long position. See `_transact` docs."""
        return self._transact(*args, **kw)

    def sell(
        self, nshares: Optional[float] = None, pos_size: Optional[float] = None,
        *args, **kw
    ):
        """Enters a short position with `nshares` > 0. See `_transact` docs."""
        if nshares is not None and nshares > 0.:
            nshares = -nshares
        if pos_size is not None and pos_size > 0.:
            pos_size = -pos_size
        return self._transact(*args, nshares=nshares, pos_size=pos_size, **kw)

    def any_long(self) -> bool:
        """Returns True if there are any open long positions."""
        return any(self.long_positions())

    def any_short(self):
        """Returns True if there are any open short positions."""
        return any(self.short_positions())

    def exit_all(self):
        """Exit all open positions"""
        for pos in self.get_open_positions():
            self.close(pos)


class BacktestEngine(object):
    """Singleton that manages a backtest."""

    def __init__(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None,
        step_size: str = '1D',
        normalize_to_midnight: bool = True, clock: Optional[ClockBase] = None,
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
        if not clock:
            assert start_date and end_date, f"must pass 'start_date' and 'end_date'"
            clock = ClockBase(pd.date_range(
                start=start_date, end=end_date,
                normalize=normalize_to_midnight,
                freq=step_size,
            ))
        self.add_clock(clock, name='main')

        self.dt = None

    @property
    def positions(self) -> List[PositionBase]:
        """
        List of all open positions in all strategies in descending order by
        current position value.
        """
        positions = list()
        for strat in getattr(self, '_strats', list()):
            positions.extend(strat.get_open_positions())
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
        # Pass all feeds and main clock to all strats
        for strat in self._strats.values():
            strat.feeds.update(self._feeds)
            strat.clock = self._clocks['main']
            strat.start()

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

        # Iterate over strategies
        for strat in self._strats.values():
            strat._pre_step()
            if isinstance(getattr(strat, 'pre_step', None), Callable):
                strat.pre_step()
            strat.step()
            if isinstance(getattr(strat, 'post_step', None), Callable):
                strat.post_step()
            strat._post_step()

        # Update all feeds
        for feedn, feed in self._feeds.items():
            feed.dt = self.dt
            feed.set_from_prev_in_df()

# ---------------------------- Strategy Library --------------------------------

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
        feed = self.feeds['price']
        aapl = feed.AAPL

        # DEBUG
        # print(self.dt)
        # print(feed)
        # print(feed[-3:])
        # print(feed._in_df.iloc[:3])
        # breakpoint()

        if aapl is None:
            return
        elif aapl < 1.0 and not self.any_long():
            self.buy(nshares=100., symbol='AAPL')
        elif aapl > 1.2 and not self.any_short():
            self.sell(nshares=100., symbol='AAPL')

        # Drop into an interactive debugging session at a particular timepoint
        # if self.dt and self.dt >= self.to_datetime('2018-10-28'):
        #     breakpoint()

        if self.any_long():
            pos = self.long_positions()[0]
            if pos.days_open >= 30:
                self.close(pos)


@dataclass
class BuyAndHold(StrategyBase):
    """
    Base class for a buy and hold strategy. This strategy enters a long position
    on security defined by `feed_id`, which defines which feed + field to buy.

    By default, stategies close all positions at the end of the backtest
    simulation, so we don't need to explitly close the position.
    """
    symbol: str = ''
    pos_size: float = 0. # long position size
    first_dt: bool = True

    def step(self):
        if self.first_dt:
            self.buy(
                symbol=self.symbol,
                pos_size=self.pos_size,
                close_opposite=False,
                allow_fractional=True,
            )
            self.first_dt = False

@dataclass
class NaiveQuantileStrat(StrategyBase):
    """From HW3"""
    # Proportion of cash_equity
    # to place on all positions
    gross_traded_pct: float = 0.1
    # Which financial ratio to use
    ratio: str = 'roi'

    def buy_top_n(self, pos_size, ratio='roi'):
        tickers = getattr(self.feeds['quantiles'], f'{ratio}_top')
        for ticker in tickers:
            self.buy(
                symbol=ticker,
                pos_size=pos_size / len(tickers),
                feed_id=('prices', ticker),
                close_opposite=False,
                allow_fractional=True,
            )

    def sell_bot_n(self, pos_size, ratio='roi'):
        tickers = getattr(self.feeds['quantiles'], f'{ratio}_bot')
        for ticker in tickers:
            self.sell(
                symbol=ticker,
                pos_size=pos_size / len(tickers),
                feed_id=('prices', ticker),
                close_opposite=False,
                allow_fractional=True,
            )

    def start(self):
        self.starting_cash_equity = self.cash_equity

    def step(self):
        # If not Monday, skip
        if not pd.to_datetime(self.dt).day_of_week == 0:
            return

        # Close all positions
        self.exit_all()

        # Open new set of positions
        gross_traded_cash = self.gross_traded_pct * self.starting_cash_equity
        self.buy_top_n(pos_size=gross_traded_cash/2., ratio=self.ratio)
        self.sell_bot_n(pos_size=gross_traded_cash/2., ratio=self.ratio)


@dataclass
class TradesFeed(FeedBase):
    USE_NS_DT = True

    # timestamp_utc_nanoseconds: int
    name: str
    PriceMillionths: int = 0
    SizeBillionths: int = 0
    Side: int = 0
    is_qual: int = 0
    cum_volm_side: int = 0
    cum_volm_all: int = 0
    cum_volm_qual: int = 0

@dataclass
class BookFeed(FeedBase):
    USE_NS_DT = True

    # timestamp_utc_nanoseconds: int
    name: str
    Ask1PriceMillionths: int = 0
    Bid1PriceMillionths: int = 0
    Ask2PriceMillionths: int = 0
    Bid2PriceMillionths: int = 0
    Bid1SizeBillionths: int = 0
    Ask1SizeBillionths: int = 0
    Bid2SizeBillionths: int = 0
    Ask2SizeBillionths: int = 0

def downsample_to_pow(val: int, pow10: int = 6) -> int:
    n = pow10 + 1
    hi, lo = str(val)[:-n], str(val)[-n:]
    roundup = lambda x: int(ceil(x / 10.0)) * 10
    suffix = str(roundup(int(lo[:2])))[0] + (pow10 * '0')
    final = int(hi + suffix)
    assert len(str(final)) == len(str(val))
    return final

@dataclass
class AccumulationStratBase(StrategyBase):
    USE_NS_DT = True
    qualifying_reaction_time_pow10: int = 6 # 1e6 ns, or 1 ms
    side: int = 1
    i = 0

    def last_n_trades(self, n: int = 1):
        df = self.feeds['trades'][-n:]
        return df

    def last_n_qual(self, n: int = 1):
        """Get the last `n` qualifying trades."""
        # If I want to achieve a 3% participation rate overall, what should I set my target rate to?

        df = self.last_n_trades(5)
        df['dt_ds'] = (
            pd.Series(df.index, dtype=np.int64)
            .apply(downsample_to_pow, args=[self.qualifying_reaction_time_pow10])
            .values
        )

        price_func = 'min' if self.side < 0 else 'max'
        grp = df.groupby('dt_ds', group_keys=False).apply(self._mark_qualified)
        grp.index.name = 'dt'
        # breakpoint()
        return df

    def _mark_qualified(self, df):
        if len(df) == 1:
            df['is_qual'] = 1
            return df
        if self.side > 0:
            qual_price = df['PriceMillionths'].max()
        else:
            qual_price = df['PriceMillionths'].min()
        qualified_mask = df['PriceMillionths'] == qual_price
        df['is_qual'] = qualified_mask.astype(int)
        # breakpoint()
        return df

    def step(self):
        self.i += 1
        if self.i % 1000 == 0:
            last = self.last_n_qual(5)['Side'].sum()
