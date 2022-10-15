"""User-defined tool functions used in the project."""

import array
import datetime
from typing import List

import numpy as np
import pandas as pd


def initialise_dict(keys, type_obj="empty_list"):
    """Initialise empty dict, with entries as specified."""
    obj = {}
    for key in keys:
        if type_obj == "empty_list":
            obj[key] = []
        elif type_obj == "empty_dict":
            obj[key] = {}
        elif type_obj == "zero":
            obj[key] = 0
        elif type_obj == 'empty_array':
            obj[key] = array.array('f')
        elif type_obj == 'empty_np_array':
            obj[key] = np.array([])

    return obj


def str_to_cum_day(str_):
    """From datetime str, get number of days since 1/1/2010."""
    if isinstance(str_, str) and len(str_) > 9:
        date = datetime.date(int(str_[6:10]), int(str_[3:5]), int(str_[0:2]))
        date0 = datetime.date(2010, 1, 1)
        return (date - date0).days

    return None


def dmy_to_cum_day(day, month, year):
    """From day, month, year, get cumulative days since 1/1/2010."""
    return (datetime.date(year, month, day) - datetime.date(2010, 1, 1)).days


def datetime_to_cols(data_frame, col_name, hour_min=0):
    """From string information, obtain datetime numerical info."""
    dtm_cols = ["day", "mth", "yr", "hr", "min"]
    start_ind = [0, 3, 6, 11, 14]
    end_ind = [2, 5, 10, 13, 16]

    dtm_cols = [col_name + dtm_cols[i] for i in range(len(dtm_cols))]
    if not hour_min:
        dtm_cols, start_ind, end_ind \
            = [x[0:3] for x in [dtm_cols, start_ind, end_ind]]
    for i, dtm_col in enumerate(dtm_cols):
        data_frame[dtm_col] = [
            None if data == "" or data[0] == "N"
            else int(data[start_ind[i]: end_ind[i]])
            for data in data_frame[col_name]
        ]
    data_frame = data_frame.drop(columns=col_name)

    return data_frame


def formatting(data, type_cols, name_col=None, hour_min=0):
    """Format data column according to type specification."""
    for i, type_col in enumerate(type_cols):
        col = i if name_col is None else name_col[i]
        if data[col] is not None:
            if type_col == "int":
                data[col] = data[col].apply(
                    lambda x: int(x) if x != " " else None)
            elif type_col == "flt":
                data[col] = data[col].apply(
                    lambda x: float(x) if x != " " else None)
            elif type_col == "dtm":
                data = datetime_to_cols(data, col, hour_min=hour_min)

    return data


def empty(data):
    """Check if data is empty."""
    return data in ["", " ", "  ", []] or data is None or pd.isnull(data)


def obtain_time(data: pd.DataFrame, data_source: str) -> pd.DataFrame:
    """Compute time-related values in DataFrame."""
    if data_source == "CLNR":
        mins = [
            int(x[14: 16]) + int(x[11: 13]) * 60 if len(x) > 16 else None
            for x in data["dtm"]]
        cum_day = [
            str_to_cum_day(x) if len(x) > 0 else None
            for x in data["dtm"]]
        month = [int(x[3: 5]) if len(x) > 5
                 else None
                 for x in data["dtm"]]

        data["mins"] = np.array(mins)
        data["cum_day"] = np.array(cum_day)
        data["month"] = np.array(month)

        data["cum_min"] = data.apply(
            lambda x: x.mins + x.cum_day * 24 * 60
            if x.cum_day is not None
            else None,
            axis=1,
        )

    elif data_source == "NTS":
        data["cum_day"] = data.apply(
            lambda x: x.start_avail + x.weekday - 1, axis=1
        )

    return data


def granularity_fits(cum_mins: list, granularity: int) -> bool:
    """Check if a sequence has values corresponding to given granularity."""
    list_fits = [m % granularity == 0 for m in cum_mins]

    return sum(list_fits) == len(cum_mins)


def get_granularity(step_len: int,
                    cum_mins: List[float],
                    granularities: List[int]):
    """Given list of cumulative minutes, get granularity of data."""
    granularity = step_len
    count = 0
    while count < step_len and not granularity_fits(cum_mins, granularity):
        granularity -= 1
        count += 1
    if granularity not in granularities:
        granularities.append(granularity)

    return granularity, granularities


def data_id(prm, data_type):
    """Return string for identifying current data_type selection."""
    return f"{data_type}_n_rows{prm['n_rows'][data_type]}_n{prm['n']}"
