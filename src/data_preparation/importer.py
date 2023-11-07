"""
Importer object for importing data.

The main method is "import_data" which imports, filters,
pre-processes data for current block,
calling other private methods as needed.
"""
import math
import multiprocessing as mp
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from data_preparation.define_blocks import (get_data_chunks, get_n_rows,
                                            save_intermediate_out, save_outs)
from data_preparation.filling_in import fill_whole_days
from data_preparation.import_homes_data import import_homes_data
from utils import (data_id, empty, formatting, get_granularity,
                   initialise_dict, list_potential_paths, obtain_time)


def keep_column(dtm_no: list, start_avail: list,
                end_avail: list) -> list:
    """Mark if date is within boundaries of valid dates for given id."""
    keep = []
    for day, start, stop in zip(dtm_no, start_avail, end_avail):
        if dtm_no is None:
            keep.append(False)
        elif start is None or math.isnan(start):
            keep.append(True)
        elif type(start) in [np.float64, float, int]:
            keep.append(start <= day <= stop)
        else:
            keep.append(
                start[0] <= day <= stop[0] or start[1] <= day <= stop[1]
            )

    return keep


def tag_availability(
        n_time_steps: int,
        list_current: dict,
        current: dict,
        i_home: int
) -> dict:
    """Tag car availability based on trips' origins and destinations."""
    n_trips = len(list_current["i_start"])
    if n_trips > 0:  # there are trips during the day
        # the first trip does not depart from home,
        # i.e. the car was not at home before the first trip
        if list_current["purposeFrom"][0] != i_home:
            current["avail"][0: list_current["i_start"][0]] = np.zeros(list_current["i_start"][0])
        for step in range(n_trips - 1):  # after all trips but last one
            if list_current["purposeTo"][step] != i_home:
                # if the trip does not finish at home
                current_trip, next_trip = list_current["i_start"][step: step + 2]
                if next_trip > current_trip:
                    current["avail"][current_trip: next_trip] = np.zeros(next_trip - current_trip)
                for step in range(len(current["avail"])):
                    if current["dist"][step] > 0:
                        assert not current["avail"][step]

        if list_current["purposeTo"][n_trips - 1] != i_home:
            # after the last one, the car is back home and available
            idx = list_current["i_start"][n_trips - 1]
            current["avail"][idx: n_time_steps] = np.ones(n_time_steps - idx)
    else:
        # if there are no trips,
        # make sure the car is set as always available
        # rather than never available
        current["avail"] = np.ones(n_time_steps)

    return current


def adjust_i_start_end(prm, step, sequence, i_start, i_end):
    """Check if trip start and end indexes match duration or adjust indexes."""
    duration_mins = sequence["cum_min_end_from_day"].iloc[step] \
        - sequence["cum_min_from_day"].iloc[step]
    extra_mins = duration_mins - (i_end - i_start) * prm["step_len"]

    if extra_mins > prm["step_len"] / 2:  # one more hour closer to truth
        mins_before_start = sequence["cum_min_from_day"].iloc[step] % prm["step_len"]
        mins_after_end = prm["step_len"] - sequence["cum_min_end_from_day"].iloc[step] \
            % prm["step_len"]
        if (
                mins_before_start < mins_after_end
                or i_end == prm["n"] - 1
        ) and i_start > 0:
            i_start -= 1
        elif i_end < prm["n"] - 1:
            i_end += 1
    elif extra_mins < prm["step_len"] / 2 and i_end > i_start:
        mins_after_start \
            = prm["step_len"] - sequence["cum_min_from_day"].iloc[step] % prm["step_len"]
        mins_before_end = sequence["cum_min_end_from_day"].iloc[step] % prm["step_len"]
        if mins_after_start < mins_before_end:
            i_start += 1
        else:
            i_end -= 1

    if i_start > i_end:
        if (sequence["cum_min_from_day"].iloc[step] + sequence["duration"].iloc[step]) \
                % (24 * 60) \
                == sequence["cum_min_end_from_day"].iloc[step]:
            # i_start > i_end because it adds up to the next day
            i_end = prm["n"] - 1
        elif (
                sequence["cum_min_from_day"].iloc[step]
                + sequence["duration"].iloc[step] > 23.9 * 60
                and sequence["cum_min_end_from_day"].iloc[step] == 0
        ):
            i_end = prm["n"] - 1
            # it finishes at nearly midnight

    assert i_start <= i_end, "i_start > i_end"

    i_end = min(i_end, prm["n"] - 1)

    return i_start, i_end


def remove_incomplete_days(days: list, to_throw: list, n: int) -> list:
    """Throw away days which could not be completed."""
    for i, day in enumerate(days):
        if i not in to_throw:
            assert len(day["mins"]) == n, \
                f"day {i} should be in to throw, len {len(day['mins'])}"
    for it, _ in enumerate(to_throw):
        i = to_throw[it]
        if i == 0:
            days = days[1:]
        elif i == len(days) - 1:
            days = days[:-1]
        else:
            days = days[0: i] + days[i + 1:]
        to_throw = [tt if tt < i else tt - 1 for tt in to_throw]

    for i, day in enumerate(days):
        assert len(day["mins"]) == n, \
            f"error len(days[{i}]['mins']) = {len(day['mins'])}"

    return days


def _append_current_day(days, current_day, id_, sequence, prm, step_keys, step):
    current_day["id"] = id_
    for day_key in ["cum_day", "month"]:
        current_day[day_key] = sequence[day_key][step]
    days.append(current_day)
    assert len(current_day["mins"]) <= prm["n"], \
        f"error len(current_day['mins']) {len(current_day['mins'])}"
    assert len(set(current_day["mins"])) == len(current_day["mins"]), \
        f"mins duplicates {current_day['mins']}"
    current_day = initialise_dict(step_keys)

    return current_day, days

def get_days_clnr(
        prm: dict,
        sequences: dict,
        data_type: str,
        save_path: Path
) -> Tuple[list, Optional[List[float]], Optional[Dict[str, int]]]:
    """Split CLNR sequences into days, fill in, or throw incomplete days."""
    step_keys = [data_type, "mins", "cum_min"]
    days = []
    current_day = initialise_dict(step_keys)
    for id_, sequence in sequences.items():
        if len(sequence[data_type]) == 0:
            continue
        # go through all the data for the current id
        for step in range(len(sequence["cum_day"])):
            if step > 0 and sequence["cum_day"][step] != sequence["cum_day"][step - 1]:
                current_day, days = _append_current_day(days, current_day, id_, sequence, prm, step_keys, step)
            for step_key in step_keys:
                current_day[step_key].append(sequence[step_key][step])

        # store final day
        current_day, days = _append_current_day(days, current_day, id_, sequence, prm, step_keys, step)

    if len(sequences) == 0:
        print("len(sequences) == 0")

    print(f"before filling in days {data_type}, len(days) = {len(days)}")
    # get whole days only - fill in if only one missing
    days, to_throw, abs_error, types_replaced_eval = fill_whole_days(
        prm, days, data_type, sequences, save_path
    )
    print(f"after filling in days {data_type}, len(days) = {len(days)}")

    del sequences

    days = remove_incomplete_days(days, to_throw, prm["n"])
    print(f"after removing incomplete days {data_type}, len(days) = {len(days)}")

    return days, abs_error, types_replaced_eval


def add_day_nts(
        prm: dict,
        list_current: dict,
        current: dict,
        days: list,
        step: int,
        sequence: dict,
        sequences: dict,
        n_no_trips: dict,
        id_: int,
) -> Tuple[dict, dict, list, dict]:
    """Add travel day to days dictionary."""
    # tag availability: not available during trips
    current = tag_availability(prm["n"], list_current, current, prm["i_home"])

    # convert dist to energy consumption in kWh
    # (convert mile to km then apply kWh/10km factor)
    assert len(current["dist"]) == prm["n"], \
        f"error len(current['dist']) = {len(current['dist'])}"
    current["car"] = [
        current["dist"][i] * 1.609344 / 10
        * prm["consfactor"][int(current["triptype"][i])]
        for i in range(len(current["dist"]))
    ]
    if sum(current["car"]) == 0:
        print("sum(current['car']) = 0")
        print(f"current['dist'] = {current['dist']}")

    # enter in days
    if (
            np.max(current['car']) < prm['car']['max_power_cutoff']
            and np.sum(current['car']) < prm['car']['max_daily_energy_cutoff']
    ):
        day = {}
        for key in prm["NTS_day_keys"]:
            day[key] = current[key]
        assert len(day["car"]) == prm["n"], \
            f"error len(day['car']) = {len(day['car'])}"
        day["id"] = int(id_)
        days.append(day)

    if step == len(sequence["dist"]) - 1 and sequence["weekday"].iloc[step] != 7:
        # missing day(s) with no trips at the end of the week
        current_n_no_trips = 7 - sequence["weekday"].iloc[step]
        days, n_no_trips = add_no_trips_day(
            prm, days, current, current_n_no_trips, n_no_trips, id_
        )

    current, list_current = new_day_nts(prm["step_len"], prm["n"], step, id_, sequences)

    return current, list_current, days, n_no_trips


def add_no_trips_day(
        prm: dict,
        days: list,
        current: dict,
        current_n_no_trips: int,
        n_no_trips: dict,
        id_: int,
        cum_day: int = None,
        weekday: int = None,
) -> Tuple[List[dict], Dict[int, int]]:
    """If a day has no trip, add this day with no trips to days."""
    n_no_trips[id_] += current_n_no_trips
    for i_no_trip in range(current_n_no_trips):
        day = {
            "cum_day": current["cum_day"] + i_no_trip + 1
            if cum_day is None
            else cum_day,
            "weekday": (
                prm["date0"]
                + timedelta(days=current["cum_day"] + i_no_trip + 1)
            ).weekday()
            if weekday is None
            else weekday,
            "id": id_,
            "avail": np.ones(prm["n"]),
            "car": np.zeros(prm["n"]),
        }
        for key in ["mins", "cum_min"]:
            day[key] = np.full(prm["n"], np.nan)
        days.append(day)

    return days, n_no_trips


def new_day_nts(step_len, n_time_steps, step: int, id_: int, sequences: dict) \
        -> Tuple[dict, dict]:
    """Initialise variables for a new time slot."""
    current = {}
    for key in ["weekday", "cum_day", "home_type", "month"]:
        current[key] = sequences[id_][key].iloc[step]
    current["id"] = id_
    current["mins"] = np.arange(n_time_steps) * step_len
    current["cum_min"] \
        = current["mins"] + sequences[id_]["cum_day"].iloc[step] * 24 * 60
    for key in ["dist", "purposeFrom", "purposeTo", "triptype", "car"]:
        current[key] = np.zeros(n_time_steps)
    current["avail"] = np.ones(n_time_steps)
    list_current = initialise_dict(["i_start", "i_end", "purposeFrom", "purposeTo"])

    return current, list_current


def _add_trip_current(sequence, current, list_current, step, prm):
    i_start = int(np.floor(sequence["cum_min_from_day"].iloc[step] / prm["step_len"]))
    i_end = int(np.floor(sequence["cum_min_end_from_day"].iloc[step] / prm["step_len"]))

    # additional minutes of trips beyond what is reflected
    # by whole number of hours resulting from i_start and i_end
    i_start, i_end = adjust_i_start_end(prm, step, sequence, i_start, i_end)
    list_current["i_start"].append(i_start)
    if len(list_current['i_start']) >= 2:
        assert i_start >= list_current["i_start"][-2] - 1, \
            f"error i_start = {i_start} after {list_current['i_start'][-2]}"

    for i in range(i_start, i_end + 1):
        n_slots = i_end + 1 - i_start
        current["dist"][i] += sequence["dist"].iloc[step] / n_slots
        current["avail"][i] = 0
        # motorway if > 10 miles
        # from Crozier 2018 Numerical analysis ...
        current["triptype"][i] = (
            0 if sequence["dist"].iloc[step] > 10 else current["home_type"]
        )
        for purpose in ["purposeFrom", "purposeTo"]:
            current[purpose][i] = sequence[purpose].iloc[step]

    for purpose in ["purposeFrom", "purposeTo"]:
        list_current[purpose].append(sequence[purpose].iloc[step])

    return list_current, current


def get_days_nts(prm: dict, sequences: dict) -> list:
    """Convert NTS sequences into days of data at adequate granularity."""
    days: List[dict] = []
    n_no_trips = initialise_dict(list(sequences.keys()), "zero")
    assert len(set(list(sequences.keys()))) == len(list(sequences.keys())), \
        "there should be no id repeats"
    for id_, sequence in sequences.items():
        # new day
        current, list_current = new_day_nts(
            prm["step_len"], prm["n"], 0, id_, sequences
        )
        assert len(current["dist"]) == prm["n"], \
            f"len(current['dist']) = {len(current['dist'])}"
        for step in range(len(sequence["dist"])):
            cum_day_t = sequence["cum_day"].iloc[step]
            if step == 0 and sequence["weekday"].iloc[step] != 1:
                current_n_no_trips = sequence["weekday"].iloc[step] - 1
                for i in range(current_n_no_trips):
                    days, n_no_trips = add_no_trips_day(
                        prm,
                        days,
                        current,
                        current_n_no_trips,
                        n_no_trips,
                        id_,
                        weekday=1 + i,
                        cum_day=cum_day_t - current_n_no_trips + i,
                    )

            elif step > 0:
                assert sequence["weekday"].iloc[step] >= sequence["weekday"].iloc[step - 1], \
                    "days not in the right order"

            # if new day / finishing a day
            if step > 0 and cum_day_t != current["cum_day"]:
                if cum_day_t - current["cum_day"] > 1:
                    # missing day(s) with no trips before current day
                    current_n_no_trips \
                        = int((cum_day_t - current["cum_day"]) - 1)
                    days, n_no_trips = add_no_trips_day(
                        prm, days, current, current_n_no_trips,
                        n_no_trips, id_
                    )

                current, list_current, days, n_no_trips = add_day_nts(
                        prm, list_current, current, days, step, sequence,
                        sequences, n_no_trips, id_
                    )

            # registering trip
            list_current, current = _add_trip_current(
                sequence, current, list_current, step, prm
            )

    # add final one
    current, list_current, days, n_no_trips = add_day_nts(
        prm, list_current, current, days, step, sequence, sequences, n_no_trips, id_
    )

    return days


def normalise(
        n_time_steps: int,  # number of time intervals per day
        days: list,
        data_type: str
) -> list:
    """Normalise daily profiles and obtain scaling factors."""
    for i, day in enumerate(days):
        if day[data_type] is not None:
            if data_type == 'gen':
                day[data_type] = np.array(day[data_type])
                day[data_type][day[data_type] < 1e-2] = 0

            sum_day = sum(day[data_type])
            day[f"norm_{data_type}"] = [
                x / sum_day if sum_day > 0 else 0 for x in day[data_type]
            ]
            assert sum(day[f"norm_{data_type}"]) == 0 \
                   or abs(sum(day[f"norm_{data_type}"]) - 1) < 1e-3, \
                   f"sum(norm_{data_type} = {sum(day['norm_' + data_type])}"
            assert len(day["norm_" + data_type]) == n_time_steps, \
                f"len(days[{i}]['norm_{data_type}]) " \
                f"= {len(day['norm_' + data_type])}"
            day["factor"] = sum_day
            if np.isnan(day["factor"]):
                print(f"days[{i}]['factor'] = {day['factor']}")
        else:
            day["norm_" + data_type], day["factor"] = None, None

    return days


def filter_validity_nts(
        data: pd.DataFrame,
        home_type: str,
        start_id_nts: dict
) -> pd.DataFrame:
    """Check which NTS data to keep."""
    data["home_type"] = data["id"].map(
        lambda id_: home_type[id_]
        if id_ in start_id_nts.keys()
        else None
    )
    # only keep mode 5/7 = household car if TripStart_B01ID is > 0
    # and i have information for start and end time
    # or i can obtain it with duration and home_type in [1,2]
    data["keep"] \
        = [k and m in [5, 7] for k, m in zip(data["keep"], data["mode"])]
    data["keep"] = data.apply(
        lambda x: x.keep if x.check_time > 0 else False, axis=1
    )
    data["time_info"] = data.apply(
        lambda x:
        empty(x.start_h) + empty(x.end_h) + empty(x.duration) < 2,
        axis=1
    )
    data["keep"] = data.apply(
        lambda x: x.keep if x.time_info else False, axis=1)
    data["keep"] = data.apply(
        lambda x: x.keep if x.home_type in [1, 2] else False, axis=1
    )

    return data


def new_time_slot_clnr(
        step_len: int, step: int, sequence: dict
) -> Tuple[List[int], int, int, list]:
    """Initialise variables for a new time slot."""
    # lower bound current slot in cumulative minutes since 01/01/2010
    min_lb = sequence["cum_min"].iloc[step]
    min_lb = min_lb - min_lb % step_len  # start of the current slot
    min_ub = min_lb + step_len - 1  # up to one minute before next slot
    # days since 01/01/2010 and minutes since the start of the day
    day, month = [sequence[e].iloc[step] for e in ["cum_day", "month"]]

    # the list of data points in the current time slot
    slot_vals: list[float] = []

    return [min_lb, min_ub], day, month, slot_vals


def update_granularity(
        step_len: int,
        sequence: dict,
        data_type: str,
        granularities: list
) -> Tuple[dict, list]:
    """Check if data fits current granularity, else decrease granularity."""
    # order sequence in chronological order
    for key in sequence:
        sequence[key] \
            = [x for _, x in sorted(zip(sequence["cum_min"], sequence[key]))]

    # get granularity data
    granularity, granularities \
        = get_granularity(step_len, sequence["cum_min"], granularities)

    output: dict = initialise_dict(
        ["n", data_type, "cum_day", "cum_min", "mins", "month", "id"]
    )

    # get initial slot
    min_bounds, day, month, slot_vals \
        = new_time_slot_clnr(step_len, 0, sequence)
    n_same_min = 0
    for step, cum_min in enumerate(sequence["cum_min"]):
        if step == len(sequence["cum_min"]) - 1 or cum_min > min_bounds[1]:
            # reaching the end of the current time slot
            target_n = step_len / granularity
            if len(slot_vals) > 0:
                assert len(slot_vals) <= 10 * step_len, \
                    f"len(slot_vals) {len(slot_vals)} > 10 * step_len"

                # if there is data in the current time slot
                # info is in kWh if household loads, in kW if solar generation
                if data_type == "loads":
                    output[data_type].append(
                        sum(slot_vals) * target_n / len(slot_vals)
                    )
                elif data_type == "gen":
                    output[data_type].append(
                        np.mean(slot_vals) * step_len / 60)

                output["n"].append(len(slot_vals))
                output["id"].append(sequence["id"][step - 1])
                output["cum_min"].append(min_bounds[0])
                output["cum_day"].append(day)
                output["month"].append(month)
                output["mins"].append(min_bounds[0] % (24 * 60))

                # get new time slot boundaries
                min_bounds, day, month, slot_vals \
                    = new_time_slot_clnr(step_len, step, sequence)

        if step > 0 and cum_min == sequence["cum_min"][step - 1]:
            n_same_min += 1  # two data points for the same exact time
        else:
            n_same_min = 0

        assert n_same_min < 10, \
            f"n_same_min = {n_same_min} data_type {data_type} step {step}"
        slot_vals.append(sequence[data_type].iloc[step])
        assert len(slot_vals) <= 10 * step_len, \
            f"len(slot_vals) {len(slot_vals)} > 10 * prm['step_len']"
        assert not np.isnan(sequence[data_type].iloc[step]), \
            f"np.isnan(sequence[{data_type}][{step}])"

    # checking it is in the right order
    assert len(output["cum_min"]) == len(output["mins"]), \
        f"step_len {data_type} len cum_min != mins"
    assert len(set(output["cum_min"])) == len(output["cum_min"]), \
        f"minutes are not unique output['cum_min'] {output['cum_min']}"

    return output, granularities


def get_sequences(
        prm: dict,
        data: pd.DataFrame,
        data_type: str
) -> Tuple[Dict[int, Any], List[int]]:
    """Split data into sequences per id."""
    sequences: Dict[int, Any] = {}
    # order data in ascending id number

    if data_type == 'car':
        sequences = {id_: data[data['id'] == id_] for id_ in set(data['id'])}
        for id_ in sequences:
            sequences[id_] = sequences[id_].sort_values(by=["cum_min", "cum_min_end"])
        # granularities = None
    else:
        data['cum_min_dtm'] = pd.to_datetime(data['cum_min'], unit='m', origin=datetime(2010, 1, 1, 0, 0))
        data.set_index('cum_min_dtm', inplace=True)
        sequences = {id_: data[data['id'] == id_] for id_ in set(data['id'])}
        # sequences_columns = ['cum_min', 'mins', data_type, 'cum_day', 'month', 'n']
        # sequences = {id_: pd.DataFrame(columns=sequences_columns) for id_ in set(data['id'])}
        # granularities: List[int] = []
        for id_ in sequences:
            if prm['H'] == 24:
                sequences[id_] = sequences[id_].resample('H').agg(
                    {data_type: 'mean', 'mins': 'first', 'cum_day': 'first', 'month': 'first', 'cum_min': 'first'}
                )
            else:
                print(f"Warning: adapt resampling for H {prm['H']}")
            sequences[id_]['cum_min'] = sequences[id_]['cum_min'].apply(lambda x: x - x % 60 * 24 / prm['H'])
            sequences[id_]['mins'] = sequences[id_]['mins'].apply(lambda x: x - x % 60 * 24 / prm['H'])
            sequences[id_].dropna(subset=['cum_min'], inplace=True)
            sequences[id_].reset_index(inplace=True, drop=True)

    assert len(sequences) > 0, "len(sequences) == 0"
    granularities = None

    return sequences, granularities


def nts_formatting(date0: datetime, data: pd.DataFrame
                   ) -> pd.DataFrame:
    """Check mins and start_h are valid, compute other time entries."""
    # correct minutes > 59
    for column in ["start_min", "end_min"]:
        data[column] = data[column].map(lambda x: 59 if x > 59 else x)

    # fill in start_h if end_h + duration
    data["no_start_h"] = data["start_h"].apply(empty)
    data["no_end_h"] = data["end_h"].apply(empty)
    if sum(data["no_start_h"]) > 0:
        data["start_h"] = data.apply(
            lambda x: x.start_h
            if not empty(x.start_h) or empty(x.duration) or x.no_end_h
            else np.floor((x.end_h * 60 + x.end_min - x.duration) / 60),
            axis=1,
        )
        data["start_min"] = data.apply(
            lambda x: x.start_min
            if not empty(x.start_min) or empty(x.duration) or x.no_end_h
            else (x.end_h * 60 + x.end_min - x.duration) % 60,
            axis=1,
        )

    if sum(data["no_end_h"]) > 0:
        data["end_h"] = data.apply(
            lambda x: x.end_h
            if not x.no_end_h or empty(x.duration) or x.no_start_h
            else np.floor((x.start_h * 60 + x.start_min + x.duration) / 60),
            axis=1,
        )
        data["end_min"] = data.apply(
            lambda x: x.end_min
            if not empty(x.end_min) or empty(x.duration) or x.no_start_h
            else (x.start_h * 60 + x.start_min + x.duration) % 60,
            axis=1,
        )

    data["no_end_h"] = data["end_h"].apply(empty)
    data["no_start_h"] = data["start_h"].apply(empty)

    assert sum(data["no_start_h"]) + sum(data["no_end_h"]) == 0, \
        "time info still missing - should be ok now!"

    data["cum_min_from_day"] = data.apply(
        lambda x: x.start_h * 60 + x.start_min, axis=1
    )
    data["cum_min_end_from_day"] = data.apply(
        lambda x: x.end_h * 60 + x.end_min, axis=1
    )
    data["cum_min"] = data.apply(
        lambda x: x.cum_min_from_day + x.cum_day * 24 * 60, axis=1
    )
    data["cum_min_end"] = data.apply(
        lambda x: x.cum_min_end_from_day + x.cum_day * 24 * 60, axis=1
    )
    data["month"] = data["cum_day"].apply(
        lambda cum_day: (date0 + timedelta(cum_day)).month
    )
    data = data.sort_values(['id', 'cum_min_from_day', 'cum_min_end_from_day'])

    return data


def filter_validity(
        data: pd.DataFrame,
        data_type: str,
        test_cell: dict,
        start_end_id: List[dict],
        prm: dict
) -> pd.DataFrame:
    """Filter rows in data, only keep valid data."""
    start_id, end_id = start_end_id
    data_source = prm["data_type_source"][data_type]
    if not (data_type == 'gen' and prm['gen_CLNR']):
        data["start_avail"] = data["id"].apply(
            lambda id_: start_id[data_source][id_]
            if id_ in start_id[data_source]
            else None
        )

        data["end_avail"] = data["id"].apply(
            lambda id_: end_id[data_source][id_]
            if id_ in end_id[data_source]
            else None
        )
    elif data_type == 'gen' and prm['var_file']['gen'][-len('parquet'):] == 'parquet':
        metadata = pd.read_csv(
            'data/data_preparation_inputs/metadata.csv',
            usecols=['ss_id', 'kwp', 'operational_at']
        )
        metadata = metadata.drop(metadata[metadata.kwp > 5].index)
        print(f"len(data) {len(data)}")
        data["month"] = data['dtm'].apply(lambda x: x.month)
        for month in set(data['month']):
            if month not in prm['months']:
                data = data.drop(data[data.month == month].index)
        print(f"len(data) {len(data)} after selecting months {prm['months']}")
        ids = set(data['id'])
        for id_ in ids:
            if id_ not in metadata['ss_id'].values:
                data = data.drop(data[data.id == id_].index)
        print(f"len(data) {len(data)} in ss_id with correct kwp")
        data["keep"] = True

        if len(data) == 0:
            return None, None

    elif data_type == 'gen' and (prm['gen_uk_power_networks'] or prm['gen_pecan_street']):
        data['keep'] = True

    data = obtain_time(data, data_type, prm)
    if not (data_type == 'gen' and prm['gen_CLNR']):
        data["keep"] = keep_column(
            data["cum_day"], data["start_avail"], data["end_avail"]
        )
        if data_type == 'gen':
            data['keep'] = data.apply(
                lambda x: False if x['Measurement Description'] != 'solar power' else x.keep,
                axis=1
            )

    # set small negative values to zero and remove larger negative values
    var_label = "dist" if data_type == "car" else data_type
    data[var_label] = data[var_label].map(
        lambda x: x if x > 0 else (0 if x > -1e-2 else None)
    )

    def _remove_none(row):
        if row[var_label] is None or np.isnan(row[var_label]):
            return False

        return row["keep"]

    data["keep"] = data.apply(_remove_none, axis=1)

    if data_type == "loads":
        data["keep"] = data.apply(
            lambda x: x.keep
            and x.id in test_cell
            and data_type == prm["test_cell_translation"][test_cell[x.id]],
            axis=1,
        )

    range_dates = [data["cum_day"].min(), data["cum_day"].max()]

    return data, range_dates


def import_segment(
        prm, chunk_rows, data_type, pf_iter_batches
) -> list:
    """In parallel or sequentially, import and process block of data."""
    data_id_ = data_id(prm, data_type)
    potential_paths_outs = list_potential_paths(prm, [data_type])
    if any(
            all(
                (
                    potential_path / f"{label}_{data_id_}_{chunk_rows[0]}_{chunk_rows[1]}.pickle"
                ).is_file()
                for label in prm["outs_labels"]
            )
            for potential_path in potential_paths_outs
    ):
        # the second one is a folder with all data types
        if chunk_rows[0] == 0:
            print("load previously saved chunks of data")
        return [None] * 7

    data_source = prm["data_type_source"][data_type]

    if data_type == "gen" and prm['var_file']['gen'][-len('parquet'):] == 'parquet':
        data = pa.Table.from_batches([next(pf_iter_batches)]).to_pandas()
        data.columns = ['gen', 'dtm', 'id']
    elif data_type == "gen" and prm["gen_uk_power_networks"]:
        data = pd.read_csv(
                prm["var_path"][data_type],
                usecols=prm['i_cols_gen'],
            )
        data['gen'] = data.apply(
            lambda x: (x.P_GEN_MIN + x.P_GEN_MAX)/2,
            axis=1
        )
    elif data_type == 'gen' and prm['var_file']['gen'] == '15minute_data_austin.csv':
        data = pd.read_csv(
            prm["var_path"][data_type],
            usecols=list(prm["i_cols"][data_type].values()),
            skiprows=max(1, chunk_rows[0]),
            names=list(prm["i_cols"][data_type]),
        )
    else:
        data = pd.read_csv(
            prm["var_path"][data_type],
            usecols=list(prm["i_cols"][data_type].values()),
            skiprows=max(1, chunk_rows[0]),
            nrows=chunk_rows[1] - chunk_rows[0],
            names=list(prm["i_cols"][data_type]),
            sep=prm["separator"][data_source],
            lineterminator=prm["line_terminator"][data_source]
        )

    # 1 - get the data in initial format
    data, all_data, range_dates, n_ids = get_data(
        data_type, prm, data_source, data
    )
    if data is None:
        return [None] * 7

    # 2 - split into sequences of subsequent times
    sequences, granularities = get_sequences(prm, data, data_type)
    del data

    # 3 - convert into days of data at adequate granularity
    if data_source == "CLNR":
        # reduce to desired granularity
        days, abs_error, types_replaced_eval = get_days_clnr(
            prm, sequences, data_type, prm["save_other"]
        )

    if data_source == "NTS":
        # convert from list of trips to 24 h-profiles
        days = get_days_nts(prm, sequences)
        abs_error, types_replaced_eval = None, None

    del sequences

    days = normalise(prm["n"], days, data_type)

    assert len(days) > 0, \
        f"len(days) {len(days)}, {data_type}, ids[0] {chunk_rows[0]}"

    out = [
        days, abs_error, types_replaced_eval, all_data,
        granularities, range_dates, n_ids
    ]
    out = save_intermediate_out(prm, out, chunk_rows, data_type)

    return out


def get_data(
        data_type: str,
        prm: dict,
        data_source: str,
        data: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, list, int]:
    """Get raw data, format, filter."""
    # 1 - import homes data
    home_type, start_end_id, test_cell = import_homes_data(prm)

    data = formatting(
        data,
        prm["type_cols"][data_type],
        data_type,
        prm,
        name_col=list(prm["i_cols"][data_type]),
        hour_min=0,
    )

    # 2 - filter data
    data, range_dates = filter_validity(
        data, data_type, test_cell, start_end_id, prm
    )
    if data is None:
        return None, None, None, None

    if data_source == "NTS":
        data = filter_validity_nts(data, home_type, start_end_id[0]["NTS"])

    # only keep valid data
    data = data[data.keep]
    data = data.reset_index()

    n_ids = len(set(data["id"]))

    if data_source == "NTS":
        data = nts_formatting(prm["date0"], data)

    # order in increasing id
    data = data.sort_values(by=["id"])
    data = data.reset_index()

    # check only one test_cell
    if data_type == "loads" or data_type == 'gen' and prm['gen_CLNR']:
        data["test_cell"] = data["id"].map(lambda id_: test_cell[id_] if id_ in test_cell else None)

    # register mapping of data for producing heatmap later
    all_data = map_all_data(data_source, data, prm)

    return data, all_data, range_dates, n_ids


def map_all_data(data_source, data, prm):
    """Check number of data points per time of day and day of year."""
    if prm["do_heat_map"]:
        all_data = np.zeros((prm['n'], 366))

        if data_source == "NTS":
            data["day_of_year"] = data.apply(
                lambda x:
                x.cum_day
                - (date(x.year, 1, 1) - prm["date0"]).days
                + 1,
                axis=1,
            )
            for i in range(len(data["start_h"])):
                start_h, day_of_year = [
                    data[e].iloc[i] for e in ["start_h", "day_of_year"]
                ]
                day_of_year_ = (day_of_year - 1) % 365
                all_data[int(start_h), int(day_of_year_)] += 1
        else:
            start_h = data['cum_min'].map(lambda x: np.floor((x % (24 * 60)) / 60))
            day_of_year = data['cum_day'].map(
                lambda x: (prm["date0"] + timedelta(days=x)).timetuple().tm_yday - 1
            )
            for i in range(len(start_h)):
                all_data[int(start_h[i]), day_of_year[i]] += 1
    else:
        all_data = None

    return all_data


def get_percentiles(days, prm):
    """Get percentiles of data values."""
    percentiles = {}
    for data_type in prm["data_types"]:
        list_data = []
        assert len(days[data_type]) > 0, \
            f"len(days[data_type]) {len(days[data_type])}"
        for day in days[data_type]:
            list_data += [v for v in day[data_type] if v > 0]
        assert len(list_data) > 0, f"len(list_data) {len(list_data)}"
        percentiles[data_type] \
            = [np.percentile(list_data, i) for i in range(101)]

    with open(prm["save_other"] / "percentiles.pickle", "wb") as file:
        pickle.dump(percentiles, file)

    return percentiles


def import_data(
        prm: dict,
) -> Tuple[dict, Dict[str, int]]:
    """Import, filter, pre-process data for current block."""
    days = {}  # bank of days of data per data type
    n_data_type = {}  # len of bank per data type
    prm['gen_uk_power_networks'] = \
        prm['var_file']['gen'] == 'EXPORT HourlyData - Customer Endpoints.csv'
    prm['gen_pecan_street'] = prm['var_file']['gen'] == '15minute_data_austin.csv'
    prm['gen_CLNR'] = prm['var_file']['gen'].endswith('TrialMonitoringData.csv')

    for data_type in prm["data_types"]:
        print(f"start import {data_type}")
        if data_type == "gen" and prm['var_file']['gen'][-len('parquet'):] == 'parquet':
            n_rows_per_chunk = 1e7
            n_batches_max = 300
            pf = ParquetFile(prm["var_path"][data_type])
            pf_iter_batches = pf.iter_batches(batch_size=n_rows_per_chunk)
            try:
                n_batches = 0
                while n_batches < n_batches_max:
                    n_batches += 1
            except Exception:
                pf_iter_batches = pf.iter_batches(batch_size=n_rows_per_chunk)
        else:
            pf_iter_batches = None
            n_batches = 0

        # identifier for saving data_type-related data
        # savings paths
        n_data_type_path \
            = prm["save_other"] / f"n_dt0_{data_id(prm, data_type)}.npy"
        if prm["n_rows"][data_type] == "all":
            prm["n_rows"][data_type] = get_n_rows(data_type, prm)
        if data_type == "gen" and prm['var_file']['gen'][-len('parquet'):] == 'parquet':
            chunks_rows = [[i, n_rows_per_chunk] for i in range(n_batches)]
        elif data_type == "gen" and prm['gen_uk_power_networks']:
            chunks_rows = [[0, prm["n_rows"][data_type]]]
        else:
            # chunks_rows = get_data_chunks(prm, data_type)
            chunks_rows = [[0, 1e5]]
        if prm["parallel"]:
            pool = mp.Pool(prm["n_cpu"])
            outs = pool.starmap(
                import_segment,
                [(prm, chunk_rows, data_type, pf_iter_batches)
                 for chunk_rows in chunks_rows],
            )
            pool.close()
        else:
            outs = [
                import_segment(prm, chunk_rows, data_type, pf_iter_batches)
                for chunk_rows in tqdm(chunks_rows)
            ]

        days[data_type] = save_outs(
            outs, prm, data_type, chunks_rows
        )
        n_data_type[data_type] = len(days[data_type])

        np.save(n_data_type_path, n_data_type[data_type])

        assert len(days[data_type]) > 0, \
            f"all len(days[{data_type}]) {len(days[data_type])}"

    get_percentiles(days, prm)

    return days, n_data_type
