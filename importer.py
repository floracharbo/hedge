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

import numpy as np
import pandas as pd

from define_blocks import (get_data_chunks, get_n_rows, save_intermediate_out,
                           save_outs)
from filling_in import fill_whole_days
from import_homes_data import import_homes_data
from utils import (data_id, empty, formatting, get_granularity,
                   initialise_dict, obtain_time)


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
    """Tag EV availability based on trips' origins and destinations."""
    n_trips = len(list_current["i_start"])
    if n_trips > 0:  # there are trips during the day
        # the first trip does not depart from home,
        # i.e. the car was not at home before the first trip
        if list_current["purposeFrom"][0] != i_home:
            current["avail"][0: list_current["i_start"][0]] \
                = [0] * list_current["i_start"][0]
        for t in range(n_trips - 1):  # after all trips but last one
            if list_current["purposeTo"][t] != i_home:
                # if the trip does not finish at home
                current_trip, next_trip = [list_current["i_start"][idx]
                                           for idx in [t, t + 1]]
                current["avail"][current_trip: next_trip] = [0] * int(
                    next_trip - current_trip
                )

        if list_current["purposeTo"][n_trips - 1] != i_home:
            # after the last one
            idx = list_current["i_start"][n_trips - 1]
            current["avail"][idx: n_time_steps] = [0] * (n_time_steps - idx)
    else:
        # if there are no trips,
        # make sure the EV is set as always available
        # rather than never available
        current["avail"] = [1 for _ in range(n_time_steps)]

    return current


def adjust_i_start_end(prm, t, sequence, i_start, i_end):
    """Check if trip start and end indexes match duration or adjust indexes."""
    duration_mins = sequence["cum_min_all_end"][t] \
        - sequence["cum_min_all"][t]
    extra_mins = duration_mins - (i_end - i_start) * prm["step_len"]

    if extra_mins > prm["step_len"] / 2:  # one more hour closer to truth
        mins_before_start = sequence["cum_min_all"][t] % prm["step_len"]
        mins_after_end = prm["step_len"] - sequence["cum_min_all_end"][t] \
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
            = prm["step_len"] - sequence["cum_min_all"][t] % prm["step_len"]
        mins_before_end = sequence["cum_min_all_end"][t] % prm["step_len"]
        if mins_after_start < mins_before_end:
            i_start += 1
        else:
            i_end -= 1

    if i_start > i_end:
        if (sequence["cum_min_all"][t] + sequence["duration"][t]) \
                % (24 * 60) \
                == sequence["cum_min_all_end"][t]:
            # i_start > i_end because it adds up to the next day
            i_end = prm["n"] - 1
        elif (
                sequence["cum_min_all"][t]
                + sequence["duration"][t] > 23.9 * 60
                and sequence["cum_min_all_end"][t] == 0
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


def get_days_clnr(
        prm: dict,
        sequences: dict,
        data_type: str,
        save_path: Path
) -> Tuple[list, Optional[List[float]], Optional[Dict[str, int]]]:
    """Split CLNR sequences into days, fill in, or throw incomplete days."""
    step_keys = ["n", data_type, "mins", "cum_min"]
    days = []
    current_day = initialise_dict(step_keys)
    for id_, sequence in sequences.items():
        if len(sequence[data_type]) == 0:
            continue
        # go through all the data for the current id
        for t in range(len(sequence["cum_day"])):
            if t > 0 and sequence["cum_day"][t] != sequence["cum_day"][t - 1]:
                # if starting a new day
                current_day["id"] = int(id_)
                days.append(current_day)
                assert len(current_day["mins"]) <= prm["n"], \
                    "error len(current_day['mins']) " \
                    f"{len(current_day['mins'])}"
                current_day = initialise_dict(step_keys)

            for step_key in step_keys:
                current_day[step_key].append(sequence[step_key][t])
            for day_key in ["cum_day", "month", "id"]:
                current_day[day_key] = sequence[day_key][t]

        # store final day
        days.append(current_day)
        assert len(set(current_day["mins"])) == len(current_day["mins"]), \
            f"mins duplicates {current_day['mins']}"

        current_day = initialise_dict(step_keys)

    if len(sequences) == 0:
        print("len(sequences) == 0")

    # get whole days only - fill in if only one missing
    days, to_throw, abs_error, types_replaced_eval = fill_whole_days(
        prm, days, data_type, sequences, save_path
    )

    del sequences
    days = remove_incomplete_days(days, to_throw, prm["n"])

    return days, abs_error, types_replaced_eval


def add_day_nts(
        prm: dict,
        list_current: dict,
        current: dict,
        days: list,
        t: int,
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
    current["EV"] = [
        current["dist"][i] * 1.609344 / 10
        * prm["consfactor"][int(current["triptype"][i])]
        for i in range(len(current["dist"]))
    ]
    if sum(current["EV"]) == 0:
        print("sum(current['EV']) = 0")
        print(f"current['dist'] = {current['dist']}")

    # enter in days
    day = {}
    for key in prm["NTS_day_keys"]:
        day[key] = current[key]
    assert len(day["EV"]) == prm["n"], \
        f"error len(day['EV']) = {len(day['EV'])}"
    day["id"] = int(id_)
    days.append(day)

    if t == len(sequence["dist"]) - 1 and sequence["weekday"][t] != 7:
        # missing day(s) with no trips at the end of the week
        current_n_no_trips = 7 - sequence["weekday"][t]
        days, n_no_trips = add_no_trips_day(
            prm, days, current, current_n_no_trips, n_no_trips, id_
        )

    current, list_current \
        = new_day_nts(prm["step_len"], prm["n"], t, id_, sequences)

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
            "avail": [1 for _ in range(prm["n"])],
            "EV": [0 for _ in range(prm["n"])],
        }
        for key in ["mins", "cum_min"]:
            day[key] = [None for _ in range(prm["n"])]
        days.append(day)

    return days, n_no_trips


def new_day_nts(step_len, n_time_steps, t: int, id_: int, sequences: dict) \
        -> Tuple[dict, dict]:
    """Initialise variables for a new time slot."""
    current = {}
    for key in ["weekday", "cum_day", "home_type", "month"]:
        current[key] = sequences[id_][key][t]
    current["id"] = id_
    current["mins"] = np.arange(n_time_steps) * step_len
    current["cum_min"] \
        = current["mins"] + sequences[id_]["cum_day"][t] * 24 * 60
    for key in ["dist", "purposeFrom", "purposeTo", "triptype", "EV"]:
        current[key] = np.zeros(n_time_steps)
    current["avail"] = np.ones(n_time_steps)
    list_current = initialise_dict(
        ["i_start", "i_end", "purposeFrom", "purposeTo"])

    return current, list_current


def _add_trip_current(sequence, current, list_current, t, prm):
    i_start = int(np.floor(sequence["cum_min_all"][t] / prm["step_len"]))
    i_end = int(np.floor(sequence["cum_min_all_end"][t] / prm["step_len"]))

    # additional minutes of trips beyond what is reflected
    # by whole number of hours resulting from i_start and i_end
    i_start, i_end = adjust_i_start_end(
        prm, t, sequence, i_start, i_end)
    list_current["i_start"].append(i_start)

    for i in range(i_start, i_end + 1):
        n_slots = i_end + 1 - i_start
        current["dist"][i] += sequence["dist"][t] / n_slots
        current["avail"][i] = 0
        # motorway if > 10 miles
        # from Crozier 2018 Numerical analysis ...
        current["triptype"][i] = (
            0 if sequence["dist"][t] > 10 else current["home_type"]
        )
        for purpose in ["purposeFrom", "purposeTo"]:
            current[purpose][i] = sequence[purpose][t]

    for purpose in ["purposeFrom", "purposeTo"]:
        list_current[purpose].append(sequence[purpose][t])

    return list_current, current


def get_days_nts(prm: dict, sequences: dict) -> list:
    """Convert NTS sequences into days of data at adequate granularity."""
    days: List[dict] = []
    n_no_trips = initialise_dict(list(sequences.keys()), "zero")
    assert len(set(list(sequences.keys()))) == len(list(sequences.keys())), \
        "there should be no id repeats"
    for id_, sequence in sequences.items():
        i_sort = np.argsort(sequence["cum_min"])
        for key in sequence.keys():
            sequence[key] = [sequence[key][i] for i in i_sort]
        # new day
        current, list_current = new_day_nts(
            prm["step_len"], prm["n"], 0, id_, sequences)
        assert len(current["dist"]) == prm["n"], \
            f"len(current['dist']) = {len(current['dist'])}"
        for t in range(len(sequence["dist"])):
            cum_day_t = sequence["cum_day"][t]
            if t == 0 and sequence["weekday"][t] != 1:
                current_n_no_trips = sequence["weekday"][t] - 1
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

            elif t > 0:
                assert sequence["weekday"][t] >= sequence["weekday"][t - 1], \
                    "days not in the right order"

            # if new day / finishing a day
            if t > 0 and cum_day_t != current["cum_day"]:
                if cum_day_t - current["cum_day"] > 1:
                    # missing day(s) with no trips before current day
                    current_n_no_trips \
                        = int((cum_day_t - current["cum_day"]) - 1)
                    days, n_no_trips = add_no_trips_day(
                        prm, days, current, current_n_no_trips,
                        n_no_trips, id_
                    )

                current, list_current, days, n_no_trips \
                    = add_day_nts(
                        prm, list_current, current, days, t, sequence,
                        sequences, n_no_trips, id_)

            # registering trip
            list_current, current \
                = _add_trip_current(sequence, current, list_current, t, prm)

    # add final one
    current, list_current, days, n_no_trips = add_day_nts(
        prm, list_current, current, days, t, sequence,
        sequences, n_no_trips, id_)

    return days


def normalise(
        n_time_steps: int,  # number of time intervals per day
        days: list,
        data_type: str
) -> list:
    """Normalise daily profiles and obtain scaling factors."""
    for i, day in enumerate(days):
        if day[data_type] is not None:
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
        step_len: int, t: int, sequence: dict
) -> Tuple[List[int], int, int, list]:
    """Initialise variables for a new time slot."""
    # lower bound current slot in cumulative minutes since 01/01/2010
    min_lb = sequence["cum_min"][t]
    min_lb = min_lb - min_lb % step_len  # start of the current slot
    min_ub = min_lb + step_len - 1  # up to one minute before next slot
    # days since 01/01/2010 and minutes since the start of the day
    day, month \
        = [sequence[e][t] for e in ["cum_day", "month"]]

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
    for t, cum_min in enumerate(sequence["cum_min"]):
        if t == len(sequence["cum_min"]) - 1 or cum_min > min_bounds[1]:
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
                output["id"].append(sequence["id"][t - 1])
                output["cum_min"].append(min_bounds[0])
                output["cum_day"].append(day)
                output["month"].append(month)
                output["mins"].append(min_bounds[0] % (24 * 60))

                # get new time slot boundaries
                min_bounds, day, month, slot_vals \
                    = new_time_slot_clnr(step_len, t, sequence)

        if t > 0 and cum_min == sequence["cum_min"][t - 1]:
            n_same_min += 1  # two data points for the same exact time
        else:
            n_same_min = 0

        assert n_same_min < 10, \
            f"n_same_min = {n_same_min} data_type {data_type} t {t}"
        slot_vals.append(sequence[data_type][t])
        assert len(slot_vals) <= 10 * step_len, \
            f"len(slot_vals) {len(slot_vals)} > 10 * prm['step_len']"
        assert not np.isnan(sequence[data_type][t]), \
            f"np.isnan(sequence[{data_type}][{t}])"

    # checking it is in the right order
    assert len(output["cum_min"]) == len(output["mins"]), \
        f"step_len {data_type} len cum_min != mins"
    assert len(set(output["cum_min"])) == len(output["cum_min"]), \
        f"minutes are not unique output['cum_min'] {output['cum_min']}"

    return output, granularities


def append_id_sequences(
        prm: dict,
        data_type: str,
        current_sequence: dict,
        granularities: list
) -> Tuple[dict, dict]:
    """Add current sequence for given id to sequences dictionary."""
    # reduce granularity & add to sequences
    if data_type == "EV":
        sequences_id = current_sequence
    else:
        sequences_id, granularities = update_granularity(
            prm["step_len"], current_sequence, data_type, granularities
        )

    current_sequence = initialise_dict(prm["sequence_entries"][data_type])

    return current_sequence, sequences_id


def get_sequences(
        prm: dict,
        data: pd.DataFrame,
        data_type: str
) -> Tuple[Dict[int, Any], List[int]]:
    """Split data into sequences per id."""
    sequences: Dict[int, Any] = {}
    # order data in ascending id number
    data.sort_values("id")
    current_sequence = initialise_dict(prm["sequence_entries"][data_type])

    id_ = data["id"][0]
    granularities: List[int] = []
    for t in range(len(data["id"])):
        if id_ != data["id"][t]:  # change of id
            current_sequence, sequences[id_] \
                = append_id_sequences(
                    prm, data_type, current_sequence, granularities)
            id_ = data["id"][t]
        for key in prm["sequence_entries"][data_type]:
            # append id sequence with data
            current_sequence[key].append(data[key][t])
            if key == data_type and np.isnan(data[key][t]):
                print(f"l320 np.isnan(data[{key}][{t}])")

    # add the final one to sequences
    _, sequences[id_] \
        = append_id_sequences(
            prm, data_type, current_sequence, granularities)
    assert len(sequences) > 0, "len(sequences) == 0"

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

    data["cum_min_all"] = data.apply(
        lambda x: x.start_h * 60 + x.start_min, axis=1)
    data["cum_min_all_end"] = data.apply(
        lambda x: x.end_h * 60 + x.end_min, axis=1)
    data["cum_min"] = data.apply(
        lambda x: x.cum_min_all + x.cum_day * 24 * 60, axis=1
    )
    data["cum_min_end"] = data.apply(
        lambda x: x.cum_min_all_end + x.cum_day * 24 * 60, axis=1
    )
    data["month"] = data["cum_day"].apply(
        lambda cum_day: (date0 + timedelta(cum_day)).month
    )

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

    data = obtain_time(data, data_source)

    data["keep"] = keep_column(
        data["cum_day"], data["start_avail"], data["end_avail"]
    )

    # set small negative values to zero and remove larger negative values
    var_label = "dist" if data_type == "EV" else data_type
    data[var_label] = data[var_label].map(
        lambda x: x if x > 0 else (0 if x > -1e-2 else None)
    )

    def _remove_none(row):
        if row[var_label] is None or np.isnan(row[var_label]):
            return False

        return row["keep"]

    data["keep"] = data.apply(_remove_none, axis=1)

    if data_source == "CLNR":
        data["keep"] = data.apply(
            lambda x: x.keep
            and x.id in test_cell
            and data_type == prm["test_cell_translation"][test_cell[x.id]],
            axis=1,
        )

    range_dates = [data["cum_day"].min(), data["cum_day"].max()]

    return data, range_dates


def import_segment(
        prm, chunk_rows, data_type
) -> list:
    """In parallel or sequentially, import and process block of data."""

    if all(
            (prm["outs_path"] / f"{label}_{data_id(prm, data_type)}_{chunk_rows[0]}.pickle").is_file()
            for label in prm["outs_labels"]
    ):
        print(f"load previous out {chunk_rows[0]} -> {chunk_rows[1]}")
        return [None] * 7
    else:
        print(f"import segment {chunk_rows[0]} -> {chunk_rows[1]}")

    data_source = prm["data_type_source"][data_type]
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

    # 2 - split into sequences of subsequent times
    sequences, granularities = get_sequences(
        prm, data, data_type)
    del data

    # 3 - convert into days of data at adequate granularity
    if data_source == "CLNR":
        # reduce to desired granularity
        days, abs_error, types_replaced_eval = get_days_clnr(
            prm, sequences, data_type, prm["save_path"]
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
        prm["type_cols"][data_source],
        name_col=list(prm["i_cols"][data_type]),
        hour_min=0,
    )

    # 2 - filter data
    data, range_dates = filter_validity(
        data, data_type, test_cell, start_end_id, prm)

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
    if data_source == "CLNR":
        data["test_cell"] = data["id"].map(lambda id_: test_cell[id_])

    # register mapping of data for producing heatmap later
    all_data = map_all_data(data_source, data, prm)

    return data, all_data, range_dates, n_ids


def map_all_data(data_source, data, prm):
    """Check number of data points per time of day and day of year."""
    if prm["do_heat_map"]:
        all_data = np.zeros((24, 366))
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

    with open(prm["save_path"] / "percentiles.pickle", "wb") as file:
        pickle.dump(percentiles, file)

    return percentiles


def import_data(
        prm: dict,
) -> Tuple[dict, Dict[str, int]]:
    """Import, filter, pre-process data for current block."""
    days = {}  # bank of days of data per data type
    n_data_type = {}  # len of bank per data type

    for data_type in prm["data_types"]:
        print(f"start import {data_type}")

        # identifier for saving data_type-related data
        # savings paths
        day0_path = prm["save_path"] / f"day0_{data_id(prm, data_type)}.pickle"
        n_data_type_path \
            = prm["save_path"] / f"n_dt0_{data_id(prm, data_type)}.npy"

        if prm["n_rows"][data_type] == "all":
            prm["n_rows"][data_type] = get_n_rows(data_type, prm)
        print(f"got n rows")
        chunks_rows = get_data_chunks(prm, data_type)
        print(f"got data_chunks")
        if prm["parallel"]:
            pool = mp.Pool(prm["n_cpu"])
            outs = pool.starmap(
                import_segment,
                [(prm, chunk_rows, data_type)
                 for chunk_rows in chunks_rows],
            )
            pool.close()

        else:
            outs = [
                import_segment(prm, chunk_rows, data_type)
                for chunk_rows in chunks_rows
            ]

        days[data_type] = save_outs(
            outs, prm, data_type, chunks_rows
        )
        n_data_type[data_type] = len(days[data_type])

        # save days for next time
        # print("save days")
        # with open(day0_path, "wb") as file:
        #     pickle.dump(days[data_type], file)
        print("save n_data_type")
        np.save(n_data_type_path, n_data_type[data_type])

        assert len(days[data_type]) > 0, \
            f"all len(days[{data_type}]) {len(days[data_type])}"

    print("get_percentiles")
    get_percentiles(days, prm)

    print("done import data")
    return days, n_data_type
