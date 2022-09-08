"""Functions related to filling in missing values in data."""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import initialise_dict


def _number_missing_points(dt: int, t: int, mins: List[int]) \
        -> Tuple[List[int], List[int]]:
    """Get the number of missing points from current time step."""
    n_miss, fill_t = [], []
    if t == 0 and mins[0] != 0:
        n_miss.append(int(mins[0] / dt))
        fill_t.append(0)
    elif mins[t] != mins[t - 1] + dt:
        n_miss.append(int((mins[t] - mins[t - 1]) / dt))
        fill_t.append(t)
    if t == len(mins) - 1 \
            and mins[t] != 24 * 60 - dt:
        n_miss.append(int((24 * 60 - dt - mins[-1]) / dt))
        fill_t.append(t + 1)

    return n_miss, fill_t


def _potential_replacements(
        prm: dict,
        t_cum: int,
        val_prev: float,
        val_next: float,
        sequence: dict,
        data_type: str
) -> Tuple[list, list, list]:
    """Get the list of potential replacements for the current time steps."""
    # potential replacement (last week, next week, yesterday, tomorrow)
    # look at the replacement t, before t - 1, or after t + 1
    bounds = ["replacement"]
    if val_prev is not None:
        bounds += ["before"]
    if val_next is not None:
        bounds += ["after"]
    # potential replacement values
    vals = initialise_dict(prm["directs"])
    replacements = []  # the potential replacements
    # sum of square of differences (replacement - current value)
    differences = []
    # types of replacement for time t
    types_replacement_t = []

    for direction in prm["directs"]:
        # looking in both directions (before and after)
        vals[direction] = initialise_dict(prm["jumps"])
        for jump in prm["jumps"]:
            # jumping by different durations (day or week)
            vals[direction][jump] = initialise_dict(bounds)
            # how many of the potential replacements are not available
            # (missing value or missing bounds for evaluation)
            sum_none = 0
            for bound in bounds:
                # potential replacement, value before, value after
                idx = [i for i in range(len(sequence["cum_min"]))
                       if sequence["cum_min"][i] == t_cum
                       + prm["direct_mult"][direction] * prm["jump_dt"][jump]
                       + prm["bound_delta"][bound]]
                if len(idx) == 0:
                    # there is no data for the time we are looking for
                    sum_none += 1
                else:
                    vals[direction][jump][bound] \
                        = sequence[data_type][idx[0]]
            if sum_none == 0:
                # potential replacement and bound value(s) available
                # append potential replacement value
                replacements.append(vals[direction][jump]["replacement"])
                # initialise the sum of square of differences
                # with value(s) before and/or after
                diff = 0
                if val_prev is not None:
                    diff += (vals[direction][jump]["before"] - val_prev) ** 2
                if val_next is not None:
                    diff += (vals[direction][jump]["after"] - val_next) ** 2
                # append the sum of squares of differences
                differences.append(diff)
                # append replacement type
                types_replacement_t.append(direction + "_" + jump)

    return replacements, types_replacement_t, differences


def _evaluate_missing(
        prm: dict,
        t: int,
        day: dict,
        data_type: str,
        sequence: dict,
        throw: bool,
        types_replaced: Optional[Dict[str, int]],
        implement: bool = True,
) -> Tuple[dict, bool, Optional[dict], float]:
    """
    Provide a data value estimate for a given time step.

    Based on previous and subsequent available time steps around
    that step and the day and week before and after.

    implement : missing day[dt][t] to be inserted
    not implement : for evaluation, actually day[dt][t] already exists
    """
    # the number of minutes since the start of the day
    t_min = t * prm["dT"]
    # the number of minutes since 01/01/2010
    t_cum = t_min + day["cum_day"] * 24 * 60
    # if data is missing t = the next val;
    # if data is already there t + 1 = the next val
    t_next_val = t if implement else t + 1

    # whether there is data available just before point to fill in
    val_prev = day[data_type][t - 1] if t > 0 else None
    # whether there is data available just after point to fill in
    val_next = (
        day[data_type][t_next_val] if t_next_val < len(day[data_type])
        else None
    )

    if prm["lin_interp"][data_type]:
        if val_prev is None:
            replacement = val_next
        elif val_next is None:
            replacement = val_prev
        else:
            replacement = np.mean([val_prev, val_next])
        if implement:  # insert missing values
            day[data_type].insert(t, replacement)
            day["cum_min"].insert(t, t_cum)
            day["mins"].insert(t, t_min)
    else:
        replacements, types_replacement_t, differences \
            = _potential_replacements(
                prm, t_cum, val_prev, val_next, sequence, data_type)
        if len(replacements) == 0:
            # we cannot replace this value. Just throw the day out
            throw = True
            replacement = None
        else:  # there are potential replacements
            # select lowest sum of squares of differences
            i_min_diff = np.argmin(differences)
            replacement = replacements[i_min_diff]
            types_replaced[types_replacement_t[i_min_diff]] += 1
            if implement:  # insert missing values
                day[data_type].insert(t, replacement)
                day["cum_min"].insert(t, t_cum)
                day["mins"].insert(t, t_min)

    return day, throw, types_replaced, replacement


def _check_fill_missing(
        prm: dict,
        day: dict,
        i: int,
        to_throw: list,
        data_type: str,
        sequences_id: dict,
        types_replaced: Optional[Dict[str, int]],
        throw: bool, ) \
        -> Tuple[Dict[str, float], Optional[Dict[str, int]], List[int]]:
    """Loop through days to check which to fill in and which to throw."""
    to_fill = []
    for t in range(len(day["mins"])):
        n_miss, fill_t = _number_missing_points(prm["dT"], t, day["mins"])
        for n_miss_, fill_t_ in zip(n_miss, fill_t):
            if n_miss_ > 1:
                throw = True
                if i not in to_throw:
                    to_throw.append(i)
            elif n_miss_ == 1:
                to_fill.append(fill_t_)

    # replace single missing values
    i_tfill = 0
    while not throw and i_tfill < len(to_fill):
        t = to_fill[i_tfill]
        day, throw, types_replaced, _ = _evaluate_missing(
            prm, t, day, data_type, sequences_id, throw, types_replaced
        )
        if throw is True:  # could not fill in
            if i not in to_throw:
                to_throw.append(i)
        else:  # we have filled in
            to_fill = [tf if tf < t else tf + 1 for tf in to_fill]

        i_tfill += 1

    return day, types_replaced, to_throw


def _assess_filling_in(
        prm,
        day: dict,
        t: int,
        data_type: str,
        sequence: dict,
        throw: bool,
        types_replaced_eval: Optional[Dict[str, int]]
) -> Tuple[Optional[Dict[str, int]], Optional[List[float]]]:
    """Compare real values with filled in values."""
    abs_error = []
    final_t = t == len(day["mins"]) - 1
    expect_next = day["mins"][t] + prm["dT"]
    next_slot = not final_t and expect_next == day["mins"][t + 1]
    expect_prev = day["mins"][t] - prm["dT"]
    prev_slot = t > 0 and day["mins"][t - 1] == expect_prev
    if (
            (t == 0 and next_slot)
            or (final_t and prev_slot)
            or (prev_slot and next_slot)
    ):
        _, _, types_replaced_eval, replaced_val \
            = _evaluate_missing(
                prm, t, day, data_type, sequence, throw,
                types_replaced_eval, implement=False)
        if replaced_val is not None:
            abs_error.append(abs(day[data_type][t] - replaced_val))

    return types_replaced_eval, abs_error


def fill_whole_days(
        prm, days: list, data_type: str, sequences: dict
) -> Tuple[list, List[int], Optional[List[float]], Optional[Dict[str, int]]]:
    """Loop through days, fill in or throw."""
    to_throw = []
    types_replaced = initialise_dict(prm["replacement_types"], "zero")
    abs_error: Optional[List[float]] = []
    types_replaced_eval: Optional[Dict[str, int]] \
        = initialise_dict(prm["replacement_types"], "zero")

    for i, day in enumerate(days):
        throw = False
        id_ = day["id"]
        # check positive values
        assert len(day[data_type]) > 0, \
            f"i = {i} dt = {data_type} len(days[i][dt])) == 0"
        for t in range(len(days[i][data_type])):
            if day[data_type][t] < 0 and day[data_type][t] > -1e-2:
                day[data_type][t] = 0
                print("Negatives should have been removed before")
            elif day[data_type][t] < 0:
                print(f"days[{i}][{data_type}][{t}] = {day[data_type][t]}")
                throw = True
                if i in to_throw:
                    print(f"double {i} in to_throw")
                to_throw.append(i)

            # test performance of filling in
            if prm["do_test_filling_in"]:
                if random.random() < prm["do_test_filling_in"]:
                    types_replaced_eval, abs_error_ \
                        = _assess_filling_in(
                            prm, day, t, data_type, sequences[id_],
                            throw, types_replaced_eval)
                    abs_error += abs_error_

        # check if values are missing
        if not throw and len(days[i]["mins"]) < 24:
            day, types_replaced, to_throw = _check_fill_missing(
                prm, day, i, to_throw, data_type, sequences[id_],
                types_replaced, throw
            )
        assert len(days[i]["mins"]) > 12 or i in to_throw, \
            f"i {i} id_ {id_} shouldn't this be in to_throw? " \
            f"len(days[i]['mins']) {len(days[i]['mins'])}"
        assert len(days[i]["mins"]) == len(day["mins"]), \
            f"len(days[i]['mins']) {len(days[i]['mins'])} != " \
            f"len(day['mins']) {len(day['mins'])}"

    if not prm["do_test_filling_in"]:
        abs_error, types_replaced_eval = None, None

    return days, to_throw, abs_error, types_replaced_eval


def stats_filling_in(
        prm: dict,
        data_type: str,
        all_abs_error: List[float],
        types_replaced: dict,
        save_path: Path
) -> dict:
    """Obtain accuracy statistics on the filling in methodology."""
    filling_in = initialise_dict(prm["filling_in_entries"])
    if len(all_abs_error) == 0:
        return None

    if prm["data_type_source"][data_type] == "CLNR":
        filling_in["mean_abs_err"] = np.mean(all_abs_error)
        filling_in["std_abs_err"] = np.std(all_abs_error)
        filling_in["p99_abs_err"] = np.percentile(all_abs_error, 99)
        filling_in["max_abs_err"] = np.max(all_abs_error)
        filling_in["share_types_replaced"] = {}
        sum_share = 0

        for key in prm["replacement_types"]:
            filling_in["share_types_replaced"][key] \
                = types_replaced[key] / len(all_abs_error)
            sum_share += filling_in["share_types_replaced"][key]
        assert not (data_type == "dem" and abs(sum_share - 1) > 1e-3), \
            f"dt = {data_type}, sum_share = {sum_share}"

    with open(save_path / f"filling_in_{data_type}.pickle", "wb") as file:
        pickle.dump(filling_in, file)

    return None
