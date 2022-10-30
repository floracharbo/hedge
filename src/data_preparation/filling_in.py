"""Functions related to filling in missing values in data."""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils import initialise_dict


def _number_missing_points(step_len: int, time: int, mins: List[int]) \
        -> Tuple[List[int], List[int]]:
    """Get the number of missing points from current time step."""
    n_miss, fill_t = [], []
    if time == 0 and mins[0] != 0:
        n_miss.append(int(mins[0] / step_len))
        fill_t.append(0)
    elif time > 0 and mins[time] != mins[time - 1] + step_len:
        n_miss.append(int((mins[time] - mins[time - 1]) / step_len))
        fill_t.append(time)
    if time == len(mins) - 1 \
            and mins[time] != 24 * 60 - step_len:
        n_miss.append(int((24 * 60 - step_len - mins[-1]) / step_len))
        fill_t.append(time + 1)

    return n_miss, fill_t


def _potential_replacements(
        prm: dict,
        t_cum: int,
        val_prev: float,
        val_next: float,
        sequence: dict,
        data_type: str,
        fill_type: str
) -> Tuple[list, list, list]:
    """Get the list of potential replacements for the current time steps."""
    # potential replacement (last week, next week, yesterday, tomorrow)
    # look at the replacement time, before time - 1, or after time + 1
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
    # types of replacement for current time
    types_replacement_t = []
    jumps = prm["jumps"][fill_type]

    for direction in prm["directs"]:
        # looking in both directions (before and after)
        vals[direction] = initialise_dict(jumps)
        for jump in jumps:
            # jumping by different durations (day or week)
            vals[direction][jump] = initialise_dict(bounds)
            # how many of the potential replacements are not available
            # (missing value or missing bounds for evaluation)
            sum_none = 0
            for bound in bounds:
                # potential replacement, value before, value after
                idx = np.where(
                    sequence["cum_min"] == t_cum
                    + prm["direct_mult"][direction] * prm["jump_dt"][jump]
                    + prm["bound_delta"][bound]
                )[0]
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
                types_replacement_t.append(f"{direction}_{jump}")

    return replacements, types_replacement_t, differences


def _evaluate_missing(
        prm: dict,
        time: int,
        day: dict,
        data_type: str,
        sequence: dict,
        throw: bool,
        types_replaced: Optional[Dict[str, int]],
        implement: bool = True,
        fill_type=None
) -> Tuple[dict, bool, Optional[dict], float]:
    """
    Provide a data value estimate for a given time step.

    Based on previous and subsequent available time steps around
    that step and the day and week before and after.

    implement : missing day[data_type][time] to be inserted
    not implement : for evaluation, actually day[data_type][t] already exists
    """
    # the number of minutes since the start of the day
    t_min = time * prm["step_len"]
    # the number of minutes since 01/01/2010
    t_cum = t_min + day["cum_day"] * 24 * 60
    # if data is missing time = the next val;
    # if data is already there time + 1 = the next val
    t_next_val = time if implement else time + 1

    # whether there is data available just before point to fill in
    val_prev = day[data_type][time - 1] if time > 0 else None
    # whether there is data available just after point to fill in
    val_next = (
        day[data_type][t_next_val] if t_next_val < len(day[data_type])
        else None
    )
    if fill_type is None:
        fill_type = prm["fill_type"][data_type]
    if fill_type == "lin_interp":
        if val_prev is None:
            replacement = val_next
        elif val_next is None:
            replacement = val_prev
        else:
            replacement = np.mean([val_prev, val_next])

    else:
        replacements, types_replacement_t, differences \
            = _potential_replacements(
                prm, t_cum, val_prev, val_next, sequence, data_type, fill_type)
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
        day[data_type].insert(time, replacement)
        day["cum_min"].insert(time, t_cum)
        day["mins"].insert(time, t_min)

    return day, throw, types_replaced, replacement


def _check_fill_missing(
        prm: dict,
        day: dict,
        i: int,
        to_throw: list,
        data_type: str,
        sequences_id: dict,
        types_replaced: Optional[Dict[str, int]],
        throw: bool,
        save_path: Path
) -> Tuple[Dict[str, float], Optional[Dict[str, int]], List[int]]:
    """Loop through days to check which to fill in and which to throw."""
    assert len(set(day["mins"])) == len(day["mins"]), \
        f"mins duplicates {day['mins']}"
    missing_fig_path = save_path / f"missing_data_{data_type}.png"
    to_fill = []
    for time in range(len(day["mins"])):
        n_miss, fill_t \
            = _number_missing_points(prm["step_len"], time, day["mins"])
        for n_miss_, fill_t_ in zip(n_miss, fill_t):
            if n_miss_ > 1:
                throw = True
                if i not in to_throw:
                    to_throw.append(i)
                if (
                        prm["plots"]
                        and n_miss_ == 2
                        and len(day["mins"]) > 20
                        and not missing_fig_path.is_file()
                ):
                    _plot_missing_data(
                        day, data_type, prm, missing_fig_path
                    )
            else:
                to_fill.append(fill_t_)
                assert not (to_fill == 0 and day["mins"][0] == 0), \
                    "why add zero if already 0?"

    assert len(set(day["mins"])) == len(day["mins"]), \
        f"mins duplicates {day['mins']}"

    # replace single missing values
    i_tfill = 0
    while not throw and i_tfill < len(to_fill):
        time = to_fill[i_tfill]
        day, throw, types_replaced, _ = _evaluate_missing(
            prm, time, day, data_type, sequences_id, throw, types_replaced
        )
        assert len(set(day["mins"])) == len(day["mins"]), \
            f"mins duplicates {day['mins']} " \
            f"to_fill[{i_tfill}] {to_fill[i_tfill]} time {time}"
        if throw is True:  # could not fill in
            if i not in to_throw:
                to_throw.append(i)
        else:  # we have filled in
            to_fill = [tf if tf < time else tf + 1 for tf in to_fill]

        i_tfill += 1
    assert throw or all(
        min % prm["step_len"] == 0 for min in day["mins"]
    ), f"day['mins'] {day['mins']} not right granularity {data_type}"
    assert throw or all(
        next_min == min + prm["step_len"]
        for min, next_min in zip(day["mins"][:-1], day["mins"][1:])
    ), f"day['mins'] {day['mins']} for right sequence {data_type}"
    assert throw or len(day["mins"]) == prm["n"], \
        f"len mins {len(day['mins'])} but no throw {data_type}"

    return day, types_replaced, to_throw


def _assess_filling_in(
        prm,
        day: dict,
        time: int,
        data_type: str,
        sequence: dict,
        throw: bool,
        types_replaced_eval: Optional[dict],
        abs_error: dict
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, np.ndarray]]]:
    """Compare real values with filled in values."""
    final_t = time == len(day["mins"]) - 1
    expect_next = day["mins"][time] + prm["step_len"]
    next_slot = not final_t and expect_next == day["mins"][time + 1]
    expect_prev = day["mins"][time] - prm["step_len"]
    prev_slot = time > 0 and day["mins"][time - 1] == expect_prev
    if (
            (time == 0 and next_slot)
            or (final_t and prev_slot)
            or (prev_slot and next_slot)
    ):
        for fill_type in prm["fill_types"]:
            _, _, types_replaced_eval[fill_type], replaced_val \
                = _evaluate_missing(
                prm, time, day, data_type, sequence, throw,
                types_replaced_eval[fill_type],
                implement=False, fill_type=fill_type
            )
            if replaced_val is not None:
                abs_error[fill_type] = np.append(
                    abs_error[fill_type],
                    abs(day[data_type][time] - replaced_val)
                )

    return types_replaced_eval, abs_error


def fill_whole_days(
        prm,
        days: list,
        data_type: str,
        sequences: dict,
        save_path: Path
) -> Tuple[list, List[int], Optional[List[float]], Optional[Dict[str, int]]]:
    """Loop through days, fill in or throw."""
    to_throw: List[int] = []
    abs_error = initialise_dict(prm["fill_types"], "empty_np_array")
    types_replaced = initialise_dict(prm["replacement_types"], "zero")
    types_replaced_eval = initialise_dict(prm["fill_types"], "empty_dict")
    for fill_type in prm["fill_types"]:
        types_replaced_eval[fill_type] \
            = initialise_dict(prm["replacement_types"], "zero")

    for i, day in enumerate(days):
        throw = False
        id_ = day["id"]
        # check positive values
        assert len(day[data_type]) > 0, \
            f"i = {i} data_type = {data_type} len(days[i][data_type])) == 0"
        for time in range(len(day[data_type])):
            assert not day[data_type][time] < 0, \
                "Negatives should have been removed before"

            # test performance of filling in
            if prm["do_test_filling_in"]:
                if random.random() < prm["prob_test_filling_in"]:
                    types_replaced_eval, abs_error \
                        = _assess_filling_in(
                            prm, day, time, data_type, sequences[id_],
                            throw, types_replaced_eval, abs_error)

        # check if values are missing
        if not throw and len(day["mins"]) < prm["n"]:
            day, types_replaced, to_throw = _check_fill_missing(
                prm, day, i, to_throw, data_type, sequences[id_],
                types_replaced, throw, save_path
            )

        assert len(days[i]["mins"]) > 12 or i in to_throw, \
            f"i {i} id_ {id_} shouldn't this be in to_throw? " \
            f"len(days[i]['mins']) {len(days[i]['mins'])}"
        assert len(days[i]["mins"]) == len(day["mins"]), \
            f"len(days[i]['mins']) {len(days[i]['mins'])} != " \
            f"len(day['mins']) {len(day['mins'])}"

    if not prm["do_test_filling_in"]:
        abs_error, types_replaced_eval = None, None
    else:
        for fill_type in prm["fill_types_choice"]:
            assert sum(types_replaced_eval[fill_type].values()) > 0, \
                f"{data_type} {fill_type} " \
                f"types_replaced_eval {types_replaced_eval}"
        for fill_type in prm["fill_types"]:
            assert len(abs_error[fill_type]) > 0, \
                f"len(abs_error[{fill_type}]) == 0"

    return days, to_throw, abs_error, types_replaced_eval


def stats_filling_in(
        prm: dict,
        data_type: str,
        all_abs_error: dict,
        types_replaced: dict,
) -> dict:
    """Obtain accuracy statistics on the filling in methodology."""
    filling_in = initialise_dict(prm["filling_in_entries"])
    if len(all_abs_error[prm["fill_types"][0]]) == 0:
        return None

    for key in [
        "mean_abs_err", "std_abs_err", "p99_abs_err", "max_abs_err", "n_sample"
    ]:
        filling_in[key] = {}
    for fill_type in prm["fill_types"]:
        filling_in["mean_abs_err"][fill_type] \
            = np.mean(all_abs_error[fill_type])
        filling_in["std_abs_err"][fill_type] \
            = np.std(all_abs_error[fill_type])
        filling_in["p99_abs_err"][fill_type] \
            = np.percentile(all_abs_error[fill_type], 99)
        filling_in["max_abs_err"][fill_type] \
            = np.max(all_abs_error[fill_type])
        filling_in["n_sample"][fill_type] = len(all_abs_error[fill_type])

    filling_in["share_types_replaced"] \
        = initialise_dict(prm["fill_types_choice"], "empty_dict")
    for fill_type in prm["fill_types_choice"]:
        sum_share = 0
        for replacement_type in prm["replacement_types"]:
            share = types_replaced[fill_type][replacement_type] \
                / filling_in["n_sample"][fill_type]
            filling_in["share_types_replaced"][fill_type][replacement_type] \
                = share
            sum_share += share
        assert not (data_type == "loads" and abs(sum_share - 1) > 1e-3), \
            f"data_type = {data_type}, sum_share = {sum_share} " \
            f"types_replaced[{fill_type}] = {types_replaced}"

    with open(prm["save_other"] / f"filling_in_{data_type}.pickle", "wb") as file:
        pickle.dump(filling_in, file)

    return None


def _plot_missing_data(day, data_type, prm, missing_fig_path):
    fig = plt.figure()
    plt.rcParams['font.size'] = '18'
    times = [min / 60 for min in day["mins"]]
    for time in range(len(day[data_type]) - 1):
        if day["mins"][time] + prm["step_len"] == day["mins"][time + 1]:
            plt.plot(
                times[time: time + 2],
                day[data_type][time: time + 2],
                '-o',
                color="k",
                lw=3
            )
        else:
            plt.plot(
                times[time: time + 2],
                day[data_type][time: time + 2],
                '--o',
                color="k"
            )
        plt.xlabel("Time [hour]")
        plt.ylabel("[kWh]")
    plt.tight_layout()
    fig.savefig(
        str(missing_fig_path)[:-4] + '.pdf', bbox_inches='tight',
        format='pdf', dpi=1200
    )
    plt.close("all")
