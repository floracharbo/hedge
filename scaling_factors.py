"""Get scaling factor transitions characteristics."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import gamma, pearsonr

from utils import initialise_dict


def _get_correlation(
        f_prevs_all: List[float],
        f_nexts_all: List[float],
        save_path: Path,
        data_type: str,
        label: Optional[str] = None):

    factors = {}
    if len(f_prevs_all) == 0:
        return None

    if data_type == "EV":
        i_not_none = [
            i
            for i in range(len(f_prevs_all))
            if f_prevs_all[i] is not None and f_nexts_all[i] is not None
        ]
        f_prevs, f_nexts = [
            [fs[i] for i in i_not_none] for fs in [f_prevs_all, f_nexts_all]
        ]
    else:
        f_prevs, f_nexts = f_prevs_all, f_nexts_all

    if len(f_prevs) == 0:
        print("len(f_prevs) == 0")
        return None

    factors["corr"], _ = pearsonr(f_prevs, f_nexts)

    for i, (f_prev, f_next) in enumerate(zip(f_prevs, f_nexts)):
        assert f_prev >= 0, f"{label} i {i} f_prev = {f_prev}"
        assert f_next >= 0, f"{label} i {i} f_next = {f_next}"

    # plot f_next vs f_prev
    if prm["plots"]:
        for stylised in [True, False]:
            fig = plt.figure()
            plt.plot(f_prevs, f_nexts, "o", label="data", alpha=0.1)
            # plt.xlim([0, max(f_prevs)])
            # plt.ylim([0, max(f_nexts)])
            title = f"f_prev vs f_next {data_type} {label}"
            if stylised:
                title += " stylised"
                ax = plt.gca()
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                ax.spines.right.set_visible(False)
                ax.spines.top.set_visible(False)
                plt.tight_layout()
            else:
                plt.title(title)
                plt.xlabel("f(t-1)")
                plt.ylabel("f(t)")
                plt.plot(
                    np.linspace(0, max(f_prevs)),
                    np.linspace(0, max(f_nexts)),
                    "--",
                    label="perfect correlation",
                )
                plt.legend()

            fig.savefig(save_path / "factors" / title.replace(" ", "_"))
            plt.close("all")

    np.save(save_path / "factors" / f"corr_{data_type}_{label}", factors["corr"])
    factors["f_prevs"] = f_prevs
    factors["f_nexts"] = f_nexts

    return factors


def _count_transitions(
        f_prevs: List[float],
        f_nexts: List[float],
        n_intervals: int,
        xs: List[float]):
    n_pos, p_pos = [np.zeros((n_intervals, n_intervals)) for _ in range(2)]
    n_zero2pos = np.zeros(n_intervals)
    i_non_zeros \
        = [i for i, (f_prev, f_next) in enumerate(zip(f_prevs, f_nexts))
           if f_prev is not None
           and f_prev > 0
           and f_next is not None
           and f_next > 0]
    i_zero_to_nonzero \
        = [i for i, (f_prev, f_next) in enumerate(zip(f_prevs, f_nexts))
           if (f_prev is None or f_prev == 0)
           and f_next is not None
           and f_next > 0]
    for i in i_non_zeros:
        i_prev = [j for j in range(n_intervals - 1) if f_prevs[i] >= xs[j]][-1]
        i_next = [j for j in range(n_intervals - 1) if f_nexts[i] >= xs[j]][-1]
        n_pos[i_prev, i_next] += 1
    for i in i_zero_to_nonzero:
        i_next = [j for j in range(n_intervals - 1) if f_nexts[i] >= xs[j]][-1]
        n_zero2pos[i_next] += 1

    for i_prev in range(n_intervals - 1):
        sum_next = sum(n_pos[i_prev])
        p_pos[i_prev] = [m / sum_next for m in n_pos[i_prev]] if sum_next > 0 \
            else None
    p_zero2pos = [n / sum(n_zero2pos) if sum(n_zero2pos) > 0
                  else np.nan
                  for n in n_zero2pos]
    p_zero2pos = np.reshape(p_zero2pos, (1, n_intervals))

    return p_pos, p_zero2pos


def _transition_intervals(
        f_prevs: List[float],
        f_nexts: List[float],
        transition: str,
        n_intervals: int,
        save_path: Path,
        plots: bool
) -> Tuple[np.ndarray, List[float], np.ndarray, List[float]]:
    xs = np.linspace(min(min(f_prevs), min(f_nexts)),
                     max(max(f_prevs), max(f_nexts)),
                     n_intervals + 1)
    mid_xs = [np.mean(xs[i: i + 2]) for i in range(n_intervals)]

    p_pos, p_zero2pos = _count_transitions(f_prevs, f_nexts, n_intervals, xs)

    labels_prob = [
        "EV non zero factor transition probabilities",
        "EV factor probabilities after a day without trips",
    ]

    for trans_prob, label_prob in zip([p_pos, p_zero2pos], labels_prob):
        # trans_prob: transition probabililites
        trans_prob[(trans_prob == 0) | (np.isnan(trans_prob))] = 1e-5
        assert (trans_prob >= 0).all(), "error not (trans_prob >= 0).all()"

        if plots:
            fig, ax = plt.subplots()
            ax.imshow(trans_prob, norm=LogNorm())
            # fig.colorbar(im, ax = ax)
            tick_labels = [0] + [xs[int(i - 1)] for i in ax.get_xticks()][1:]
            tick_labels_format = ["{:.1f}".format(label) for label in tick_labels]
            title = f"{label_prob}, {transition} n_intervals {n_intervals}"
            plt.title(title)
            ax.set_xticklabels(tick_labels_format)
            ax.set_yticklabels(tick_labels_format)
            ax.set_xlabel("f(t + 1)")
            ax.set_ylabel("f(t)")
            fig.savefig(save_path / "factors" / title.replace(" ", "_"))

    return p_pos, p_zero2pos, xs, mid_xs


def _print_error_factors(ex, label, f_prevs, f_nexts):
    """Print info if Exception in computing pearsonr."""
    print(f"ex = {ex}, label = {label}")
    for factor in f_prevs + [f_nexts[-1]]:
        if type(factor) not in [float, np.float64]:
            print(f"label {label} type(factor) {type(factor)}")
            print(f"np.isnan(f) = {np.isnan(factor)}")


def _fit_gamma(f_prevs, f_nexts, save_path, data_type, plots, label=None):
    assert sum(1 for f_prev in f_prevs if f_prev is None) == 0,\
        "None f_prevs"

    f_next_sort = [f_next for f_prev, f_next in sorted(zip(f_prevs, f_nexts))]
    f_prev_sort = [f_prev for f_prev, f_next in sorted(zip(f_prevs, f_nexts))]

    errors = [f_next_sort[i] - f_prev_sort[i] for i in range(len(f_prevs))]
    gamma_prms = gamma.fit(errors)

    # plot
    if plots:
        fig = plt.figure()
        plt.hist(errors, density=1, alpha=0.5, label="data")
        x = np.linspace(
            gamma.ppf(0.0001, *gamma_prms),
            gamma.ppf(0.9999, *gamma_prms),
            100,
        )
        plt.plot(x, gamma.pdf(x, gamma_prms[0], gamma_prms[1], gamma_prms[2]),
                 label="gamma pdf")
        plt.legend(loc="upper right")
        title = (
            f"fit of gamma distribution to error around "
            f"perfect correlation {data_type} {label}"
        )
        plt.title(title)
        fig.savefig(save_path / "factors" / title.replace(" ", "_"))
        plt.close("all")

    mean_residual = gamma.stats(*gamma_prms, moments="m")

    sum_log_likelihood = 0
    for error in errors:
        sum_log_likelihood += np.log(gamma.pdf(error, *gamma_prms))

    return mean_residual, gamma_prms


def _enough_data(banks_, weekday_type):
    return sum(banks_[day_type] is None for day_type in weekday_type) == 0


def _ev_transitions(prm, factors, save_path):
    p_pos, p_zero2pos, xs, mid_xs = [{} for _ in range(4)]

    for transition in prm["day_trans"]:
        f_prevs, f_nexts \
            = [factors["EV"][transition][f] for f in ["f_prevs", "f_nexts"]]
        # EV transitions
        [p_pos[transition], p_zero2pos[transition],
         xs[transition], mid_xs[transition]] = _transition_intervals(
            f_prevs, f_nexts, transition, prm["n_intervals"], save_path, prm["plots"])
    dictionaries = p_pos, p_zero2pos, xs, mid_xs
    labels = ["p_pos", "p_zero2pos", "xs", "mid_xs"]
    for dictionary, label in zip(dictionaries, labels):
        with open(save_path / "factors" / f"EV_{label}.pickle", "wb") as file:
            pickle.dump(dictionary, file)


def _scaling_factors_behaviour_types(
        prm: dict,
        banks: dict,
        save_path: Path
) -> Tuple[dict, dict, dict]:
    mean_residual, gamma_prms = [
        initialise_dict(prm["CLNR_types"], "empty_dict") for _ in range(2)
    ]
    factors = initialise_dict(prm["data_types"], "empty_dict")
    n_transitions = intitialise_dict(prm["data_types"], "zero")

    for data_type in prm["behaviour_types"]:
        if not _enough_data(banks[data_type], prm["weekday_type"]):
            continue
        factors[data_type] = initialise_dict(prm["day_trans"], "empty_dict")
        for transition in prm["day_trans"]:
            f_prevs_all, f_nexts_all \
                = [banks[data_type][transition][f"f{i_f}"]
                   for i_f in range(2)]
            if len(f_prevs_all) == 0:
                print(f"{data_type} transition {transition} len f_prevs_all == 0")
            factors[data_type][transition] = _get_correlation(
                f_prevs_all, f_nexts_all, save_path, data_type, transition)
            n_transitions[data_type] += len(f_prevs_all)

    if "dem" in prm["data_types"]:
        for transition in prm["day_trans"]:
            # Household demand
            if factors["dem"][transition] is None:
                mean_residual["dem"][transition], gamma_prms["dem"][transition] \
                    = None, None
                print(f"factors['dem'][{transition}] is None")
                continue

            f_prevs, f_nexts = [
                factors["dem"][transition][f] for f in ["f_prevs", "f_nexts"]
            ]

            assert sum(1 for f_prev in f_prevs if f_prev == 0) == 0, \
                "need to account for Nones for fit gamma"
            mean_residual["dem"][transition], gamma_prms["dem"][transition] \
                = _fit_gamma(f_prevs, f_nexts, save_path, "dem", prm["plots"], transition)
            if mean_residual["dem"][transition] is None:
                print(f"mean_residual['dem'][{transition}] = {mean_residual['dem'][transition]}")
                print(f"np.shape(f_prevs) = {np.shape(f_prevs)}")
                print(f"np.shape(f_nexts) = {np.shape(f_nexts)}")

    if "EV" in prm["data_types"] \
            and _enough_data(banks["EV"], prm["weekday_type"]):
        _ev_transitions(prm, factors, save_path)

    return factors, mean_residual, gamma_prms, n_transitions


def _scaling_factors_generation(n_data_type_, days_, save_path, plots):
    # generation
    # obtain subsequent factors
    factors = []
    mean_residual, gamma_prms = {}, {}
    n_transitions = 0
    for i_month, month in enumerate(range(1, 13)):
        f_prev_gen, f_next_gen = [], []
        count = 0
        for i in range(n_data_type_ - 1):
            day, next_day = [days_[i_] for i_ in [i, i + 1]]
            same_id = day["id"] == next_day["id"]
            subsequent_days = day["cum_day"] + 1 == next_day["cum_day"]
            current_month = day["month"] == month
            if (
                    same_id
                    and subsequent_days
                    and current_month
            ):  # record transition
                count += 1
                f_prev_gen.append(day["factor"])
                f_next_gen.append(next_day["factor"])

        n_transitions += len(f_prev_gen)

        factors.append(
            _get_correlation(f_prev_gen, f_next_gen, save_path, "gen", month)
        )

        # fit gamma
        if len(f_prev_gen) > 0:
            mean_residual[i_month], gamma_prms[i_month] = _fit_gamma(
                factors[i_month]["f_prevs"], factors[i_month]["f_nexts"],
                save_path, "gen", plots, month)
        else:
            mean_residual[i_month], gamma_prms[i_month] = None, None

    return factors, mean_residual, gamma_prms, n_transitions


def scaling_factors(prm, banks, days, n_data_type, save_path):
    """
    Obtain scaling factors corresponding to the data.

    n_data_type: len of bank per data type
    """
    # Initialise dictionaries
    f_max, f_min, f_mean \
        = [initialise_dict(prm["data_types"]) for _ in range(3)]

    # generation assumed to be month-dependent
    # whilst household consumption and travel patterns assumed not to be
    if "gen" in prm["data_types"]:
        for i_month in range(12):
            list_factors = banks["gen"][i_month]["factor"]
            f_max["gen"].append(
                np.max(list_factors) if len(list_factors) > 0 else None
            )
            f_min["gen"].append(
                np.min(list_factors) if len(list_factors) > 0 else None
            )
            f_mean["gen"].append(
                np.mean(list_factors) if len(list_factors) > 0 else None
            )

    for data_type in prm["behaviour_types"]:
        list_factors = [day["factor"] for day in days[data_type]]
        f_max[data_type] = np.max(list_factors)
        f_min[data_type] = np.min(list_factors)
        f_mean[data_type] = np.mean(list_factors)

    factors, mean_residual, gamma_prms, n_transitions \
        = _scaling_factors_behaviour_types(prm, banks, save_path)

    for transition in prm["day_trans"]:
        if gamma_prms['dem'][transition] is None:
            print(f"gamma_prms['dem'][{transition}] is None")

    if "gen" in prm["data_types"]:
        factors["gen"], mean_residual["gen"], gamma_prms["gen"], n_transitions["gen"]  \
            = _scaling_factors_generation(
            n_data_type["gen"], days["gen"], save_path, prm["plots"])

    folder_path = save_path / "factors"
    for property_, obj in zip(["f_min", "f_max", "f_mean", "n_transitions"],
                              [f_min, f_max, f_mean, n_transitions]):
        path = folder_path / f"{property_}.pickle"
        with open(path, "wb") as file:
            pickle.dump(obj, file)
        np.save(path, obj)

    if len(mean_residual.keys()) < 2:
        print(f"missing mean_residual, mean_residual.keys() = {mean_residual.keys()}")
    if len(gamma_prms.keys()) < 2:
        print(f"missing gamma_prms, gamma_prms.keys() = {gamma_prms.keys()}")

    for property_, obj in zip(["mean_residual", "gamma_prms"], [mean_residual, gamma_prms]):
        with open(folder_path / f"{property_}.pickle", "wb") as file:
            pickle.dump(obj, file)