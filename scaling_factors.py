"""Get scaling factor transitions characteristics."""

import pickle
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import norm, pearsonr

from utils import initialise_dict


def _plot_f_next_vs_prev(prm, factors_path, f_prevs, f_nexts, label):
    # plot f_next vs f_prev
    if prm["plots"]:
        for stylised in [True, False]:
            fig = plt.figure()
            plt.plot(f_prevs, f_nexts, "o", label="data", alpha=0.1)
            title = f"f_prev vs f_next {label}"
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

            fig.savefig(factors_path / title.replace(" ", "_"))
            plt.close("all")


def _get_correlation(
        f_prevs_all: List[float],
        f_nexts_all: List[float],
        prm: dict,
        data_type: str,
        label: Optional[str] = None):

    if len(f_prevs_all) == 0:
        return None

    factors = {}
    if data_type == "EV":
        mask_nones = \
            np.isnan(f_prevs_all.astype(float)) \
            | np.isnan(f_nexts_all.astype(float))
        f_prevs, f_nexts \
            = [fs[~mask_nones] for fs in [f_prevs_all, f_nexts_all]]
    else:
        f_prevs, f_nexts = f_prevs_all, f_nexts_all

    if len(f_prevs) == 0:
        print("len(f_prevs) == 0")
        return None

    for i, (f_prev, f_next) in enumerate(zip(f_prevs, f_nexts)):
        assert f_prev >= 0, f"{label} i {i} f_prev = {f_prev}"
        assert f_next >= 0, f"{label} i {i} f_next = {f_next}"

    factors["corr"], _ = pearsonr(f_prevs, f_nexts)

    factors_path = prm["save_path"] / "factors"

    _plot_f_next_vs_prev(
        prm, factors_path, f_prevs, f_nexts, f"{data_type} {label}"
    )

    np.save(factors_path / f"corr_{data_type}_{label}", factors["corr"])
    factors["f_prevs"] = f_prevs
    factors["f_nexts"] = f_nexts

    return factors


def _count_transitions(
        f_prevs: List[float],
        f_nexts: List[float],
        n_intervals: int,
        fs_brackets: List[float]):
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
        i_prev = [
            j for j in range(n_intervals - 1) if f_prevs[i] >= fs_brackets[j]
        ][-1]
        i_next = [
            j for j in range(n_intervals - 1) if f_nexts[i] >= fs_brackets[j]
        ][-1]
        n_pos[i_prev, i_next] += 1
    for i in i_zero_to_nonzero:
        i_next = [
            j for j in range(n_intervals - 1) if f_nexts[i] >= fs_brackets[j]
        ][-1]
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
        prm: int,
) -> Tuple[np.ndarray, List[float], np.ndarray, List[float]]:
    fs_brackets = np.linspace(
        min(min(f_prevs), min(f_nexts)),
        max(max(f_prevs), max(f_nexts)),
        prm["n_intervals"] + 1
    )
    mid_fs_brackets \
        = [np.mean(fs_brackets[i: i + 2]) for i in range(prm["n_intervals"])]

    p_pos, p_zero2pos = _count_transitions(
        f_prevs, f_nexts, prm["n_intervals"], fs_brackets
    )

    labels_prob = [
        "EV non zero factor transition probabilities",
        "EV factor probabilities after a day without trips",
    ]

    for trans_prob, label_prob in zip([p_pos, p_zero2pos], labels_prob):
        # trans_prob: transition probabililites
        trans_prob[(trans_prob == 0) | (np.isnan(trans_prob))] = 1e-5
        assert (trans_prob >= 0).all(), "error not (trans_prob >= 0).all()"

        if prm["plots"]:
            fig, ax = plt.subplots()
            ax.imshow(trans_prob, norm=LogNorm())
            tick_labels = [
                f"{label:.1f}" for label in
                [0] + [fs_brackets[int(i - 1)] for i in ax.get_xticks()][1:]
            ]
            title \
                = f"{label_prob} {transition} n_intervals {prm['n_intervals']}"
            plt.title(title)
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels)
            ax.set_xlabel("f(t + 1)")
            ax.set_ylabel("f(t)")
            fig.savefig(prm["save_path"] / "factors" / title.replace(" ", "_"))

    return p_pos, p_zero2pos, fs_brackets, mid_fs_brackets


def _print_error_factors(ex, label, f_prevs, f_nexts):
    """Print info if Exception in computing pearsonr."""
    print(f"ex = {ex}, label = {label}")
    for factor in f_prevs + [f_nexts[-1]]:
        if type(factor) not in [float, np.float64]:
            print(f"label {label} type(factor) {type(factor)}")
            print(f"np.isnan(f) = {np.isnan(factor)}")


def _compare_factor_error_distributions(prm, errors, save_label):
    sum_log_likelihood = {}
    prms = {}
    for label, distr in prm["candidate_factor_distributions"].items():
        prms[label] = distr.fit(errors)
        sum_log_likelihood[label] = 0
        for error in errors:
            sum_log_likelihood[label] += np.log(distr.pdf(error, *prms[label]))
    for item_label, item in zip(['sum_log_likelihood','all_distribution_prms'], [sum_log_likelihood, prms]):
        with open(
                prm["save_path"]
                / "factors"
                / f"{item_label}_{save_label}.pickle",
                "wb"
        ) as file:
            pickle.dump(item, file)



def _fit_residual_distribution(f_prevs, f_nexts, prm, data_type, label=None):
    assert sum(1 for f_prev in f_prevs if f_prev is None) == 0,\
        "None f_prevs"

    f_next_sort = [f_next for f_prev, f_next in sorted(zip(f_prevs, f_nexts))]
    f_prev_sort = [f_prev for f_prev, f_next in sorted(zip(f_prevs, f_nexts))]

    errors = [f_next_sort[i] - f_prev_sort[i] for i in range(len(f_prevs))]
    residual_distribution_prms = norm.fit(errors)

    # plot
    if prm["plots"]:
        fig = plt.figure()
        plt.hist(errors, density=1, alpha=0.5, label="data")
        factor_residuals = np.linspace(
            norm.ppf(0.0001, *residual_distribution_prms),
            norm.ppf(0.9999, *residual_distribution_prms),
            100,
        )
        plt.plot(
            factor_residuals,
            norm.pdf(
                factor_residuals, *residual_distribution_prms
            ),
            label="norm pdf"
        )
        plt.legend(loc="upper right")
        title = (
            f"fit of norm distribution to error around "
            f"perfect correlation {data_type} {label}"
        )
        plt.title(title)
        fig.savefig(prm["save_path"] / "factors" / title.replace(" ", "_"))
        plt.close("all")

    mean_residual = norm.stats(*residual_distribution_prms, moments="m")
    if prm["test_factor_distr"]:
        _compare_factor_error_distributions(
            prm, errors, f"{data_type}_{label}"
        )

    return mean_residual, residual_distribution_prms


def _enough_data(banks_, weekday_type):
    return sum(banks_[day_type] is None for day_type in weekday_type) == 0


def _ev_transitions(prm, factors):
    p_pos, p_zero2pos, fs_brackets, mid_fs_brackets = [{} for _ in range(4)]

    for transition in prm["day_trans"]:
        f_prevs, f_nexts \
            = [factors["EV"][transition][f] for f in ["f_prevs", "f_nexts"]]
        # EV transitions
        [p_pos[transition], p_zero2pos[transition],
         fs_brackets[transition], mid_fs_brackets[transition]] \
            = _transition_intervals(f_prevs, f_nexts, transition, prm)
    dictionaries = p_pos, p_zero2pos, fs_brackets, mid_fs_brackets
    labels = ["p_pos", "p_zero2pos", "fs_brackets", "mid_fs_brackets"]
    for dictionary, label in zip(dictionaries, labels):
        with open(
                prm["save_path"] / "factors" / f"EV_{label}.pickle", "wb"
        ) as file:
            pickle.dump(dictionary, file)


def _scaling_factors_behaviour_types(
        prm: dict,
        banks: dict,
) -> Tuple[dict, dict, dict, dict]:
    mean_residual, residual_distribution_prms = [
        initialise_dict(prm["CLNR_types"], "empty_dict") for _ in range(2)
    ]
    factors = initialise_dict(prm["data_types"], "empty_dict")
    n_transitions = initialise_dict(prm["data_types"], "zero")

    for data_type in prm["behaviour_types"]:
        if not _enough_data(banks[data_type], prm["weekday_type"]):
            continue
        factors[data_type] = initialise_dict(prm["day_trans"], "empty_dict")
        for transition in prm["day_trans"]:
            f_prevs_all, f_nexts_all \
                = [np.array(banks[data_type][transition][f"f{i_f}"])
                   for i_f in range(2)]
            if len(f_prevs_all) == 0:
                print(
                    f"{data_type} transition {transition} "
                    f"len f_prevs_all == 0"
                )
            factors[data_type][transition] = _get_correlation(
                f_prevs_all, f_nexts_all, prm,
                data_type, transition
            )
            n_transitions[data_type] += len(f_prevs_all)

    if "loads" in prm["data_types"]:
        for transition in prm["day_trans"]:
            # Household demand
            if factors["loads"][transition] is None:
                mean_residual["loads"][transition] = None
                residual_distribution_prms["loads"][transition] = None
                print(f"factors['loads'][{transition}] is None")
                continue

            f_prevs, f_nexts = [
                factors["loads"][transition][f] for f in ["f_prevs", "f_nexts"]
            ]

            assert sum(1 for f_prev in f_prevs if f_prev == 0) == 0, \
                "need to account for Nones for fit norm"
            mean_residual["loads"][transition], residual_distribution_prms["loads"][transition] \
                = _fit_residual_distribution(
                f_prevs, f_nexts, prm, "loads", transition
            )
            if mean_residual["loads"][transition] is None:
                print(f"mean_residual['loads'][{transition}] "
                      f"= {mean_residual['loads'][transition]}")
                print(f"np.shape(f_prevs) = {np.shape(f_prevs)}")
                print(f"np.shape(f_nexts) = {np.shape(f_nexts)}")

    if "EV" in prm["data_types"] \
            and _enough_data(banks["EV"], prm["weekday_type"]):
        _ev_transitions(prm, factors)

    return factors, mean_residual, residual_distribution_prms, n_transitions


def _get_month_factors_gen(n_data_type_, days_):
    f_prev_gen, f_next_gen = [], []
    for i in range(n_data_type_ - 1):
        day, next_day = [days_[i_] for i_ in [i, i + 1]]
        same_id = day["id"] == next_day["id"]
        subsequent_days = day["cum_day"] + 1 == next_day["cum_day"]
        if (
                same_id
                and subsequent_days
        ):  # record transition
            f_prev_gen.append(day["factor"])
            f_next_gen.append(next_day["factor"])

    return f_prev_gen, f_next_gen


def _scaling_factors_generation(n_data_type_, days_, prm):
    # generation
    # obtain subsequent factors
    n_transitions = 0
    f_prev_gen, f_next_gen \
        = _get_month_factors_gen(n_data_type_, days_)
    n_transitions += len(f_prev_gen)

    factors = _get_correlation(
            f_prev_gen, f_next_gen, prm, "gen"
        )

    # fit norm
    if len(f_prev_gen) > 0:
        mean_residual, residual_distribution_prms = _fit_residual_distribution(
            factors["f_prevs"], factors["f_nexts"],
            prm, "gen")

    return factors, mean_residual, residual_distribution_prms, n_transitions


def _get_factors_stats(prm, days, banks):
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

    folder_path = prm["save_path"] / "factors"
    for property_, obj in zip(
            ["f_min", "f_max", "f_mean"],
            [f_min, f_max, f_mean]
    ):
        path = folder_path / f"{property_}.pickle"
        with open(path, "wb") as file:
            pickle.dump(obj, file)

    return f_max, f_min, f_mean


def scaling_factors(prm, banks, days, n_data_type):
    """
    Obtain scaling factors corresponding to the data.

    n_data_type: len of bank per data type
    """
    _get_factors_stats(prm, days, banks)

    factors, mean_residual, residual_distribution_prms, n_transitions \
        = _scaling_factors_behaviour_types(prm, banks)

    if "gen" in prm["data_types"]:
        [
            factors["gen"], mean_residual["gen"],
            residual_distribution_prms["gen"], n_transitions["gen"]
        ] = _scaling_factors_generation(
            n_data_type["gen"], days["gen"], prm
        )

    folder_path = prm["save_path"] / "factors"
    for property_, obj in zip(
            ["n_transitions", "factors"],
            [n_transitions, factors]
    ):
        path = folder_path / f"{property_}.pickle"
        with open(path, "wb") as file:
            pickle.dump(obj, file)

    for property_, obj in zip(
            ["mean_residual", "residual_distribution_prms"],
            [mean_residual, residual_distribution_prms]
    ):
        with open(folder_path / f"{property_}.pickle", "wb") as file:
            pickle.dump(obj, file)
