"""Get scaling factor transitions characteristics."""

import math
import pickle
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import norm, pearsonr
from scipy import interpolate
import pandas as pd

from src.utils import initialise_dict


def _plot_f_next_vs_prev(prm, factors_path, f_prevs, f_nexts, label):
    if prm["plots"]:
        for stylised in [True, False]:
            fig = plt.figure()
            p95 = np.percentile(f_prevs, 95)
            plt.plot(f_prevs, f_nexts, "o", label="data", alpha=0.05)
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
            plt.xlim([- 1, p95])
            plt.ylim([- 1, p95])
            fig.savefig(factors_path / title.replace(" ", "_"))
            plt.close("all")


def _get_correlation(
        f_prevs_all: List[float],
        f_nexts_all: List[float],
        prm: dict,
        data_type: str,
        label: Optional[str] = None):

    if len(f_prevs_all) == 0:
        print(f"len(f_prevs_all) == 0 in _get_correlation")
        return None

    factors = {}
    if data_type == "car":
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

    factors_path = prm["save_other"] / "factors"

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
            j for j in range(n_intervals) if f_prevs[i] >= fs_brackets[j]
        ][-1]
        i_next = [
            j for j in range(n_intervals) if f_nexts[i] >= fs_brackets[j]
        ][-1]
        n_pos[i_prev, i_next] += 1
    for i in i_zero_to_nonzero:
        i_next = [
            j for j in range(n_intervals) if f_nexts[i] >= fs_brackets[j]
        ][-1]
        n_zero2pos[i_next] += 1

    if np.sum(n_pos) > 0:
        for i_prev in range(n_intervals):
            sum_next = sum(n_pos[i_prev])
            if sum_next > 0:
                p_pos[i_prev] = n_pos[i_prev] / sum_next
            else:
                print(f"i_prev {i_prev} sum_next {sum_next}")
                p_pos[i_prev] = np.empty((1, n_intervals))
        for i_next in range(n_intervals):
            assert isinstance(p_pos[i_next], np.ndarray)
            non0 = np.where(p_pos[:, i_next] != 0)[0]
            if len(non0) > 1:
                f = interpolate.interp1d(non0, p_pos[non0, i_next])
                new_xs = range(min(non0), max(non0))
                p_pos[new_xs, i_next] = f(new_xs)
        for i in range(len(p_pos)):
            if abs(sum(p_pos[i]) - 1) > 1e-3:
                p_pos[i] /= sum(p_pos[i])
    else:
        p_pos = n_pos

    if np.sum(n_zero2pos) > 0:
        p_zero2pos = n_zero2pos / sum(n_zero2pos) if sum(n_zero2pos) > 0 else np.zeros((1, n_intervals))
        p_zero2pos = np.reshape(p_zero2pos, (1, n_intervals))

        assert abs(np.sum(p_zero2pos) - 1) < 1e-3, f"sum(p_zero2pos) {sum(p_zero2pos)}"

    else:
        p_zero2pos = n_zero2pos

    print(f"p_zero2pos {p_zero2pos}")

    return p_pos, p_zero2pos


def _transition_intervals(
        f_prevs: List[float],
        f_nexts: List[float],
        transition: str,
        prm: int,
        data_type: str,
) -> Tuple[np.ndarray, List[float], np.ndarray, List[float]]:
    fs_brackets = np.linspace(
        min(min(f_prevs), min(f_nexts)),
        max(max(f_prevs), max(f_nexts)),
        prm["n_intervals"] + 1
    )
    mid_fs_brackets = [
        np.mean(fs_brackets[i: i + 2]) for i in range(prm["n_intervals"])
    ]
    print(transition)
    p_pos, p_zero2pos = _count_transitions(
        f_prevs, f_nexts, prm["n_intervals"], fs_brackets
    )

    labels_prob = [
        f"{data_type} non zero factor transition probabilities",
        f"{data_type} factor probabilities after a day without trips",
    ]

    for trans_prob, label_prob in zip([p_pos, p_zero2pos], labels_prob):
        # trans_prob: transition probabililites
        print(F"np.shape(trans_prob) {np.shape(trans_prob)}")
        if len(np.shape(trans_prob)) == 1:
            trans_prob = np.reshape(trans_prob, (1, len(trans_prob)))
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
                = f"{data_type} {label_prob} {transition} n_intervals {prm['n_intervals']}"
            plt.title(title)
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels)
            ax.set_xlabel("f(t + 1)")
            ax.set_ylabel("f(t)")
            fig.savefig(prm["save_other"] / "factors" / title.replace(" ", "_"))
            plt.close("all")

    return p_pos, p_zero2pos, fs_brackets, mid_fs_brackets


def _print_error_factors(ex, label, f_prevs, f_nexts):
    """Print info if Exception in computing pearsonr."""
    print(f"ex = {ex}, label = {label}")
    for factor in f_prevs + [f_nexts[-1]]:
        if type(factor) not in [float, np.float64]:
            print(f"label {label} type(factor) {type(factor)}")
            print(f"np.isnan(f) = {np.isnan(factor)}")


def sinh_archsinh_transformation(x, epsilon, delta):
    return norm.pdf(
        np.sinh(delta * np.arcsinh(x) - epsilon)
    ) * delta * np.cosh(delta * np.arcsinh(x) - epsilon)/np.sqrt(1 + np.power(x, 2))


def integral_cdf(xs, ps):
    i_valid = [
        i for i, (p, x) in enumerate(zip(ps, xs))
        if not math.isinf(p) and not math.isinf(x) and not np.isnan(p) and not np.isnan(x)
    ]
    ps = [ps[i] for i in i_valid]
    xs = [xs[i] for i in i_valid]
    return sum(
        ps[i] * (xs[i + 1] - xs[i]) if i < len(xs) - 1 else ps[i] * (xs[i] - xs[i - 1])
        for i in range(len(xs))
    )


def change_scale_norm_pdf(new_scale, old_xs, initial_norm_pdf, lb, ub):
    if not (0.95 < integral_cdf(old_xs, initial_norm_pdf) < 1.02):
        print(
            f"integral_cdf(old_xs_valid, initial_norm_pdf_valid) "
            f"{integral_cdf(old_xs, initial_norm_pdf)}"
        )
        np.save("initial_norm_pdf", initial_norm_pdf)
        np.save("old_xs", old_xs)
        sys.exit()
    assert 0.95 < integral_cdf(old_xs, initial_norm_pdf) < 1.02, \
        f"integral_cdf(old_xs, initial_norm_pdf)  {integral_cdf(old_xs, initial_norm_pdf) }"

    new_xs = [old_xs[i] * new_scale for i in range(len(old_xs))]
    new_norm_pdf_2 = [initial_norm_pdf[i] / new_scale for i in range(len(old_xs))]
    if not (0.95 < integral_cdf(new_xs, new_norm_pdf_2) < 1.02):
        np.save("new_norm_pdf_2", new_norm_pdf_2)
        np.save("new_xs", new_xs)
        print(f"len(new_xs) {len(new_xs)}")
        print(f"len(old_xs) {len(old_xs)}")
        print(f"new_scale {new_scale}")
        print(f"integral_cdf(new_xs, new_norm_pdf_2) {integral_cdf(new_xs, new_norm_pdf_2)}")
        sys.exit()
    assert 0.95 < integral_cdf(new_xs, new_norm_pdf_2) < 1.02, \
        f"l212 integral_cdf(new_xs, new_norm_pdf_2) {integral_cdf(new_xs, new_norm_pdf_2)}"

    return new_xs, new_norm_pdf_2


def _compare_factor_error_distributions(prm, errors, save_label):
    sum_log_likelihood = {}
    prms = {}
    for label, distr in prm["candidate_factor_distributions"].items():
        prms[label] = distr.fit(errors)
        sum_log_likelihood[label] = 0
        for error in errors:
            sum_log_likelihood[label] += np.log(distr.pdf(error, *prms[label]))
    extreme_p = 1e-5
    if prm['kurtosis']:
        for kurtosis_increase in [1.05, 1.25, 1.5]:
            ps = np.linspace(extreme_p, 1 - extreme_p, 10000)
            xs = [norm.ppf(p, *prms['norm']) for p in ps]
            new_xs, pdf = change_scale_norm_pdf(
                prms['norm'][1],
                xs,
                sinh_archsinh_transformation(xs, 0, kurtosis_increase),
                norm.ppf(extreme_p),
                norm.ppf(1 - extreme_p)
            )
            sum_log_likelihood[f'norm_kurtosis{kurtosis_increase}'] = 0
            for error in errors:
                i_befores = [i for i, x in enumerate(new_xs) if x <= error]
                i_afters = [i for i, x in enumerate(new_xs) if x >= error]
                i_before = i_befores[-1] if len(i_befores) > 0 else 0
                i_after = i_afters[0] if len(i_afters) > 0 else len(new_xs) - 1
                error_pdf = np.mean([pdf[i_before], pdf[i_after]])
                sum_log_likelihood[f'norm_kurtosis{kurtosis_increase}'] += np.log(error_pdf)

    for item_label, item in zip(
            ['sum_log_likelihood', 'all_distribution_prms'],
            [sum_log_likelihood, prms]
    ):
        with open(
                prm["save_other"]
                / "factors"
                / f"{item_label}_{save_label}.pickle",
                "wb"
        ) as file:
            pickle.dump(item, file)
    i_max = np.argmax(list(sum_log_likelihood.values()))
    distr_max = list(sum_log_likelihood.keys())[i_max]
    if distr_max != 'norm':
        print(f"max log likelihood for {distr_max} {save_label}")

    return distr_max


def min_max_x_to_kurtosis_pdf(xmin, xmax, norm_prms, kurtosis):
    pmin = norm.cdf(xmin * norm_prms[1])
    pmax = 1 - norm.cdf(xmax * norm_prms[1])
    extreme_p = min([pmin, 1 - pmax])
    ps = np.linspace(extreme_p, 1 - extreme_p, 1000)
    xs = [norm.ppf(p, *norm_prms) for p in ps]
    new_xs, pdf = change_scale_norm_pdf(
        norm_prms[1], xs,
        sinh_archsinh_transformation(xs, 0, kurtosis),
        norm.ppf(extreme_p), norm.ppf(1 - extreme_p)
    )

    return new_xs, pdf


def _fit_residual_distribution(f_prevs, f_nexts, prm, data_type, label=None):
    assert sum(1 for f_prev in f_prevs if f_prev is None) == 0,\
        "None f_prevs"

    f_next_sort = np.array([f_next for f_prev, f_next in sorted(zip(f_prevs, f_nexts))])
    f_prev_sort = np.array([f_prev for f_prev, f_next in sorted(zip(f_prevs, f_nexts))])

    errors = f_next_sort - f_prev_sort

    if prm["test_factor_distr"]:
        distr_str = _compare_factor_error_distributions(
            prm, errors, f"{data_type}_{label}"
        )
    else:
        distr_str = 'norm'

    # plot
    if prm["plots"]:
        p95 = np.percentile(np.where(errors > 0)[0], 95)
        if (    prm['kurtosis']
                and len(distr_str.split('_')) == 2
                and distr_str.split('_')[1][0: len('kurtosis')] == 'kurtosis'
        ):
            norm_prms = norm.fit(errors)
            kurtosis = float(distr_str.split('_')[1][len('kurtosis'):])
            factor_residuals, pdf = min_max_x_to_kurtosis_pdf(
                norm.ppf(0.01), norm.ppf(0.99), norm_prms, kurtosis
            )
            mean_residual = norm.stats(*norm_prms, moments="m")
            residual_distribution_prms = ['kurtosis'] + list(norm_prms) + [kurtosis]
        else:
            distr = prm["candidate_factor_distributions"][distr_str]
            residual_distribution_prms = ['distr_str'] + list(distr.fit(errors))
            factor_residuals = np.linspace(
                distr.ppf(0.01, *residual_distribution_prms[1:]),
                distr.ppf(0.99, *residual_distribution_prms[1:]),
                100,
            )
            pdf = distr.pdf(
                factor_residuals, *residual_distribution_prms[1:]
            )
            mean_residual = distr.stats(*residual_distribution_prms[1:], moments="m")

        fig = plt.figure()
        plt.hist(errors, density=1, alpha=0.5, label="data", bins=50)
        plt.plot(factor_residuals, pdf, label=f"{distr_str} pdf")
        assert 0.95 < integral_cdf(factor_residuals, pdf) < 1.02, \
            f"integral_cdf(factor_residuals, pdf) {integral_cdf(factor_residuals, pdf) }"

        if prm['kurtosis'] and residual_distribution_prms[0] == 'kurtosis':
            pdf_norm = norm.pdf(
                factor_residuals, *residual_distribution_prms[1: 3]
            )
            plt.plot(factor_residuals, pdf_norm, ls='--', label="norm pdf")
            if np.isnan(integral_cdf(factor_residuals, pdf_norm)):
                print(
                    f"integral_cdf(factor_residuals, pdf_norm) "
                    f"{integral_cdf(factor_residuals, pdf_norm)}"
                )
                np.save('factor_residuals', factor_residuals)
                np.save('pdf_norm', pdf_norm)
            assert 0.95 < integral_cdf(factor_residuals, pdf_norm) < 1.02, \
                f"integral_cdf(factor_residuals, pdf_norm) " \
                f"{integral_cdf(factor_residuals, pdf_norm)}"

        plt.xlim([-p95, p95])
        plt.legend(loc="upper right")
        title = (
            f"fit of {distr_str} distribution to error around "
            f"perfect correlation {data_type} {label}"
        )
        plt.title(title)
        fig.savefig(prm["save_other"] / "factors" / title.replace(" ", "_").replace('.', '_'))
        plt.close("all")

    return mean_residual, residual_distribution_prms


def _enough_data(banks_, weekday_type):
    day_types = all(day_type in banks_ for day_type in weekday_type)
    return not day_types or sum(banks_[day_type] is None for day_type in weekday_type) == 0


def _ev_transitions(prm, factors, data_type):
    p_pos, p_zero2pos, fs_brackets, mid_fs_brackets = [{} for _ in range(4)]

    transitions_specific = prm["day_trans"][0] in factors[data_type]
    if transitions_specific:
        for transition in prm["day_trans"]:
            if factors[data_type][transition] is None:
                print(F"factors[{data_type}][{transition}] is None")
            else:
                print(f"data_type {data_type} transition {transition}")
                f_prevs, f_nexts = [
                    factors[data_type][transition][f] for f in ["f_prevs", "f_nexts"]
                ]
                # car transitions
                [p_pos[transition], p_zero2pos[transition],
                 fs_brackets[transition], mid_fs_brackets[transition]] \
                    = _transition_intervals(f_prevs, f_nexts, transition, prm, data_type)

        all_f_prevs, all_f_nexts = np.array([]), np.array([])

        for transition in prm["day_trans"]:
            all_f_prevs = np.concatenate((all_f_prevs, factors[data_type][transition]['f_prevs']))
            all_f_nexts = np.concatenate((all_f_nexts, factors[data_type][transition]['f_nexts']))
    else:
        all_f_prevs = factors[data_type]['f_prevs']
        all_f_nexts = factors[data_type]['f_nexts']
    [
        p_pos['all'], p_zero2pos['all'],
        fs_brackets['all'], mid_fs_brackets['all']
    ] = _transition_intervals(all_f_prevs, all_f_nexts, 'all', prm, data_type)
    dictionaries = p_pos, p_zero2pos, fs_brackets, mid_fs_brackets
    labels = ["p_pos", "p_zero2pos", "fs_brackets", "mid_fs_brackets"]
    for dictionary, label in zip(dictionaries, labels):
        with open(
                prm["save_hedge"] / "factors" / f"{data_type}_{label}.pickle", "wb"
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
            print(f"not enough data for {data_type} to get factors")
            continue
        factors[data_type] = initialise_dict(prm["day_trans"], "empty_dict")
        for transition in prm["day_trans"]:
            if len(banks[data_type][transition]["f0"]) == 0:
                print(F"len(banks[{data_type}][{transition}]['f0']) == 0")
            f_prevs_all, f_nexts_all = [
                np.array(banks[data_type][transition][f"f{i_f}"])
                for i_f in range(2)
            ]
            if len(f_prevs_all) == 0:
                print(
                    f"{data_type} transition {transition} "
                    f"len f_prevs_all == 0"
                )
            factors[data_type][transition] = _get_correlation(
                f_prevs_all, f_nexts_all, prm,
                data_type, transition
            )
            if factors[data_type][transition] is None:
                print(
                    f"transition {transition} factors[data_type][transition] is None in _scaling_factors_behaviour_types"
                )
                continue
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
                = _fit_residual_distribution(f_prevs, f_nexts, prm, "loads", transition)
            if mean_residual["loads"][transition] is None:
                print(f"mean_residual['loads'][{transition}] "
                      f"= {mean_residual['loads'][transition]}")
                print(f"np.shape(f_prevs) = {np.shape(f_prevs)}")
                print(f"np.shape(f_nexts) = {np.shape(f_nexts)}")

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
    else:
        print(f"_scaling_factors_generation len(f_prev_gen) {len(f_prev_gen)}")

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

    folder_path = prm["save_hedge"] / "factors"
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
    print("scaling_factors")
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
    for data_type in prm["data_types"]:
        print(data_type)
        if _enough_data(banks[data_type], prm["weekday_type"]):
            _ev_transitions(prm, factors, data_type)




    path = prm["save_other"] / "factors" / "n_transitions.pickle"
    with open(path, "wb") as file:
        pickle.dump(n_transitions, file)

    folder_path = prm["save_hedge"] / "factors"
    for property_, obj in zip(
        ["mean_residual", "factors", "residual_distribution_prms"],
        [mean_residual, factors, residual_distribution_prms],
    ):
        path = folder_path / f"{property_}.pickle"
        with open(path, "wb") as file:
            pickle.dump(obj, file)
