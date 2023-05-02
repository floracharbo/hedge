"""Get scaling factor transitions characteristics."""

import math
import pickle
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import interpolate
from scipy.stats import norm, pearsonr

from src.utils import f_to_interval, get_cmap, initialise_dict, save_fig


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
            save_path = factors_path / title.replace(" ", "_")
            save_fig(fig, prm, save_path)
            plt.close("all")


def _get_correlation(
        f_prevs_all: List[float],
        f_nexts_all: List[float],
        prm: dict,
        data_type: str,
        label: Optional[str] = None):

    if len(f_prevs_all) == 0:
        print("len(f_prevs_all) == 0 in _get_correlation")
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
    with open(factors_path / f"corr_{data_type}_{label}.pickle", "wb") as f:
        pickle.dump(factors["corr"], f)
    factors["f0of2"] = f_prevs
    factors["f1of2"] = f_nexts

    return factors


def _interpolate_missing_p_pos_2d(
        p_pos, mid_fs_brackets, data_type, transition, prm, plot=True
):
    non0 = np.where((p_pos != 0) & (~np.isnan(p_pos)))
    if len(non0[0]) > 3:
        points = [
            [mid_fs_brackets[non0[0][i]], mid_fs_brackets[non0[1][i]]]
            for i in range(len(non0[0]))
        ]
        values = p_pos[non0]
        grid_y, grid_x = np.meshgrid(mid_fs_brackets, mid_fs_brackets)
        p_pos[p_pos == 0] = np.nan
        interpolated_p_pos = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')
        for i_prev in range(len(p_pos)):
            if (
                    np.sum(interpolated_p_pos[i_prev]) != 0
                    and abs(sum(interpolated_p_pos[i_prev]) - 1) > 1e-3
            ):
                interpolated_p_pos[i_prev] /= sum(interpolated_p_pos[i_prev])
        if plot:
            img = [None, None]
            fig, axs = plt.subplots(2, figsize=(5, 10))
            img[0] = axs[0].imshow(p_pos, origin='lower', norm=LogNorm(vmin=1e-4), cmap=get_cmap())
            axs[0].title.set_text('Original')
            img[1] = axs[1].imshow(
                interpolated_p_pos, origin='lower', norm=LogNorm(vmin=1e-4), cmap=get_cmap()
            )
            axs[1].title.set_text('Linear interpolation')
            for i in range(2):
                axs[i] = _add_tick_labels_heatmap(axs[i], mid_fs_brackets)
                plt.colorbar(img[i], ax=axs[i])
            title = f"interpolate_2d_p_pos_{data_type}_{transition}"
            save_path = prm['save_other'] / "factors"
            if data_type == 'car':
                with open(save_path / f"p_pos_{data_type}_{transition}.pickle", "wb") as f:
                    pickle.dump(p_pos, f)
                with open(
                        save_path / f"interpolated_p_pos_{data_type}_{transition}.pickle", "wb"
                ) as f:
                    pickle.dump(interpolated_p_pos, f)
            save_path = prm['save_other'] / "factors" / title.replace(" ", "_")
            save_fig(fig, prm, save_path)
            plt.close('all')
        interpolated_p_pos[np.isnan(interpolated_p_pos)] = 0
    else:
        print(
            f"only {len(non0[0])} nonzero points "
            f"for {data_type}, {transition}: cannot perform 2d interpolation"
        )
        interpolated_p_pos = p_pos

    return interpolated_p_pos


def _count_transitions(
    data_type,
    consecutive_factors,
    prm: int,
    fs_brackets: List[float],
    mid_fs_brackets: List[float],
    n_consecutive_days: int = 2,
    transition: str = "",
):
    print(f"{data_type} {transition}")
    shape = tuple(prm['n_intervals'] for _ in range(n_consecutive_days))
    n_pos, p_pos = [np.zeros(shape) for _ in range(2)]
    n_zero2pos = np.zeros(prm['n_intervals'])
    if n_consecutive_days == 2:
        f_prevs, f_nexts = consecutive_factors[0], consecutive_factors[1]
        i_non_zeros = [
            i for i, (f_prev, f_next) in enumerate(zip(f_prevs, f_nexts))
            if f_prev is not None
            and f_prev > 0
            and f_next is not None
            and f_next > 0
        ]
        i_zero_to_nonzero = [
            i for i, (f_prev, f_next) in enumerate(zip(f_prevs, f_nexts))
            if (f_prev is None or f_prev == 0)
            and f_next is not None
            and f_next > 0
        ]
    else:
        i_non_zeros = list(range(len(consecutive_factors[0])))
        i_zero_to_nonzero = []

    if not (data_type == 'car' and transition == 'we2wd'):
        assert len(i_non_zeros) > 0, \
            f"len(i_non_zeros) == 0, data_type {data_type} " \
            f"transition {transition} n_consecutive_days {n_consecutive_days}"

    for i in i_non_zeros:
        idx = tuple(
            f_to_interval(f, fs_brackets)
            for f in consecutive_factors[:, i]
        )
        n_pos[idx] += 1
    if not (data_type == 'car' and transition == 'we2wd'):
        assert np.sum(n_pos) > 0, \
            f"np.sum(n_pos) == 0, data_type {data_type} " \
            f"transition {transition} n_consecutive_days {n_consecutive_days}"

    for i in i_zero_to_nonzero:
        i_next = [
            j for j in range(prm['n_intervals']) if f_nexts[i] >= fs_brackets[j]
        ][-1]
        n_zero2pos[i_next] += 1
    if not (data_type == 'car' and transition == 'we2wd'):
        if n_consecutive_days == 2:
            for i_prev in range(prm['n_intervals']):
                sum_next = sum(n_pos[i_prev])
                if sum_next > 0:
                    p_pos[i_prev] = n_pos[i_prev] / sum_next
                else:
                    print(f"i_prev {i_prev} sum_next {sum_next}")
                    p_pos[i_prev] = np.zeros((1, prm['n_intervals']))
            p_pos = _interpolate_missing_p_pos_2d(
                p_pos, mid_fs_brackets, data_type, transition, prm
            )

        elif n_consecutive_days == 3:
            for i_prev in range(prm['n_intervals']):
                for i_prev2 in range(prm['n_intervals']):
                    sum_next = sum(n_pos[i_prev, i_prev2])
                    if sum_next > 0:
                        p_pos[i_prev, i_prev2] = n_pos[i_prev, i_prev2] / sum_next
                    else:
                        p_pos[i_prev, i_prev2] = np.zeros((1, prm['n_intervals']))

                    if not np.all(p_pos >= 0):
                        print(f"184 p_pos[{i_prev}, {i_prev2}] = {p_pos[i_prev, i_prev2]}")
                        print(f"n_pos[{i_prev}, {i_prev2}]= {n_pos[i_prev, i_prev2]}")
                    assert np.all(p_pos >= 0), f"{data_type} {transition}"
                    # non0 = np.where(p_pos[i_prev, i_prev2] != 0)[0]
                    # if len(non0) > 1:
                    #     interpolate_function = interpolate.interp1d(
                    #         non0, p_pos[i_prev, i_prev2, non0]
                    #     )
                    #     new_xs = range(min(non0), max(non0))
                    #     p_pos[i_prev, i_prev2, new_xs] = interpolate_function(new_xs)
                    #
                    # sum_p_pos = sum(p_pos[i_prev, i_prev2])
                    # if sum_p_pos != 0 and abs(sum_p_pos - 1) > 1e-3:
                    #     p_pos[i_prev, i_prev2] /= sum_p_pos

                p_pos[i_prev] = _interpolate_missing_p_pos_2d(
                    p_pos[i_prev], mid_fs_brackets, data_type, f"{transition}_3d_i_prev{i_prev}",
                    prm, plot=False
                )

    else:
        print(f"implement n_consecutive_days {n_consecutive_days}")

    if not (data_type == 'car' and transition == 'we2wd'):
        assert np.all(p_pos >= 0), f"not np.all(p_pos >= 0) {data_type} {transition}"
    if n_consecutive_days == 2 and np.sum(n_zero2pos) > 0:
        print(f"np.sum(n_zero2pos) {np.sum(n_zero2pos)} > 0")
        p_zero2pos = n_zero2pos / sum(n_zero2pos) if sum(n_zero2pos) > 0 \
            else np.zeros((1, prm['n_intervals']))
        p_zero2pos = np.reshape(p_zero2pos, (1, prm['n_intervals']))

        assert abs(np.sum(p_zero2pos) - 1) < 1e-3, f"sum(p_zero2pos) {sum(p_zero2pos)}"

    else:
        if n_consecutive_days == 2:
            print(f"n_consecutive_days = 2 and np.sum(n_zero2pos) {np.sum(n_zero2pos)} == 0")
        p_zero2pos = n_zero2pos

    np.save(
        prm['save_other'] / 'factors'
        / f"n_transitions_{data_type}_{transition}_{n_consecutive_days}_consecutive_days",
        np.sum(n_pos)
    )
    return p_pos, p_zero2pos


def _add_tick_labels_heatmap(ax, mid_fs_brackets):
    tick_locations = [i * 2 for i in range(int(len(mid_fs_brackets) / 2))]
    if not len(mid_fs_brackets) - 1 in tick_locations:
        tick_locations[-1] = len(mid_fs_brackets) - 1
    tick_labels = [round(mid_fs_brackets[i], 1) for i in tick_locations]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)
    return ax


def _transition_intervals(
        consecutive_factors,
        transition: str,
        prm: int,
        data_type: str,
        n_consecutive_days: int,
) -> Tuple[np.ndarray, List[float], np.ndarray, List[float]]:
    if transition == 'we2wd' and data_type == 'car':
        return None, None, None, None

    consecutive_factors = np.array(consecutive_factors)
    consecutive_factors_positives = consecutive_factors[consecutive_factors > 0]
    factors_brackets = consecutive_factors_positives if n_consecutive_days == 2 \
        else consecutive_factors
    if prm['brackets_definition'] == 'percentile':
        fs_brackets = np.percentile(
            factors_brackets,
            np.linspace(0, 100, prm["n_intervals"] + 1)
        )
    elif prm['brackets_definition'] == 'linspace':
        fs_brackets = np.linspace(
            np.min(factors_brackets), np.max(factors_brackets), prm["n_intervals"] + 1
        )

    mid_fs_brackets = [
        np.mean(fs_brackets[i: i + 2]) for i in range(prm["n_intervals"])
    ]

    p_pos, p_zero2pos = _count_transitions(
        data_type, consecutive_factors, prm, fs_brackets, mid_fs_brackets,
        n_consecutive_days=n_consecutive_days, transition=transition,
    )
    labels_prob = {
        'p_pos': f"{data_type} non zero factor transition probabilities",
        'p_zero2pos': f"{data_type} factor probabilities after a day without trips",
    }

    for trans_prob, label_prob in zip([p_pos, p_zero2pos], labels_prob.keys()):
        # trans_prob: transition probabilities
        if len(np.shape(trans_prob)) == 1:
            trans_prob = np.reshape(trans_prob, (1, len(trans_prob)))
        # trans_prob[(trans_prob == 0) | (np.isnan(trans_prob))] = 1e-5
        if not ((trans_prob >= 0).all()):
            print(f"label_prob {label_prob} not (trans_prob >= 0).all()")
            if not (data_type == 'car' and transition == 'we2wd'):
                assert np.sum(trans_prob) > 0, "error not np.sum(trans_prob) > 0"

        if prm["plots"] and n_consecutive_days == 2:
            fig, ax = plt.subplots()
            img = ax.imshow(trans_prob, norm=LogNorm(vmin=1e-4), origin='lower', cmap=get_cmap())
            ax = _add_tick_labels_heatmap(ax, mid_fs_brackets)
            plt.colorbar(img, ax=ax)
            title = \
                f"{data_type} {labels_prob[label_prob]} {transition} " \
                f"n_intervals {prm['n_intervals']} " \
                f"brackets_definition {prm['brackets_definition']}"
            plt.title(title)
            # ax.set_xticklabels(tick_labels)
            # ax.set_yticklabels(tick_labels)
            ax.set_xlabel("f(t)")
            ax.set_ylabel("f(t + 1)")
            save_path = prm["save_other"] / "factors" / title.replace(" ", "_")
            save_fig(fig, prm, save_path)
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
    ps = ps[i_valid]
    xs = xs[i_valid]
    return sum(
        ps[i] * (xs[i + 1] - xs[i]) if i < len(xs) - 1 else ps[i] * (xs[i] - xs[i - 1])
        for i in i_valid
    )


def change_scale_norm_pdf(new_scale, old_xs, initial_norm_pdf, lb, ub):
    assert 0.95 < integral_cdf(old_xs, initial_norm_pdf) < 1.02, \
        f"integral_cdf(old_xs, initial_norm_pdf)  {integral_cdf(old_xs, initial_norm_pdf) }"

    new_xs = np.array(old_xs) * new_scale
    new_norm_pdf_2 = np.array(initial_norm_pdf) / new_scale
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
        if (
                prm['kurtosis']
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
            residual_distribution_prms = [distr_str] + list(distr.fit(errors))
            factor_residuals = np.linspace(
                distr.ppf(0.01, *residual_distribution_prms[1:]),
                distr.ppf(0.99, *residual_distribution_prms[1:]),
                100,
            )
            pdf = distr.pdf(
                factor_residuals, *residual_distribution_prms[1:]
            )
            mean_residual = distr.stats(*residual_distribution_prms[1:], moments="m")

        if prm['plots']:
            fig = plt.figure()
            plt.hist(errors, density=1, alpha=0.5, label="data", bins=50)
            plt.plot(factor_residuals, pdf, label=f"{distr_str} pdf")
            save_path = prm['save_other'] / 'factors' / \
                f'hist_errors_vs_pdf_{distr_str}_{data_type}_{label}'
            save_fig(fig, prm, save_path)
        assert 0.9 < integral_cdf(factor_residuals, pdf) < 1.02, \
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
        save_path = prm["save_other"] / "factors" / title.replace(" ", "_").replace('.', '_')
        save_fig(fig, prm, save_path)
        plt.close("all")

    return mean_residual, residual_distribution_prms


def _enough_data(banks_, weekday_type):
    day_types = all(day_type in banks_ for day_type in weekday_type)
    return not day_types or sum(banks_[day_type] is None for day_type in weekday_type) == 0


def _ev_transitions(prm, factors, data_type, n_consecutive_days):
    p_pos, p_zero2pos, fs_brackets, mid_fs_brackets = [{} for _ in range(4)]

    transitions_specific = prm["day_trans"][0] in factors[data_type]
    if transitions_specific:
        for transition in prm["day_trans"]:
            if factors[data_type][transition] is None:
                print(F"factors[{data_type}][{transition}] is None")
            else:
                print(f"data_type {data_type} transition {transition}")
                consecutive_factors = np.array([
                    factors[data_type][transition][f"f{f}of{n_consecutive_days}"]
                    for f in range(n_consecutive_days)
                ])
                # car transitions
                [p_pos[transition], p_zero2pos[transition],
                 fs_brackets[transition], mid_fs_brackets[transition]] = _transition_intervals(
                    consecutive_factors, transition, prm, data_type, n_consecutive_days
                )

        all_consecutive_factors = [[] for f in range(n_consecutive_days)]

        for transition in prm["day_trans"]:
            for f in range(n_consecutive_days):
                all_consecutive_factors[f].extend(
                    factors[data_type][transition][f"f{f}of{n_consecutive_days}"]
                )
    else:
        try:
            all_consecutive_factors = [
                factors[data_type][f"f{f}of{n_consecutive_days}"] for f in range(n_consecutive_days)
            ]
        except Exception as ex:
            print(f"{ex}: factors[{data_type}].keys() = {factors[data_type].keys()}")
            np.save('factors_errors', factors)

    all_consecutive_factors = np.array(all_consecutive_factors)
    [
        p_pos['all'], p_zero2pos['all'],
        fs_brackets['all'], mid_fs_brackets['all']
    ] = _transition_intervals(all_consecutive_factors, 'all', prm, data_type, n_consecutive_days)

    return p_pos, p_zero2pos, fs_brackets, mid_fs_brackets


def _get_residuals_loads(prm, factors, n_consecutive_days=2):
    mean_residual, residual_distribution_prms = {}, {}
    for transition in prm["day_trans"]:
        print(f"loads {transition}")
        # Household demand
        if factors["loads"][transition] is None:
            mean_residual[transition] = None
            residual_distribution_prms["loads"][transition] = None
            print(f"factors['loads'][{transition}] is None")
            continue

        consecutive_factors = np.array([
            factors["loads"][transition][f"f{i_f}of{n_consecutive_days}"]
            for i_f in range(n_consecutive_days)
        ])
        print(f"np.shape(consecutive_factors) {np.shape(consecutive_factors)}")
        f_prevs = consecutive_factors[0]
        f_nexts = consecutive_factors[1]
        assert sum(1 for f_prev in f_prevs if f_prev == 0) == 0, \
            "need to account for Nones for fit norm"
        mean_residual[transition], residual_distribution_prms[transition] \
            = _fit_residual_distribution(f_prevs, f_nexts, prm, "loads", transition)
        if mean_residual[transition] is None:
            print(f"mean_residual['loads'][{transition}] "
                  f"= {mean_residual[transition]}")
            print(f"np.shape(f_prevs) = {np.shape(f_prevs)}")
            print(f"np.shape(f_nexts) = {np.shape(f_nexts)}")

    return mean_residual, residual_distribution_prms


def _scaling_factors_behaviour_types(
    prm: dict,
    banks: dict,
    n_consecutive_days: int,
    factors: dict,
) -> Tuple[dict, dict, dict, dict]:
    mean_residual, residual_distribution_prms = [
        initialise_dict(prm["CLNR_types"], "empty_dict") for _ in range(2)
    ]
    n_transitions = initialise_dict(prm["data_types"], "zero")

    for data_type in prm["behaviour_types"]:
        if not _enough_data(banks[data_type], prm["weekday_type"]):
            print(f"not enough data for {data_type} to get factors")
            continue
        for transition in prm["day_trans"]:
            if len(banks[data_type][transition][f"f0of{n_consecutive_days}"]) == 0:
                print(f"len(banks[{data_type}][{transition}]['f0of{n_consecutive_days}']) == 0")
            all_consecutive_factors = np.array([
                np.array(banks[data_type][transition][f"f{i_f}of{n_consecutive_days}"])
                for i_f in range(n_consecutive_days)
            ])
            f_prevs_all, f_nexts_all = all_consecutive_factors[0:2]
            if n_consecutive_days == 2:
                factors[data_type][transition] = _get_correlation(
                    f_prevs_all, f_nexts_all, prm, data_type, transition
                )
            elif n_consecutive_days == 3:
                for i in range(3):
                    label = f"f{i}of{n_consecutive_days}"
                    factors[data_type][transition][label] = banks[data_type][transition][label]
            if factors[data_type][transition] is None:
                print(
                    f"transition {transition} factors[data_type][transition] is None "
                    f"in _scaling_factors_behaviour_types"
                )
                continue
            n_transitions[data_type] += len(f_prevs_all)

    if n_consecutive_days == 2 and "loads" in prm["data_types"]:
        mean_residual['loads'], residual_distribution_prms['loads'] = _get_residuals_loads(
            prm, factors
        )

    if n_consecutive_days == 3 and 'gen' in factors:
        for i in range(3):
            label = f"f{i}of{n_consecutive_days}"
            factors['gen'][label] = []
            for m in range(12):
                factors['gen'][label].extend(banks['gen'][m][label])

    return factors, mean_residual, residual_distribution_prms, n_transitions


def _get_month_factors_gen(n_data_type_, days_, n_consecutive_days):
    subsequent_factors = []
    for i in range(n_data_type_ - (n_consecutive_days - 1)):
        consecutive_days = [days_[i_] for i_ in range(i, i + n_consecutive_days)]
        same_id = all(
            consecutive_days[d]["id"] == consecutive_days[0]["id"]
            for d in range(n_consecutive_days)
        )
        subsequent_days = all(
            consecutive_days[0]["cum_day"] + d == consecutive_days[d]["cum_day"]
            for d in range(n_consecutive_days)
        )
        if same_id and subsequent_days:
            subsequent_factors.append(
                [consecutive_days[d]['factor'] for d in range(n_consecutive_days)]
            )
    subsequent_factors = np.array(subsequent_factors)

    return subsequent_factors


def _scaling_factors_generation(n_data_type_, days_, prm, n_consecutive_days):
    # generation
    # obtain subsequent factors
    n_transitions = 0
    subsequent_factors = _get_month_factors_gen(n_data_type_, days_, n_consecutive_days)
    n_transitions += len(subsequent_factors)

    if n_consecutive_days == 2:
        f_prev_gen = subsequent_factors[:, 0]
        f_next_gen = subsequent_factors[:, 1]
        factors = _get_correlation(
            f_prev_gen, f_next_gen, prm, "gen", n_consecutive_days
        )

        # fit norm
        if len(f_prev_gen) > 0:
            mean_residual, residual_distribution_prms = _fit_residual_distribution(
                factors["f0of2"], factors["f1of2"],
                prm, "gen")
        else:
            print(f"_scaling_factors_generation len(f_prev_gen) {len(f_prev_gen)}")
    else:
        mean_residual, residual_distribution_prms = None, None
        factors = {}
        for i in range(n_consecutive_days):
            label = f"f{i}of{n_consecutive_days}"
            factors[label] = subsequent_factors[:, i]

    return factors, mean_residual, residual_distribution_prms, n_transitions


def _get_factors_stats(prm, days, banks):
    # Initialise dictionaries
    f_max, f_min, f_mean = [initialise_dict(prm["data_types"]) for _ in range(3)]

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
    factors = initialise_dict(prm["data_types"], "empty_dict")
    for data_type in prm['behaviour_types']:
        factors[data_type] = initialise_dict(prm["day_trans"], "empty_dict")

    _get_factors_stats(prm, days, banks)
    for n_consecutive_days in [3, 2]:
        factors, mean_residual, residual_distribution_prms, n_transitions \
            = _scaling_factors_behaviour_types(prm, banks, n_consecutive_days, factors)
        p_pos, p_zero2pos, fs_brackets, mid_fs_brackets = [{} for _ in range(4)]
        for data_type in prm["data_types"]:
            print(data_type)
            if _enough_data(banks[data_type], prm["weekday_type"]):
                if data_type == 'gen':
                    [
                        factors["gen"], mean_residual["gen"],
                        residual_distribution_prms["gen"], n_transitions["gen"]
                    ] = _scaling_factors_generation(
                        n_data_type["gen"], days["gen"], prm, n_consecutive_days
                    )
                [
                    p_pos[data_type], p_zero2pos[data_type],
                    fs_brackets[data_type], mid_fs_brackets[data_type]
                ] = _ev_transitions(
                    prm, factors, data_type, n_consecutive_days=n_consecutive_days
                )
        dictionaries = [p_pos, p_zero2pos, fs_brackets, mid_fs_brackets]
        labels = ["p_pos", "p_zero2pos", "fs_brackets", "mid_fs_brackets"]
        for dictionary, label in zip(dictionaries, labels):
            with open(
                    prm["save_hedge"]
                    / "factors"
                    / f"{label}_n_consecutive_days{n_consecutive_days}_brackets_definition_"
                      f"{prm['brackets_definition']}.pickle", "wb"
            ) as file:
                pickle.dump(dictionary, file)

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
