"""
Home energy data generator (HEDGE).

Generates subsequent days of EV, PV generation and home electricity data
for a given number of homes.

The main method is 'make_next_day', which generates new day of data
(EV, dem, gen profiles), calling other methods as needed.
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import yaml
from scipy.stats import gamma

from utils import initialise_dict


class HEDGE:
    """
    Home energy data generator (HEDGE).

    Generates subsequent days of EV, gen and dem data for
    a given number of homes.
    """

    def __init__(
        self,
        n_homes: int,
        factors0: dict = None,
        clusters0: dict = None,
    ):
        """Initialise HEDGE object and initial properties."""
        # update object properties
        self.factors = factors0
        self.clusters = clusters0
        self.labels_day = ["wd", "we"]
        self.n_homes = n_homes

        # load input data
        prm, config = self._load_inputs()

        # update date and time information
        self.date = datetime(2020, config["month"], 1)
        if config["day_week0"] is not None:
            while self.date.weekday() != config["day_week0"]:
                self.date += timedelta(days=1)

        self._init_factors_clusters()

        self.profs = self._load_profiles(prm)

        # number of time steps per day
        if "dem" in self.profs:
            self.n_steps = len(self.profs["dem"]["wd"][0][0])
        elif "EV" in self.profs:
            self.n_steps = len(self.profs["EV"]["cons"]["wd"][0][0])
        else:
            self.n_steps = len(self.profs["gen"][0][0])

        self.save_day_path = Path(config["results_save_HEDGE"])

    def make_next_day(self, plotting=False) -> dict:
        """Generate new day of data (EV, gen, dem profiles)."""
        self.date += timedelta(days=1)
        day_type, transition = self._transition_type()
        prev_clusters = self.clusters.copy()

        # obtain scaling factors
        factors, interval_f_ev = self._next_factors(transition, prev_clusters)

        # obtain clusters
        clusters = self._next_clusters(transition, prev_clusters)

        # obtain profile indexes
        i_month = self.date.month - 1
        i_profiles = {}
        for data_type in self.data_types:
            i_profiles[data_type] = self._select_profiles(
                data_type, day_type, clusters=clusters, i_month=i_month
            )

        # obtain days
        day = {}
        if "dem" in self.data_types:
            day["dem"] = [
                [p * factors["dem"][home]
                 for p in self.profs["dem"][day_type][
                     clusters["dem"][home]][i_profiles["dem"][home]]]
                for home in range(self.n_homes)]
        if "gen" in self.data_types:
            gen_profs = self.profs["gen"][i_month]
            day["gen"] = [
                [
                    p * factors["gen"][home]
                    for p in gen_profs[i_profiles["gen"][home]]
                ]
                for home in range(self.n_homes)
            ]
        if "EV" in self.data_types:
            day["loads_EV"] = [
                [p * factors["EV"][home]
                 for p in self.profs["EV"]["cons"][day_type][
                     clusters["EV"][home]][i_profiles["EV"][home]]]
                for home in range(self.n_homes)
            ]

            # check loads EV are consistent with maximum battery load
            interval_f_ev, factors, day, i_ev = self._adjust_max_ev_loads(
                day, interval_f_ev, factors, transition, clusters,
                day_type, i_profiles["EV"]
            )

            day["avail_EV"] = [
                self.profs["EV"]["avail"][day_type][
                    clusters["EV"][home]][i_profiles["EV"][home]]
                for home in range(self.n_homes)
            ]

        self.factors = factors
        self.clusters = clusters

        if plotting:
            self._plotting_profiles(day)

        return day

    def _import_dem(self, prm, config):
        for transition in prm["day_trans"]:
            if self.gamma_prms["dem"][transition] is None:
                continue
            self.gamma_prms["dem"][transition] \
                = list(self.gamma_prms["dem"][transition])
            self.gamma_prms["dem"][transition][2] *= config["f_std_share"]
        self.select_cdfs["dem"] = {}
        for day_type in prm["weekday_type"]:
            self.select_cdfs["dem"][day_type] = [
                min_cdf + config["clust_dist_share"] * (max_cdf - min_cdf)
                for min_cdf, max_cdf in zip(
                    self.min_cdfs["dem"][day_type],
                    self.max_cdfs["dem"][day_type]
                )
            ]

    def _load_inputs(self):
        # load inputs
        prm = yaml.safe_load(open("inputs/parameters.yaml"))
        config = yaml.safe_load(open("inputs/config_hedge.yaml"))

        # add relevant parameters to object properties
        self.data_types = config["data_types"]
        self.behaviour_types \
            = [data_type
               for data_type in self.data_types if data_type != "gen"]
        self.bat = prm["bat"]

        # update paths
        self.save_path = Path(config["results_path_HEDGE"])

        # possible types of transition between week day types (week/weekend)
        prm["day_trans"] = []
        for prev_day in prm["weekday_type"]:
            for next_day in prm["weekday_type"]:
                prm["day_trans"].append(f"{prev_day}2{next_day}")

        # general inputs with all data types
        folder_path = self.save_path / "factors"
        for property_ in ["f_min", "f_max", "f_mean"]:
            path = folder_path / f"{property_}.pickle"
            with open(path, "rb") as file:
                self.__dict__[property_] = pickle.load(file)

        for property_ in ["mean_residual", "gamma_prms"]:
            with open(folder_path / f"{property_}.pickle", "rb") as file:
                self.__dict__[property_] = pickle.load(file)

        path_clusters = self.save_path / "clusters"
        for property_ in ["p_clus", "p_trans", "min_cdfs", "max_cdfs"]:
            path = path_clusters / f"{property_}.pickle"
            with open(str(path), "rb") as file:
                self.__dict__[property_] = pickle.load(file)

        with open(path_clusters / "n_clusters.pickle", "rb") as file:
            prm["n_clusters"] = pickle.load(file)
        self.n_all_clusters_ev = prm["n_clusters"]["EV"] + 1

        self.select_cdfs = {}
        # household demand-specific inputs
        if "dem" in self.data_types:
            self._import_dem(prm, config)

        # EV-specific inputs
        if "EV" in self.data_types:
            for property_ in ["p_pos", "p_zero2pos", "xs", "mid_xs"]:
                path = folder_path / f"EV_{property_}.pickle"
                with open(path, "rb") as file:
                    self.__dict__[property_] = pickle.load(file)

        # PV generation-specific inputs
        if "gen" in self.data_types:
            self.gamma_prms["gen"][2] *= config["f_std_share"]

            self.select_cdfs["gen"] \
                = [min_cdf + config["clust_dist_share"] * (max_cdf - min_cdf)
                   for min_cdf, max_cdf in zip(
                    self.min_cdfs["gen"], self.max_cdfs["gen"])]

        return prm, config

    def _init_factors_clusters(self):
        day_type, transition = self._transition_type()

        if self.factors is None:
            self.factors = {}

            if "dem" in self.data_types:
                self.factors["dem"] = [
                    self.f_mean["dem"]
                    + gamma.ppf(
                        np.random.rand(),
                        *self.gamma_prms["dem"][transition])
                    for _ in range(self.n_homes)
                ]

            if "gen" in self.data_types:
                i_month = self.date.month - 1
                self.factors["gen"] = [
                    self.f_mean["gen"][i_month]
                    + gamma.ppf(
                        np.random.rand(),
                        *self.gamma_prms["gen"][i_month])
                    for _ in range(self.n_homes)]

            if "EV" in self.data_types:
                randoms = np.random.rand(self.n_homes)
                self.factors["EV"] = [
                    self._ps_rand_to_choice(
                        self.p_zero2pos[transition],
                        randoms[home]
                    )
                    for home in range(self.n_homes)
                ]

        if self.clusters is None:
            self.clusters = {}
            for data in self.behaviour_types:
                self.clusters[data] \
                    = [self._ps_rand_to_choice(
                        self.p_clus[data][day_type], np.random.rand())
                        for _ in range(self.n_homes)]

    def _next_factors(self, transition, prev_clusters):
        prev_factors = self.factors.copy()
        factors = initialise_dict(self.data_types)
        random_f \
            = [[np.random.rand() for _ in range(self.n_homes)]
               for _ in self.data_types]
        interval_f_ev = []

        for home in range(self.n_homes):
            if "EV" in self.data_types:
                current_interval \
                    = [i for i in range(len(self.xs[transition]) - 1)
                       if self.xs[transition][i]
                       <= prev_factors["EV"][home]][-1]
                if prev_clusters["EV"][home] == self.n_all_clusters_ev - 1:
                    # no trip day
                    probabilities = self.p_zero2pos[transition]
                else:
                    probabilities = self.p_pos[transition][current_interval]
                interval_f_ev.append(
                    self._ps_rand_to_choice(
                        probabilities,
                        random_f[0][home],
                    )
                )
                factors["EV"].append(
                    self.mid_xs[transition][int(interval_f_ev[home])]
                )

            if "gen" in self.data_types:
                i_month = self.date.month - 1
                # factor for generation
                # without differentiation between day types
                delta_f = gamma.ppf(
                    random_f[1][home],
                    *self.gamma_prms["gen"][i_month]
                )
                factors["gen"].append(
                    prev_factors["gen"][home]
                    + delta_f
                    - self.mean_residual["gen"][i_month]
                )
                factors["gen"][home] = min(
                    max(self.f_min["gen"][i_month], factors["gen"][home]),
                    self.f_max["gen"][i_month]
                )

            if "dem" in self.data_types:
                # factor for demand - differentiate between day types
                delta_f = gamma.ppf(
                    random_f[2][home],
                    *list(self.gamma_prms["dem"][transition])
                )
                factors["dem"].append(
                    prev_factors["dem"][home]
                    + delta_f
                    - self.mean_residual["dem"][transition]
                )

            for data_type in self.behaviour_types:
                factors[data_type][home] = min(
                    max(self.f_min[data_type], factors[data_type][home]),
                    self.f_max[data_type]
                )

        return factors, interval_f_ev

    def _next_clusters(self, transition, prev_clusters):
        clusters = initialise_dict(self.behaviour_types)

        random_clus = [
            [np.random.rand() for _ in range(self.n_homes)]
            for _ in self.behaviour_types
        ]
        for home in range(self.n_homes):
            for it, data in enumerate(self.behaviour_types):
                prev_cluster = prev_clusters[data][home]
                probs = self.p_trans[data][transition][prev_cluster]
                cum_p = [sum(probs[0:i]) for i in range(1, len(probs))] + [1]
                clusters[data].append(
                    [c > random_clus[it][home] for c in cum_p].index(True)
                )

        return clusters

    def _transition_type(self):
        day_type = "wd" if self.date.weekday() < 5 else "we"
        prev_day_type \
            = "wd" if (self.date - timedelta(days=1)).weekday() < 5 \
            else "we"
        transition = f"{prev_day_type}2{day_type}"

        return day_type, transition

    def _adjust_max_ev_loads(self, day, interval_f_ev, factors,
                             transition, clusters, day_type, i_ev):
        for home in range(self.n_homes):
            it = 0
            while np.max(day["loads_EV"][home]) > self.bat["c_max"] \
                    and it < 100:
                if it == 99:
                    print("100 iterations _adjust_max_ev_loads")
                if factors["EV"][home] > 0 and interval_f_ev[home] > 0:
                    interval_f_ev[home] -= 1
                    factors["EV"][home] = self.mid_xs[transition][
                        int(interval_f_ev[home])]
                    ev_cons = self.profs["EV"]["cons"][day_type][
                        clusters["EV"][home]][i_ev[home]]
                    assert sum(ev_cons) == 0 or abs(sum(ev_cons) - 1) < 1e-3, \
                        f"ev_cons {ev_cons}"
                    day["loads_EV"][home] \
                        = [p * factors["EV"][home] for p in ev_cons]
                else:
                    i_ev[home] = np.random.choice(np.arange(
                        self.n_prof["EV"][day_type][clusters["EV"][home]]))
                it += 1

        return interval_f_ev, factors, day, i_ev

    def _select_profiles(self,
                         data: str,
                         day_type: str = None,
                         i_month: int = 0,
                         clusters: List[int] = None
                         ) -> List[int]:
        """Randomly generate index of profile to select for given data."""
        i_profs = []
        for home in range(self.n_homes):
            if data in self.behaviour_types:
                n_profs = self.n_prof[data][day_type][clusters[data][home]]
            else:
                n_profs = self.n_prof[data][i_month]
            avail_profs = list(range(n_profs))
            i_prof = np.random.choice(avail_profs)
            if len(avail_profs) > 1:
                avail_profs.remove(i_prof)
            i_profs.append(i_prof)

        return i_profs

    def _ps_rand_to_choice(self, probs: List[float], rand: float) -> int:
        """Given list of probabilities, select index."""
        p_intervals = [sum(probs[0:i]) for i in range(len(probs))]
        choice = [ip for ip in range(len(p_intervals))
                  if rand > p_intervals[ip]][-1]

        return choice

    def _load_ev_profiles(self, input_dir, profiles, prm):
        labels_day = self.labels_day

        for data in ["cons", "avail"]:
            profiles["EV"][data] = initialise_dict(labels_day)
            for day_type in labels_day:
                profiles["EV"][data][day_type] = initialise_dict(
                    range(self.n_all_clusters_ev))
        self.n_prof["EV"] = initialise_dict(labels_day)

        for data, label in zip(["cons", "avail"], ["norm_EV", "EV_avail"]):
            path = input_dir / "profiles" / label
            files = os.listdir(path)
            for file in files:
                if file[0] != ".":
                    cluster = int(file[1])
                    data_type = file[3: 5]
                    profiles_ = np.load(path / file, mmap_mode="r")
                    # mmap_mode = 'r': not loaded, but elements accessible
                    if data == 'cons':
                        assert all(
                            [
                                sum(prof) == 0 or abs(sum(prof) - 1) < 1e-3
                                for prof in profiles_
                            ]
                        ), f"sum profiles not 1 path {path} file {file}"

                    prof_shape = np.shape(profiles_)
                    if len(prof_shape) == 1:
                        profiles_ = np.reshape(
                            prof_shape, (1, len(prof_shape))
                        )
                    profiles["EV"][data][data_type][cluster] = profiles_

        for day_type in labels_day:
            self.n_prof["EV"][day_type] = [
                len(profiles["EV"]["cons"][day_type][clus])
                for clus in range(self.n_all_clusters_ev)
            ]

        return profiles

    def _load_dem_profiles(self, profiles, prm):

        profiles["dem"] = initialise_dict(prm["weekday_type"], [])
        self.n_prof["dem"] = {}
        clusters = [
            int(file[1])
            for file in os.listdir(prm['profiles_path'] / "norm_dem")
        ]

        n_dem_clus = max(clusters) + 1

        path = self.save_path / "profiles" / "norm_dem"
        for day_type in prm["weekday_type"]:
            profiles["dem"][day_type] = [
                np.load(path / f"c{cluster}_{day_type}.npy", mmap_mode="r")
                for cluster in range(n_dem_clus)
            ]
            self.n_prof["dem"][day_type] = [
                len(profiles["dem"][day_type][clus])
                for clus in range(n_dem_clus)
            ]

        return profiles

    def _load_gen_profiles(self, inputs_path, profiles):
        path = inputs_path / "profiles" / "norm_gen"

        profiles["gen"] = [np.load(
            path / f"i_month{i_month}.npy", mmap_mode="r")
            for i_month in range(12)
        ]

        self.n_prof["gen"] = [len(profiles["gen"][m]) for m in range(12)]

        return profiles

    def _load_profiles(self, prm: dict) -> dict:
        """Load banks of profiles from files."""
        profiles: Dict[str, Any] = {"EV": {}}
        prm['profiles_path'] = self.save_path / "profiles"
        self.n_prof: dict = {}

        # EV profiles
        if "EV" in self.data_types:
            profiles \
                = self._load_ev_profiles(self.save_path, profiles, prm)

        # dem profiles
        if "dem" in self.data_types:
            profiles = self._load_dem_profiles(profiles, prm)

        # PV generation bank and month
        if "gen" in self.data_types:
            profiles = self._load_gen_profiles(self.save_path, profiles)

        return profiles

    def _check_feasibility(self, day: dict) -> List[bool]:
        """Given profiles generated, check feasibility."""
        feasible = [True for a in range(self.n_homes)]
        if self.max_discharge is not None:
            for home in range(self.n_homes):
                if self.max_discharge < np.max(day["loads_EV"][home]):
                    feasible[home] = False
                    for t in range(len(day["loads_EV"][home])):
                        if day["loads_EV"][home][t] > self.max_discharge:
                            day["loads_EV"][home][t] = self.max_discharge
        for home in range(self.n_homes):
            if feasible[home]:
                feasible[home] = self._check_charge(home, day)

        return feasible

    def _check_charge(self, home: int, day: dict) -> bool:
        """Given profiles generated, check feasibility of battery charge."""
        t = 0
        feasible = True
        store0 = self.bat["SoC0"] * self.bat["cap"]

        while feasible:
            last_step = t == self.n_steps
            # regular initial minimum charge
            min_charge_t_0 = (
                store0 * day["avail_EV"][home][t]
                if last_step
                else self.bat["min_charge"] * day["avail_EV"][home][t]
            )
            # min_charge if need to charge up ahead of last step
            if day["avail_EV"][home][t]:  # if you are currently in garage
                # obtain all future trips
                trip_loads: List[float] = []
                dt_to_trips: List[int] = []

                end = False
                t_trip = t
                while not end:
                    trip_load, dt_to_trip, t_end_trip \
                        = self._next_trip_details(t_trip, home, day)
                    if trip_load is None:
                        end = True
                    else:
                        feasible = self._check_trip_load(
                            feasible, trip_load, dt_to_trip, store0,
                            t, day["avail_EV"][home])
                        trip_loads.append(trip_load)
                        dt_to_trips.append(dt_to_trip)
                        t_trip = t_end_trip

                charge_req = self._get_charge_req(
                    store0, trip_loads, dt_to_trips,
                    t_end_trip, day["avail_EV"][home])

            else:
                charge_req = 0
            min_charge_t = [
                max(min_charge_t_0[home], charge_req[home])
                for home in range(self.n_homes)
            ]

            # determine whether you need to charge ahead for next EV trip
            # check if opportunity to charge before trip > 37.5
            if feasible and t == 0 and day["avail_EV"][home][0] == 0:
                feasible = self._ev_unavailable_start(
                    t, home, day, store0)

            # check if any hourly load is larger than d_max
            if sum(1 for t in range(self.n_steps)
                   if day["loads_EV"][home][t] > self.bat["d_max"] + 1e-2)\
                    > 0:
                # would have to break constraints to meet demand
                feasible = False

            if feasible:
                feasible = self._check_min_charge_t(
                    min_charge_t, day, home, t, store0)

            t += 1

        return feasible

    def _get_charge_req(self,
                        store0: float,
                        trip_loads: List[float],
                        dt_to_trips: List[int],
                        t_end_trip: int,
                        avail_ev: List[bool]
                        ) -> float:
        # obtain required charge before each trip, starting with end
        n_avail_until_end = sum(
            avail_ev[t] for t in range(t_end_trip, self.n_steps)
        )
        # this is the required charge for the current step
        # if there is no trip
        # or this is what is needed coming out of the last trip
        if len(trip_loads) == 0:
            n_avail_until_end -= 1
        charge_req = max(0, store0 - self.bat["c_max"] * n_avail_until_end)
        for it in range(len(trip_loads)):
            trip_load = trip_loads[- (it + 1)]
            dt_to_trip = dt_to_trips[- (it + 1)]
            if it == len(trip_loads) - 1:
                dt_to_trip -= 1
            # this is the required charge at the current step
            # if this is the most recent trip,
            # or right after the previous trip
            charge_req = max(
                0,
                charge_req + trip_load - dt_to_trip * self.bat["c_max"]
            )
        return charge_req

    def _check_trip_load(
            self,
            feasible: bool,
            trip_load: float,
            dt_to_trip: int,
            store0: float,
            t: int,
            avail_ev_: list
    ) -> bool:
        if trip_load > self.bat["cap"] + 1e-2:
            # load during trip larger than whole
            feasible = False
        elif (
                dt_to_trip > 0
                and sum(avail_ev_[0: t]) == 0
                and trip_load / dt_to_trip > store0 + self.bat["c_max"]
        ):
            feasible = False

        return feasible

    def _ev_unavailable_start(self, t, home, day, store0):
        feasible = True
        trip_load, dt_to_trip, _ \
            = self._next_trip_details(t, home, day)
        if trip_load > store0:
            # trip larger than initial charge
            # and straight away not available
            feasible = False
        if sum(day["avail_EV"][home][0:23]) == 0 \
                and sum(day["loads_EV"][home][0:23]) \
                > self.bat["c_max"] + 1e-2:
            feasible = False
        trip_load_next, next_dt_to_trip, _ \
            = self._next_trip_details(dt_to_trip, home, day)
        if next_dt_to_trip > 0 \
            and trip_load_next - (self.bat["store0"] - trip_load) \
                < self.bat["c_max"] / next_dt_to_trip:
            feasible = False

        return feasible

    def _check_min_charge_t(self,
                            min_charge_t: float,
                            day: dict,
                            home: int,
                            t: int,
                            store0: float
                            ) -> bool:
        feasible = True
        if min_charge_t > self.bat["cap"] + 1e-2:
            feasible = False  # min_charge_t larger than total cap
        if min_charge_t > self.bat["store0"] \
                - sum(day["loads_EV"][home][0: t]) \
                + (sum(day["loads_EV"][home][0: t]) + 1) * self.bat["c_max"] \
                + 1e-3:
            feasible = False

        if t > 0 and sum(day["avail_EV"][home][0:t]) == 0:
            # the EV has not been available at home to recharge until now
            store_t_a = store0 - sum(day["loads_EV"][home][0:t])
            if min_charge_t > store_t_a + self.bat["c_max"] + 1e-3:
                feasible = False

        return feasible

    def _next_trip_details(
            self,
            start_t: int,
            home: int,
            day: dict) \
            -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Identify the next trip time and requirements for given time step."""
        # next time the EV is on a trip
        ts_trips = [i for i in range(len(day["avail_EV"][home][start_t:]))
                    if day["avail_EV"][home][start_t + i] == 0]
        if len(ts_trips) > 0 and start_t + ts_trips[0] < self.n_steps:
            # future trip that starts before end
            t_trip = int(start_t + ts_trips[0])

            # next time the EV is back from the trip to the garage
            ts_back = [t_trip + i
                       for i in range(len(day["avail_EV"][home][t_trip:]))
                       if day["avail_EV"][home][t_trip + i] == 1]
            t_back = int(ts_back[0]) if len(ts_back) > 0 \
                else len(day["avail_EV"][home])
            dt_to_trip = t_trip - start_t  # time until trip
            t_end_trip = int(min(t_back, self.n_steps))

            # EV load while on trip
            trip_load = np.sum(day["loads_EV"][home][t_trip: t_end_trip])

            return trip_load, dt_to_trip, t_end_trip

        return None, None, None

    def _plotting_profiles(self, day):
        if not os.path.exists(self.save_day_path):
            os.mkdir(self.save_day_path)
        y_labels = {
            "EV": "Electric vehicle loads",
            "gen": "PV generation",
            "dem": "Household loads"
        }
        font = {'size': 22}
        matplotlib.rc('font', **font)
        hr_per_t = 24 / self.n_steps
        hours = [i * hr_per_t for i in range(self.n_steps)]
        for data_type in self.data_types:
            key = "loads_EV" if data_type == "EV" else data_type
            for a in range(self.n_homes):
                fig = plt.figure()
                plt.plot(hours, day[key][a], color="blue", lw=3)
                plt.xlabel("Time [hours]")
                plt.ylabel(f"{y_labels[data_type]} [kWh]")
                y_fmt = tick.FormatStrFormatter('%.1f')
                plt.gca().yaxis.set_major_formatter(y_fmt)
                plt.tight_layout()
                fig.savefig(self.save_day_path / f"{data_type}_a{a}")
                plt.close("all")

                if "EV" in self.data_types:
                    bands_bEV = []
                    non_avail = [
                        i for i in range(self.n_steps)
                        if day["avail_EV"][a][i] == 0
                    ]
                    if len(non_avail) > 0:
                        current_band = [non_avail[0] * hr_per_t]
                        if len(non_avail) > 1:
                            for i in range(1, len(non_avail)):
                                if non_avail[i] != non_avail[i - 1] + 1:
                                    current_band.append(
                                        (non_avail[i - 1] + 0.99) * hr_per_t
                                    )
                                    bands_bEV.append(current_band)
                                    current_band = [non_avail[i] * hr_per_t]
                        current_band.append(
                            (non_avail[-1] + 0.999) * hr_per_t
                        )
                        bands_bEV.append(current_band)

                    fig, ax = plt.subplots()
                    ax.step(
                        hours[0: self.n_steps],
                        day["loads_EV"][a][0: self.n_steps],
                        color='k',
                        where='post',
                        lw=3
                    )
                    for band in bands_bEV:
                        ax.axvspan(
                            band[0], band[1], alpha=0.3, color='grey'
                        )
                    grey_patch = matplotlib.patches.Patch(
                        alpha=0.3, color='grey', label='EV unavailable')
                    plt.legend(handles=[grey_patch], fancybox=True)
                    plt.xlabel("Time [hours]")
                    plt.ylabel("EV loads and at-home availability")
                    fig.tight_layout()
                    fig.savefig(self.save_day_path / f"avail_EV_a{a}")
                    plt.close("all")
