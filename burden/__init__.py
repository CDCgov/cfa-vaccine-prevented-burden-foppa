import dataclasses
import functools
from typing import Any, Literal

import numpy as np
import numpy.random


@dataclasses.dataclass
class Parameters:
    oc_time_mean: float
    oc_time_scale: float
    vax_time_mean: float
    vax_time_scale: float
    p_oc_cf: float
    p_vax: float
    ve: float
    ve_delay: float
    period_duration: float
    n_people: int
    rng: numpy.random.Generator

    def __post_init__(self):
        assert self.oc_time_scale > 0.0
        assert self.vax_time_scale > 0.0
        assert 0.0 <= self.p_oc_cf <= 1.0
        assert 0.0 <= self.p_vax <= 1.0
        assert 0.0 <= self.ve <= 1.0
        assert self.ve_delay >= 0.0
        assert self.period_duration > 0.0
        assert self.n_people > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert parameters to a dictionary, excluding the RNG"""
        d = dataclasses.asdict(self)
        del d["rng"]
        return d


class Simulation:
    def __init__(
        self, params: Parameters, t_min: float | None = None, t_max: float | None = None
    ):
        # store parameters
        self.params = params

        # derive parameters
        self.n_ve_delay = round(self.params.ve_delay / self.params.period_duration)

        # run simulation, producing times of each individual's events
        self.t_oc_cf, self.t_oc, self.t_vax = self._simulate(self.params)

        # summarize the simulation, creating timeseries of outcomes and vaccinations
        self.n_oc_cf, self.n_oc, self.n_vax, self.period_starts = self._summarize(
            t_oc_cf=self.t_oc_cf,
            t_oc=self.t_oc,
            t_vax=self.t_vax,
            period_duration=self.params.period_duration,
            t_min=t_min,
            t_max=t_max,
        )

        # run estimates
        self.n_original, self.n_improved = self._estimate(
            n_vax=self.n_vax,
            n_oc=self.n_oc,
            n_people=self.params.n_people,
            ve=self.params.ve,
            n_ve_delay=self.n_ve_delay,
        )

        self.total_oc_cf = self.n_oc_cf.sum()
        self.total_oc = self.n_oc.sum()
        self.total_vax = self.n_vax.sum()
        self.total_original = self.n_original.sum()
        self.total_improved = self.n_improved.sum()

    def to_dict(self) -> dict[str, float | int | np.ndarray]:
        """
        Cast simulation object as a polars-compatible dict. Arrays are cast
        as lists.
        """
        return {
            key: self._ensure_list(getattr(self, key))
            for key in [
                "total_oc_cf",
                "total_oc",
                "total_vax",
                "total_original",
                "total_improved",
                "period_starts",
                "n_oc_cf",
                "n_oc",
                "n_vax",
                "n_original",
                "n_improved",
            ]
        } | self.params.to_dict()

    @staticmethod
    def _ensure_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        else:
            return x

    @classmethod
    def _simulate(cls, params: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # draw outcome and vaccination times for everyone, even if they do not suffer
        # the outcome, either counterfactually or in-factually
        t_oc = params.rng.normal(
            loc=params.oc_time_mean, scale=params.oc_time_scale, size=params.n_people
        )
        t_vax = params.rng.normal(
            loc=params.vax_time_mean, scale=params.vax_time_scale, size=params.n_people
        )

        # who suffers the outcome in the counterfactual?
        n_oc_cf = round(params.p_oc_cf * params.n_people)
        has_oc_cf = cls._bool_permutation(
            n_true=n_oc_cf, n=params.n_people, rng=params.rng
        )

        # who is vaccinated in-fact?
        n_vax = round(params.p_vax * params.n_people)
        is_vax = cls._bool_permutation(n_true=n_vax, n=params.n_people, rng=params.rng)
        # if they were vaccinated, was vaccination effective?
        is_ve_eff = params.rng.choice(
            [True, False], p=[params.ve, 1.0 - params.ve], size=params.n_people
        )

        # if outcome occurred in counterfactual, was it prevented in-fact?
        is_prevented = is_vax & is_ve_eff & (t_oc - t_vax > params.ve_delay)
        # did they suffer the outcome in fact?
        has_oc = has_oc_cf & ~is_prevented

        return (t_oc[has_oc_cf], t_oc[has_oc], t_vax[is_vax])

    @staticmethod
    def _bool_permutation(n_true: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Create a random boolean array with a fixed number of true values.

        Args:
            n_true: Number of ``True`` values to include.
            n: Total length of the output array.
            rng: Random number generator used to permute the values.

        Returns:
            A permuted boolean array of length ``n`` with exactly ``n_true``
            true values.
        """
        return rng.permutation([True] * n_true + [False] * (n - n_true))

    @staticmethod
    def _summarize(
        t_oc_cf: np.ndarray,
        t_oc: np.ndarray,
        t_vax: np.ndarray,
        period_duration: float,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate event times into binned time series.

        Args:
            t_min: Optional lower bound for the binning range. If omitted, the
                minimum observed event time is used.
            t_max: Optional upper bound for the binning range. If omitted, the
                maximum observed event time is used.

        Returns:
            A tuple of:
              - binned counterfactual outcome time
              - binned outcome times
              - binned vaccination counts
              - starts of each period (i.e., left side of each bin)
        """
        # by default, use time bounds from the data themselves
        ts = np.concat((t_oc_cf, t_oc, t_vax))
        t_min = t_min or ts.min()
        t_max = t_max or ts.max()
        assert isinstance(t_min, float)
        assert isinstance(t_max, float)

        # bin event times into periods
        # t_max should fall in the last bin
        bins = np.arange(t_min, t_max + period_duration, period_duration)
        assert len(bins) >= 2
        assert bins[0] == t_min
        assert t_max <= bins[-1]
        if len(bins) > 2:
            assert bins[-2] < t_max

        def bin(x: np.ndarray) -> np.ndarray:
            counts, bins_out = np.histogram(x, bins=bins)
            assert (bins == bins_out).all()
            return counts

        return (bin(t_oc_cf), bin(t_oc), bin(t_vax), bins[:-1])

    @staticmethod
    def _estimate(
        n_vax: np.ndarray, n_oc: np.ndarray, n_people: int, ve: float, n_ve_delay: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate counterfactual burden

        Returns:
            tuple of arrays representing:
                - time series estimated by original method
                - by improved method
        """
        f = functools.partial(foppa, n_vax=n_vax, n_oc=n_oc, n_people=n_people, ve=ve)

        return (
            f(n_ve_delay=0, vax_rate_type="impulse"),
            f(n_ve_delay=n_ve_delay, vax_rate_type="constant"),
        )


def foppa(
    n_vax: np.ndarray,
    n_oc: np.ndarray,
    n_people: float,
    ve: float,
    n_ve_delay: int,
    vax_rate_type: Literal["impulse", "constant"],
) -> np.ndarray:
    """Estimate counterfactual outcomes using the Foppa method.

    Args:
        n_vax: Time series of vaccination counts.
        n_oc: Time series of observed outcome counts.
        n_people: Population size.
        ve: Vaccine effectiveness.
        n_ve_delay: Minimum number of periods between vaccination and
            prevented outcome.
        vax_rate_type: Vaccination rate type

    Returns:
        Estimated counterfactual outcome counts for each time period
    """
    assert len(n_vax) == len(n_oc)
    n_vax_per_capita = n_vax / n_people

    # the "cumulative uptake" depends on the time method
    match vax_rate_type:
        case "impulse":
            cum_uptake = n_vax_per_capita.cumsum()
        case "constant":
            cum_uptake = _avg_cum_timeseries(n_vax_per_capita)
        case _:
            raise ValueError(f"Unknown vax_rate_type: {vax_rate_type}")

    # compute protection: VE times time-shifted (and potentially averaged) uptake
    prot = ve * _shift_right_and_fill(cum_uptake, shift=n_ve_delay)
    assert all(prot >= 0.0) and all(prot <= 1.0)

    # in most cases, RR = 1/(1-v). But if v=1, then risk ratio is infinite
    risk_ratio = 1.0 / (1.0 - prot)

    return n_oc * risk_ratio


def _shift_right_and_fill(x: np.ndarray, shift: int, value=0.0) -> np.ndarray:
    """
    Add values to the start of a vector, dropping elements at the end to
    maintain the same length

    E.g.: [x0, x1, ..., xn] -> [0, 0, x0, x1, ..., xn-2]
    """
    if shift > len(x):
        return np.zeros_like(x)
    else:
        return _prepend(x=x, shift=shift, value=value)[0 : len(x)]


def _prepend(x: np.ndarray, shift: int, value=0.0) -> np.ndarray:
    """Add values to the start of a vector"""
    return np.concat([np.repeat(value, shift), x])


def _avg_cum_timeseries(x: np.ndarray) -> np.ndarray:
    """
    Average cumulative time series

    For input x, the output y is:
      y_0 = 0.5*x_0
      y_1 = x_0 + 0.5*x_1
      y_2 = x_0 + x_1 + 0.5*x_2
      etc.

    Args:
        x: input

    Returns:
        The averaged cumulative time series.
    """
    assert len(x) > 0
    # make a new vector [0.0, x0, x1, x2, ..., x(n-1)]
    # and cum sum it: [0.0, x0, x0 + x1, ..., x0 + x1 + ... + x(n-1)]
    cum = _shift_right_and_fill(x, shift=1, value=0.0).cumsum()
    # add [0.5*x0, , ...] to produce [0.5*x0, x0 + 0.5*x1, ...]
    return cum + 0.5 * x
