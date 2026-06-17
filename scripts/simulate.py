import argparse
import json
from typing import Any, Callable

import numpy as np
import numpy.random
import polars as pl
import scipy.stats.qmc

from burden import Parameters, Simulation


class Experiment:
    def __init__(
        self,
        n_sim: int,
        min_total_oc: int,
        parameters: dict[str, dict[str, Any]],
        seed: int | None = None,
    ):
        transforms = self._parse_parameters(parameters)
        raw_simulations = [
            Simulation(p)
            for p in self.sample_params(n=n_sim, seed=seed, transforms=transforms)
        ]
        self.simulations = [s for s in raw_simulations if s.total_oc >= min_total_oc]

        print(
            f"Of {n_sim} total simulations, keeping {len(self.simulations)}"
            f" with at least {min_total_oc} total outcomes"
        )

    def to_df(self) -> pl.DataFrame:
        return (
            pl.from_dicts(s.to_dict() for s in self.simulations)
            # estimation errors
            .with_columns(
                error_original=pl.col("total_original") - pl.col("total_oc_cf"),
                error_improved=pl.col("total_improved") - pl.col("total_oc_cf"),
            )
            # difference in absolute errors (positive means original is worse)
            .with_columns(
                error_diff=pl.col("error_original").abs()
                - pl.col("error_improved").abs(),
                error_abs_rel=pl.col("error_original").abs()
                / pl.col("error_improved").abs(),
            )
        )

    @classmethod
    def sample_params(
        cls,
        n: int,
        seed: int | None,
        transforms: dict[str, Callable[[float], float]],
    ) -> list[Parameters]:
        """Sample parameter sets using Latin hypercube sampling.

        Args:
            n: Number of parameter sets to generate.
            seed: Seed for coordinates and for spawning per-simulation RNGs.
            transforms: Mapping from parameter names to unit-interval transforms.

        Returns:
            A list of sampled `Parameters` instances.
        """
        coords = scipy.stats.qmc.LatinHypercube(d=len(transforms), rng=seed).random(n)
        rngs = numpy.random.default_rng(seed).spawn(n)
        return [
            Parameters(
                rng=rng,
                **{key: f(value) for (key, f), value in zip(transforms.items(), row)},  # type: ignore
            )
            for row, rng in zip(coords, rngs)
        ]

    @classmethod
    def _parse_parameters(
        cls, parameters: dict[str, dict[str, Any]]
    ) -> dict[str, Callable[[float], float]]:
        """
        E.g., { "ve": { "type": "scale", "min": 0.0, "max": 1.0 } } -> { "ve": cls._scale_fun(0.0, 1.0) }
        """
        return {name: cls._parse_parameter(spec) for name, spec in parameters.items()}

    @classmethod
    def _parse_parameter(cls, spec: dict[str, Any]) -> Callable[[float], float]:
        """
        E.g., { "type": "constant", "value": 1 } -> cls._constant_fun(1)
        """
        match spec["type"]:
            case "constant":
                return cls._constant_fun(spec["value"])
            case "scale":
                return cls._scale_fun(spec["min"], spec["max"])
            case "scale_log":
                return cls._scale_log_fun(spec["min"], spec["max"])
            case _:
                raise ValueError(f"Unsupported transform: {spec}")

    @staticmethod
    def _scale_fun(min_value, max_value) -> Callable[[float], float]:
        """Create a linear scaling function from the unit interval.

        Args:
            min_value: Output value when the input is 0.
            max_value: Output value when the input is 1.

        Returns:
            A function that maps values in ``[0, 1]`` to the interval
            ``[min_value, max_value]``.
        """
        assert max_value > min_value

        def f(x):
            assert 0 <= x <= 1
            return min_value + x * (max_value - min_value)

        return f

    @staticmethod
    def _scale_log_fun(min_value, max_value) -> Callable[[float], float]:
        """Create a log-linear scaling function from the unit interval.

        Args:
            min_value: Output value when the input is 0. Must be positive.
            max_value: Output value when the input is 1. Must be positive.

        Returns:
            A function that maps values in ``[0, 1]`` to the interval
            ``[min_value, max_value]`` on a logarithmic scale.
        """
        assert min_value > 0.0
        assert max_value > min_value

        def f(x):
            assert 0 <= x <= 1
            return np.exp(
                np.log(min_value) + x * (np.log(max_value) - np.log(min_value))
            )

        return f

    @staticmethod
    def _constant_fun(value) -> Callable[[float], float]:
        """Create a function that always returns a constant value.

        Args:
            value: Constant output value.

        Returns:
            A function that ignores its input and returns ``value``.
        """

        def f(x):
            assert 0 <= x <= 1
            return value

        return f


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True, help="Output parquet")
    args = p.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    ex = Experiment(
        n_sim=config["n_simulations"],
        min_total_oc=config["min_n_total_outcomes"],
        parameters=config["parameters"],
        seed=config["seed"],
    )
    ex.to_df().write_parquet(args.output)
