import argparse
import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import polars as pl
from helpers import add_panel_labels, apply_comma_ticks, colors

from burden import Parameters, Simulation

TICK_THICKNESS = 3
LINE_WIDTH = 2.0


def _line_style(variable: str) -> str | tuple[int, tuple[int, ...]]:
    if variable in ["vax", "oc", "oc_cf"]:
        return "-"
    if variable == "original":
        return (0, (3, 3))
    if variable == "improved":
        return (0, (5, 5))
    return "-"


def _line_width(variable: str, is_data: bool) -> float:
    return 2.0 * LINE_WIDTH if is_data else LINE_WIDTH


def make_params(
    vax_time_mean: float, ve_delay: float, period_duration: float, seed=4367
) -> Parameters:
    return Parameters(
        oc_time_mean=0.0,
        vax_time_mean=vax_time_mean,
        oc_time_scale=3.0,
        vax_time_scale=3.0,
        p_oc_cf=1.0,
        p_vax=1.0,
        ve=0.5,
        ve_delay=ve_delay,
        n_people=int(1e5),
        period_duration=period_duration,
        rng=numpy.random.default_rng(seed),
    )


def make_timeseries(
    vax_time_mean: float,
    ve_delay: float,
    period_duration: float,
    t_min: float,
    t_max: float,
) -> pl.DataFrame:
    s = Simulation(
        params=make_params(
            vax_time_mean=vax_time_mean,
            ve_delay=ve_delay,
            period_duration=period_duration,
        ),
        t_min=t_min,
        t_max=t_max,
    )

    period_midpoint = t_min + period_duration * (np.array(range(len(s.n_oc_cf))) + 0.5)

    return pl.DataFrame(
        {
            "vax_time_mean": vax_time_mean,
            "ve_delay": ve_delay,
            "period_duration": period_duration,
            "period_midpoint": period_midpoint,
            "oc_cf": s.n_oc_cf,
            "oc": s.n_oc,
            "vax": s.n_vax,
            "original": s.n_original,
            "improved": s.n_improved,
        }
    )


def make_data(
    vmt_min=-20.0,
    vmt_max=10.0,
    period_durations=[1.0, 5.0, 10.0],
    t_min=-50.0,
    t_max=25.0,
    ve_delays=[0.0, 10.0],
    n: int = 4,
) -> pl.DataFrame:
    f = functools.partial(make_timeseries, t_min=t_min, t_max=t_max)
    ts = np.linspace(vmt_min, vmt_max, n)

    return pl.concat(
        [
            f(t, ve_delay=0.0, period_duration=duration)
            for t in ts
            for duration in period_durations
        ]
        + [f(t, ve_delay=delay, period_duration=1.0) for t in ts for delay in ve_delays]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out_flag = Path(args.output)
    out_dir = out_flag.parent
    assert out_dir.is_dir()

    data = make_data()

    delay_data = (
        data.filter(pl.col("period_duration") == 1.0)
        .unpivot(
            index=["vax_time_mean", "ve_delay", "period_duration", "period_midpoint"]
        )
        # separate the rows -- override the vax values, so we don't get repeats
        .with_columns(
            row_id=pl.when(pl.col("variable") == pl.lit("vax"))
            .then(pl.lit("vax"))
            .otherwise(pl.format("ve_delay={}", pl.col("ve_delay"))),
        )
        .drop("ve_delay")
        .unique()
        # keep track of data vs. estimates
        .with_columns(is_data=pl.col("variable").is_in(["vax", "oc_cf", "oc"]))
        # sort for plotting
        .with_columns(
            variable_order=pl.col("variable").replace_strict(
                ["vax", "oc", "oc_cf", "original", "improved"],
                list(range(5)),
            )
        )
        .sort("variable_order", "period_midpoint")
        .with_row_index("index")
    )

    duration_data = (
        data.filter(pl.col("ve_delay") == 0.0, pl.col("period_duration") != 1.0)
        .drop(["ve_delay", "vax"])
        .unpivot(index=["vax_time_mean", "period_duration", "period_midpoint"])
        # keep track of data vs. estimates
        .with_columns(is_data=pl.col("variable").is_in(["oc_cf", "oc"]))
        # sort for plotting
        .with_columns(
            variable_order=pl.col("variable").replace_strict(
                ["oc", "oc_cf", "improved", "original"],
                list(range(4)),
            )
        )
        .sort("variable_order", "period_midpoint")
        .with_row_index("index")
    )

    row_ids = delay_data["row_id"].unique().sort().to_list()
    row_ids = [x for x in row_ids if x == "vax"] + [x for x in row_ids if x != "vax"]
    vax_time_means = delay_data["vax_time_mean"].unique().sort().to_list()
    period_durations = duration_data["period_duration"].unique().sort().to_list()

    n_delay_rows = len(row_ids)
    n_duration_rows = len(period_durations)
    n_rows = n_delay_rows + n_duration_rows
    n_cols = len(vax_time_means)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.0 * n_rows),
        squeeze=False,
        sharex=False,
        sharey="row",
    )

    for r, row_id in enumerate(row_ids):
        for c, vax_time_mean in enumerate(vax_time_means):
            ax = axs[r][c]
            panel = delay_data.filter(
                pl.col("row_id") == pl.lit(row_id),
                pl.col("vax_time_mean") == pl.lit(vax_time_mean),
            )
            for variable in ["vax", "oc", "oc_cf", "original", "improved"]:
                series = panel.filter(pl.col("variable") == pl.lit(variable))
                if series.height == 0:
                    continue
                is_data = bool(series["is_data"][0])
                ax.plot(
                    series["period_midpoint"],
                    series["value"],
                    color=colors[variable],
                    linestyle=_line_style(variable),
                    linewidth=_line_width(variable, is_data),
                    label=variable,
                )

            ax.plot(
                [vax_time_mean],
                [0],
                linestyle="none",
                marker="|",
                markersize=10,
                color=colors["vax"],
                markeredgewidth=TICK_THICKNESS,
            )

            if r == 0:
                ax.set_title(f"Mean vax.time = {vax_time_mean:g}")
            if c == 0:
                if row_id == "vax":
                    ax.set_ylabel("No. vaccinations")
                else:
                    ve_delay = float(row_id.removeprefix("ve_delay="))
                    ax.set_ylabel(f"VE delay = {ve_delay:g}")
            apply_comma_ticks(ax)

    for r2, period_duration in enumerate(period_durations):
        r = n_delay_rows + r2
        for c, vax_time_mean in enumerate(vax_time_means):
            ax = axs[r][c]
            panel = duration_data.filter(
                pl.col("period_duration") == pl.lit(period_duration),
                pl.col("vax_time_mean") == pl.lit(vax_time_mean),
            )
            for variable in ["oc", "oc_cf", "original", "improved"]:
                series = panel.filter(pl.col("variable") == pl.lit(variable))
                if series.height == 0:
                    continue
                is_data = bool(series["is_data"][0])
                ax.plot(
                    series["period_midpoint"],
                    series["value"],
                    color=colors[variable],
                    linestyle=_line_style(variable),
                    linewidth=_line_width(variable, is_data),
                    label=variable,
                )
                ax.scatter(
                    series["period_midpoint"],
                    series["value"],
                    color=colors[variable],
                    s=8,
                )

            ax.plot(
                [vax_time_mean],
                [0],
                linestyle="none",
                marker="|",
                markersize=10,
                color=colors["vax"],
                markeredgewidth=TICK_THICKNESS,
            )

            if c == 0:
                ax.set_ylabel(f"Period duration = {period_duration:g}")
            apply_comma_ticks(ax)

    add_panel_labels(ax for row_axes in axs for ax in row_axes)

    fig.supxlabel(
        "Time (midpoint of reporting period relative to peak counterfactual outcomes; arbitrary units)"
    )
    fig.tight_layout()
    fig.savefig(str(out_dir / "demo.png"), dpi=300)
    plt.close(fig)

    out_flag.touch()
