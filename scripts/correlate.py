import argparse
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats
from helpers import apply_comma_ticks

import burden

LABELS = {
    "oc_time_scale": "Outcome time scale",
    "p_oc_cf": "Prop. with counterf. outcome",
    "p_vax": "Prop. vaccinated",
    "period_duration": "Period duration",
    "vax_time_mean": "Mean vax. time",
    "vax_time_scale": "Vax. time scale",
    "ve": "VE",
    "ve_delay": "VE delay",
}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out_flag = Path(args.output)
    out_dir = out_flag.parent
    assert out_dir.is_dir()

    data = (
        pl.read_parquet(args.input)
        # don't allow zero denominator
        .filter(pl.col("error_improved") != 0)
        # require at least some amount of error, to avoid looking at numerical instabilities
        .filter(
            pl.col("error_improved").abs() > 1.0, pl.col("error_original").abs() > 1.0
        )
        .with_columns(
            error_abs_rel=pl.col("error_original").abs()
            / pl.col("error_improved").abs()
        )
    )

    # get parameter names (except those that are fixed, or rng)
    pars = set(inspect.signature(burden.Parameters).parameters.keys()) - set(
        ["oc_time_mean", "n_people", "rng"]
    )
    assert pars.issubset(set(data.columns))

    # extract correlations
    def f(par) -> dict:
        result = scipy.stats.kendalltau(data["error_abs_rel"], data[par])
        return {"par": par, "statistic": result.statistic, "pvalue": result.pvalue}  # type: ignore

    cor = (
        pl.from_dicts(f(par) for par in pars)
        .sort("pvalue")
        .with_columns(pl.col("statistic").round(2), pl.col("pvalue").round_sig_figs(2))
    )

    cor.write_csv(out_dir / "cor.csv")

    chart_data = (
        data.select(pars | {"error_abs_rel"})
        .with_row_index()
        .unpivot(index=["index", "error_abs_rel"])
    )

    variables = chart_data["variable"].unique().sort().to_list()
    n_cols = 3
    n_rows = int(np.ceil(len(variables) / n_cols))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharey=True,
    )

    used_axes = []
    for i, variable in enumerate(variables):
        r, c = divmod(i, n_cols)
        ax = axs[r][c]
        used_axes.append(ax)
        plot_data = chart_data.filter(pl.col("variable") == pl.lit(variable))
        ax.scatter(plot_data["value"], plot_data["error_abs_rel"], color="black", s=8)
        ax.set_yscale("log")
        ax.set_title(LABELS[variable])
        apply_comma_ticks(ax)
        if c == 0:
            ax.set_ylabel("Relative absolute error")

    for j in range(len(variables), n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axs[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(out_dir / "cor.png"), dpi=300)
    plt.close(fig)

    out_flag.touch()
