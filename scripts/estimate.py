import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from helpers import add_panel_labels, apply_comma_ticks
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import LogLocator, NullFormatter

MIN_VALUE = -300
MAX_VALUE = 600
BIN_STEP = 100
HEIGHT = 250
CENSORED_MARKERS = {"min": "v", "none": "o", "max": "^"}


def censor(
    df: pl.DataFrame,
    col: str,
    min_value: float = MIN_VALUE,
    max_value: float = MAX_VALUE,
) -> pl.DataFrame:
    c = pl.col(col)

    type_ = (
        pl.when(c < min_value)
        .then(pl.lit("min"))
        .when(c > max_value)
        .then(pl.lit("max"))
        .otherwise(pl.lit("none"))
    )

    value = (
        pl.when(type_ == pl.lit("min"))
        .then(min_value)
        .when(type_ == pl.lit("max"))
        .then(max_value)
        .when(type_ == pl.lit("none"))
        .then(c)
    )

    return df.with_columns(
        type_.alias(f"{col}_censoring"), value.alias(f"{col}_censored")
    )


def make_chart_row(
    ax_points: Axes,
    ax_hist: Axes,
    data: pl.DataFrame,
    x_title1: str | None,
    x_title2: str | None,
    y_title: str,
) -> None:
    for censoring, marker in CENSORED_MARKERS.items():
        points = data.filter(pl.col("value_censoring") == pl.lit(censoring))
        ax_points.scatter(
            points["total_oc_cf"],
            points["value_censored"],
            color="black",
            marker=MarkerStyle(marker),
            alpha=0.33,
            s=10,
        )

    ax_points.set_xscale("log")
    ax_points.xaxis.set_major_locator(LogLocator(base=10.0))
    ax_points.xaxis.set_minor_formatter(NullFormatter())
    apply_comma_ticks(ax_points, x=True, y=False)
    ax_points.set_ylim(MIN_VALUE, MAX_VALUE)
    ax_points.set_ylabel(y_title)
    if x_title1 is not None:
        ax_points.set_xlabel(x_title1)

    bins = np.arange(MIN_VALUE, MAX_VALUE + BIN_STEP, BIN_STEP)
    ax_hist.hist(
        data["value_censored"],
        bins=bins,  # type: ignore
        orientation="horizontal",
        color="black",
    )
    ax_hist.set_ylim(MIN_VALUE, MAX_VALUE)
    ax_hist.set_xlim(0, 1e4)
    ax_hist.set_xticks([0, 2.5e3, 5e3, 7.5e3, 1e4])
    ax_hist.set_yticklabels([])
    apply_comma_ticks(ax_hist, x=True, y=False)
    if x_title2 is not None:
        ax_hist.set_xlabel(x_title2)


def make_chart(data: pl.DataFrame, rows: list[str], y_titles: list[str]) -> Figure:
    n = len(rows)
    x_title1s = [None] * (n - 1) + ["True no. counterfactual outcomes"]
    x_title2s = [None] * (n - 1) + ["No. simulations"]
    fig, axs = plt.subplots(
        n,
        2,
        figsize=(7.5, 2.8 * n),
        gridspec_kw={"width_ratios": [1.0, 0.6]},
        squeeze=False,
    )

    for i, (row, x_title1, x_title2, y_title) in enumerate(
        zip(rows, x_title1s, x_title2s, y_titles)
    ):
        make_chart_row(
            axs[i][0],
            axs[i][1],
            data.filter(pl.col("variable") == pl.lit(row)),
            x_title1=x_title1,
            x_title2=x_title2,
            y_title=y_title,
        )

    add_panel_labels(ax for row_axes in axs for ax in row_axes)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out_flag = Path(args.output)
    out_dir = out_flag.parent
    assert out_dir.is_dir()

    report_buffer = io.StringIO()

    data = (
        pl.read_parquet(args.input)
        .with_row_index()
        .select(
            [
                "index",
                "total_oc_cf",
                "error_original",
                "error_improved",
                "error_diff",
            ]
        )
    )

    print(f"{data.height} total simulations", file=report_buffer)

    print(
        data.with_columns(
            cmp=(pl.col("error_original").abs() - pl.col("error_improved").abs())
            .sign()
            .replace_strict(
                {-1.0: "original better", 1.0: "improved better", 0.0: "equal"}
            )
        )
        .group_by("cmp")
        .agg(pl.len().alias("n"))
        .with_columns(frac=(pl.col("n") / pl.col("n").sum()).round(2)),
        file=report_buffer,
    )

    print(
        data.select(["index", "error_original", "error_improved"])
        .unpivot(index="index")
        .group_by("variable")
        .agg(mean=pl.col("value").mean(), std=pl.col("value").std())
        .with_columns(pl.col("mean", "std").round(1)),
        file=report_buffer,
    )

    chart_data = data.unpivot(index=["index", "total_oc_cf"]).pipe(censor, "value")

    print(
        "Censoring report:\n",
        chart_data.filter(pl.col("value_censoring") != pl.lit("none"))
        .group_by(["variable", "value_censoring"])
        .agg(n=pl.len()),
        file=report_buffer,
    )

    fig = make_chart(
        chart_data,
        rows=["error_original", "error_improved", "error_diff"],
        y_titles=[
            "Error: original method",
            "Error: improved method",
            "Original abs. error - improved abs. error",
        ],
    )
    fig.savefig(str(out_dir / "compare.png"), dpi=300)
    plt.close(fig)

    with open(out_dir / "estimate_report.txt", "w") as f:
        f.write(report_buffer.getvalue())

    report_buffer.close()

    out_flag.touch()
