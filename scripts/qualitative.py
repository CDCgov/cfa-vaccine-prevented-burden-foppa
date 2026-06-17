import argparse
from pathlib import Path

import polars as pl
import polars.selectors as cs


def add_percent(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.format("{} ({}%)", num, (num / den * 100).round().cast(pl.Int64))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="simulations parquet")
    p.add_argument("--output", help="output flag")
    args = p.parse_args()

    out_flag = Path(args.output)
    out_dir = out_flag.parent
    assert out_dir.exists() and out_dir.is_dir()

    data = (
        pl.read_parquet(args.input)
        .group_by(
            when=pl.when(pl.col("vax_time_mean") >= 0.0)
            .then(pl.lit("vtm>=0"))
            .otherwise(pl.lit("vtm<0"))
        )
        .agg(
            n=pl.col("error_diff").len(),
            n_original_better=(
                pl.col("error_original").abs() < pl.col("error_improved").abs()
            ).sum(),
            n_improved_better=(
                pl.col("error_original").abs() > pl.col("error_improved").abs()
            ).sum(),
            n_equal=(pl.col("error_improved") == pl.col("error_original")).sum(),
            error_original_mean=pl.col("error_original").mean(),
            error_improved_mean=pl.col("error_improved").mean(),
            error_abs_original_mean=pl.col("error_original").abs().mean(),
            error_abs_improved_mean=pl.col("error_improved").abs().mean(),
        )
    )

    # table of simulation counts (and percents) by mean vaccine time
    table_n = (
        data.select("when", "n", cs.starts_with("n_"))
        .with_columns(
            add_percent(pl.col(c), pl.col("n"))
            for c in data.columns
            if c.startswith("n_")
        )
        .with_columns(pl.col("n").cast(pl.String))
        .unpivot(index="when")
        .pivot(on="when", index="variable")
        .sort("variable")
        .select(["variable", "vtm<0", "vtm>=0"])
    )

    # table of errors
    table_error = (
        data.select("when", cs.starts_with("error").round(1))
        .unpivot(index="when")
        .pivot(on="when", values="value")
        .with_columns(pl.all().cast(pl.String))
        .sort("variable", descending=True)
        .select(["variable", "vtm<0", "vtm>=0"])
    )

    # combine them into a single table
    table = pl.concat([table_n, table_error])
    table.write_csv(out_dir / "qual.csv")

    out_flag.touch()
