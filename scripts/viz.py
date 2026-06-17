import matplotlib.pyplot as plt
import numpy.random
import polars as pl
import streamlit as st
from helpers import add_panel_labels, apply_comma_ticks, colors

import burden


def app():
    with st.sidebar:
        oc_time_mean = 0.0
        oc_time_scale = 1.0
        vax_time_mean = st.slider("mean vax time", -10.0, 10.0, -5.0)
        vax_time_scale = st.slider("vax time scale", 1e-6, 10.0, 1.0)
        p_oc_cf = st.slider("p oc cf", 0.0, 1.0, 0.1)
        p_vax = st.slider("p vax", 0.0, 1.0, 0.5)
        ve = st.slider("ve", 0.0, 1.0, 0.5)
        ve_delay = st.slider("ve delay", 0.0, 30.0, 0.0)
        period = st.slider("period", 1e-6, 10.0, 1.0)
        n_people = int(1e5)
        seed = st.number_input("seed", min_value=0, max_value=1000, value=123)
        rng = numpy.random.default_rng(seed)

    s = burden.Simulation(
        burden.Parameters(
            oc_time_mean=oc_time_mean,
            oc_time_scale=oc_time_scale,
            vax_time_mean=vax_time_mean,
            vax_time_scale=vax_time_scale,
            p_oc_cf=p_oc_cf,
            p_vax=p_vax,
            ve=ve,
            ve_delay=ve_delay,
            n_people=n_people,
            period_duration=period,
            rng=rng,
        )
    )

    data_wide = pl.DataFrame(
        {
            "vax": s.n_vax,
            "oc": s.n_oc,
            "oc_cf": s.n_oc_cf,
            "improved": s.n_improved,
            "original": s.n_original,
        }
    ).with_row_index("period")
    data = data_wide.unpivot(index="period")

    data_sum = dict(data.group_by("variable").agg(pl.col("value").sum()).iter_rows())
    data_sum["error_original"] = data_sum["original"] - data_sum["oc_cf"]
    data_sum["error_improved"] = data_sum["improved"] - data_sum["oc_cf"]

    plt.style.use("ggplot")
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    vax_data = data.filter(pl.col("variable") == pl.lit("vax"))
    axs[0].plot(vax_data["period"], vax_data["value"], color=colors["vax"])
    axs[0].set_ylabel("Count")

    oc_data = data.filter(pl.col("variable") != pl.lit("vax")).with_columns(
        is_data=pl.col("variable").is_in(["oc", "oc_cf"])
    )
    for variable in ["oc", "oc_cf", "improved", "original"]:
        series = oc_data.filter(pl.col("variable") == pl.lit(variable))
        is_data = bool(series["is_data"][0])
        axs[1].plot(
            series["period"],
            series["value"],
            color=colors[variable],
            linestyle="-" if is_data else "--",
            label=variable,
        )

    axs[1].set_xlabel("period")
    axs[1].set_ylabel("Count")
    axs[1].legend(loc="upper right")
    apply_comma_ticks(axs[0])
    apply_comma_ticks(axs[1])
    add_panel_labels(axs)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.table(data_wide)


if __name__ == "__main__":
    app()
