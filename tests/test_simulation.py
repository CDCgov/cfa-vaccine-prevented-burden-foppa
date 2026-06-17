from dataclasses import replace

import numpy.random

from burden import Parameters, Simulation

params = Parameters(
    n_people=int(1e2) + 1,
    p_oc_cf=0.1,
    p_vax=0.25,
    oc_time_mean=0.0,
    oc_time_scale=1.0,
    vax_time_mean=10.0,
    vax_time_scale=1.0,
    ve=0.5,
    ve_delay=1.0,
    period_duration=1.0,
    rng=numpy.random.default_rng(),
)


def test_params_dict():
    d = params.to_dict()
    assert isinstance(d, dict)
    assert d["p_oc_cf"] == params.p_oc_cf


def test_example():
    sim = Simulation(params)

    assert len(sim.t_oc_cf) == round(params.n_people * params.p_oc_cf)
    assert len(sim.t_vax) == round(params.n_people * params.p_vax)


def test_zero_ve():
    sim = Simulation(replace(params, ve=0.0))
    assert len(sim.t_oc) == len(sim.t_oc_cf)
    assert (sim.t_oc == sim.t_oc_cf).all()


def test_zero_coverage():
    sim = Simulation(replace(params, p_vax=0.0))
    assert (sim.t_oc == sim.t_oc_cf).all()


def test_long_delay():
    Simulation(replace(params, ve_delay=100.0))


def test_can_summarize():
    sim = Simulation(params)
    assert sim.total_oc_cf == len(sim.t_oc_cf)
    assert sim.total_oc == len(sim.t_oc)
    assert sim.total_vax == len(sim.t_vax)
