import functools

import numpy as np
from numpy.testing import assert_allclose

import burden

original_foppa = functools.partial(burden.foppa, n_ve_delay=0, vax_rate_type="impulse")


class TestOriginalFoppa:
    def test_trivial(self):
        """If there are no outcomes or vaccinations, nothing happens"""
        current = original_foppa(
            n_vax=np.array([0]),
            n_oc=np.array([0]),
            n_people=100,
            ve=0.2,
        )
        expected = np.array([0])
        assert (current == expected).all()

    def test_no_vaccination(self):
        """If there is no vaccination, nothing is prevented"""
        current = original_foppa(
            n_vax=np.repeat(0.0, 4),
            n_oc=np.array([1.0, 2.0, 3.0, 4.0]),
            n_people=100,
            ve=0.2,
        )
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        assert (current == expected).all()

    def test_no_oc(self):
        """If there are no outcomes, nothing is prevented"""
        current = original_foppa(
            n_vax=np.array([1.0, 2.0, 3.0, 4.0]),
            n_oc=np.repeat(0.0, 4),
            n_people=100,
            ve=0.2,
        )
        expected = np.repeat(0.0, 4)
        assert (current == expected).all()

    def test_simple(self):
        """Simple case"""
        current = original_foppa(
            n_vax=np.array([1.0, 2.0, 3.0, 4.0]),
            n_oc=np.array([1.0, 2.0, 3.0, 4.0]),
            n_people=1000,
            ve=1.0,
        )

        expected = np.array([1.001, 2.006, 3.018, 4.040])
        assert_allclose(current, expected, rtol=0.0, atol=1e-3)

    def test_handles_perfect_vaccination(self):
        """When VE*coverage=1, does not error out"""

        current = original_foppa(
            n_vax=np.array([10]),
            n_oc=np.array([1]),
            n_people=10,
            ve=1.0,
        )
        expected = np.array([np.inf])
        assert (current == expected).all()

    def test_avg_cum_timeseries(self):
        np.testing.assert_array_equal(
            burden._avg_cum_timeseries(np.array([1.0, 2.0, 3.0])),
            np.array([1.0 / 2, 1.0 + 2.0 / 2, 1.0 + 2.0 + 3.0 / 2]),
        )


class TestImprovedFoppa:
    def test_prepend(self):
        x = np.array([1.0, 2.0, 3.0])

        np.testing.assert_array_equal(burden._prepend(x, shift=0), x)

        np.testing.assert_array_equal(
            burden._prepend(x, shift=2, value=0.0),
            np.array([0.0, 0.0, 1.0, 2.0, 3.0]),
        )

    def test_shift_right_and_fill(self):
        np.testing.assert_array_equal(
            burden._shift_right_and_fill(np.array([1.0, 2.0, 3.0]), shift=2, value=0.0),
            np.array([0.0, 0.0, 1.0]),
        )

    def test_avg_cum_timeseries(self):
        np.testing.assert_array_equal(
            burden._avg_cum_timeseries(np.array([1.0, 2.0, 3.0])),
            np.array([0.5, 2.0, 4.5]),
        )

    def test_trivial(self):
        """If there are no outcomes or vaccinations, nothing happens"""
        current = burden.foppa(
            n_vax=np.array([0]),
            n_oc=np.array([0]),
            n_people=100,
            ve=0.2,
            n_ve_delay=0,
            vax_rate_type="constant",
        )
        np.testing.assert_array_equal(current, np.array([0]))

    def test_no_vaccination(self):
        """If there is no vaccination, nothing is prevented"""
        n_oc = np.array([1.0, 2.0, 3.0, 4.0])
        current = original_foppa(
            n_vax=np.repeat(0.0, 4), n_oc=n_oc, n_people=100, ve=0.2
        )
        np.testing.assert_array_equal(current, n_oc)

    def test_no_oc(self):
        """If there are no outcomes, nothing is prevented"""
        n_vax = np.array([1.0, 2.0, 3.0, 4.0])
        current = original_foppa(
            n_vax=n_vax, n_oc=np.repeat(0.0, 4), n_people=100, ve=0.2
        )
        np.testing.assert_array_equal(current, np.repeat(0.0, 4))

    def test_simple(self):
        """Simple case where you can do the math by hand"""
        n_vax = np.array([1.0, 2.0, 1.0, 0.5])
        n_oc = np.array([1.0, 2.0, 3.0, 4.0])
        current = burden.foppa(
            n_vax=n_vax,
            n_oc=n_oc,
            n_people=100,
            ve=0.75,
            vax_rate_type="constant",
            n_ve_delay=2,
        )

        expected = np.array(
            [
                1.0,
                2.0,
                3.0 / (1.0 - 0.75 * ((1.0 / 2) / 100)),
                4.0 / (1.0 - 0.75 * ((1.0 + 2.0 / 2) / 100)),
            ]
        )
        assert_allclose(current, expected, rtol=0.0, atol=1e-3)
