# -*- coding: utf-8 -*-

"""Tests for the small parsing/naming helpers of the Normal Mode Sampling node."""

import numpy as np
import pytest

from seamm_util import Q_
from normal_mode_sampling_step.normal_mode_sampling import NormalModeSampling


@pytest.fixture(scope="module")
def node():
    return NormalModeSampling()


# --------------------------------------------------------------------------- #
# _parse_mode_selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spec, n, expected_true",
    [
        ("all", 6, [0, 1, 2, 3, 4, 5]),
        ("", 4, [0, 1, 2, 3]),
        ("1-3, 7", 8, [0, 1, 2, 6]),
        ("2 4 6", 6, [1, 3, 5]),
        ("5", 3, []),  # out of range -> ignored
        ("1-100", 3, [0, 1, 2]),  # range clamped to what exists
    ],
)
def test_parse_mode_selection(node, spec, n, expected_true):
    mask = node._parse_mode_selection(spec, n)
    assert mask.dtype == bool
    assert list(np.where(mask)[0]) == expected_true


# --------------------------------------------------------------------------- #
# _configuration_name
# --------------------------------------------------------------------------- #
def test_configuration_name_sequential(node):
    assert node._configuration_name("sequential", 2, 7, 5) == "7"


def test_configuration_name_sample_number(node):
    # per_ref = 5, running index 7 -> 2nd sample of the 2nd reference.
    assert node._configuration_name("sample number", 2, 7, 5) == "2"


def test_configuration_name_reference_sample(node):
    assert node._configuration_name("reference,sample", 2, 7, 5) == "2,2"
    assert node._configuration_name("reference,sample", 1, 1, 5) == "1,1"


# --------------------------------------------------------------------------- #
# _distribution_key / _is_none / _as_float
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, expected",
    [
        ("Wigner (quantum)", "wigner"),
        ("classical (thermal)", "classical"),
        ("ground state (0 K)", "ground"),
        ("something else", "wigner"),
    ],
)
def test_distribution_key(node, value, expected):
    assert node._distribution_key(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [("none", True), ("None", True), ("", True), ("  ", True), ("4.0", False)],
)
def test_is_none(node, value, expected):
    assert node._is_none(value) is expected


def test_as_float_plain_and_quantity(node):
    assert node._as_float(5.0, "K") == pytest.approx(5.0)
    assert node._as_float("7.5", "K") == pytest.approx(7.5)
    assert node._as_float(Q_(300.0, "K"), "K") == pytest.approx(300.0)
    # A quantity is converted into the requested units.
    assert node._as_float(Q_(1.0, "kcal/mol"), "kJ/mol") == pytest.approx(4.184)


# --------------------------------------------------------------------------- #
# _make_rng
# --------------------------------------------------------------------------- #
def test_make_rng_reproducible(node):
    a = node._make_rng("42").standard_normal(5)
    b = node._make_rng(42).standard_normal(5)
    assert np.array_equal(a, b)


def test_make_rng_random_is_a_generator(node):
    rng = node._make_rng("random")
    assert isinstance(rng, np.random.Generator)
