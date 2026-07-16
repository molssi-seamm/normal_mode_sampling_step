# -*- coding: utf-8 -*-

"""Tests for the Normal Mode Sampling control parameters."""

from normal_mode_sampling_step import NormalModeSamplingParameters


def test_parameters_construct_with_defaults():
    P = NormalModeSamplingParameters()
    # Every documented parameter is present.
    for key in (
        "structure",
        "structure configurations",
        "structure configuration name",
        "number of samples",
        "distribution",
        "temperature",
        "amplitude cap",
        "energy ceiling",
        "modes",
        "random seed",
        "system name",
        "configuration name",
        "results",
    ):
        assert key in P


def test_key_defaults():
    P = NormalModeSamplingParameters()
    assert P["distribution"].default == "Wigner (quantum)"
    assert P["number of samples"].default == 20
    assert P["structure configurations"].default == "current"
    assert P["random seed"].default == "random"
    assert P["energy ceiling"].default == "none"


def test_distribution_enumeration():
    P = NormalModeSamplingParameters()
    enum = P["distribution"].enumeration
    assert "Wigner (quantum)" in enum
    assert "classical (thermal)" in enum
    assert "ground state (0 K)" in enum
