# -*- coding: utf-8 -*-

"""Energy-targeted sampling, the frequency cutoff and the limits on the geometry."""

import numpy as np
import pytest

import normal_mode_sampling_step.normal_mode_sampling as nms
from normal_mode_sampling_step import NormalModeSampling, NormalModeSamplingParameters

KCAL = 4.184  # kJ/mol


def triatomic(frequencies=(50.0, 1600.0, 3700.0)):
    """A bent H-O-H-like molecule whose Hessian has exactly the given vibrational
    frequencies (cm^-1), plus zero translations and rotations."""
    masses = np.array([1.008, 15.999, 1.008])
    coords = np.array([[0.757, 0.586, 0.0], [0.0, 0.0, 0.0], [-0.757, 0.586, 0.0]])
    B = nms._translation_rotation_basis(masses, coords)
    # an orthonormal basis of the vibrations: the complement of B
    Q, _ = np.linalg.qr(np.hstack([B, np.random.default_rng(0).normal(size=(9, 9))]))
    V = Q[:, B.shape[1] :]
    omega = 2.0 * np.pi * nms._C_CM_S * np.asarray(frequencies)
    lam = omega**2 / nms._LAMBDA_AU_TO_SI
    H_mw = V @ np.diag(lam) @ V.T
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    H = H_mw * np.outer(sqrt_m, sqrt_m)
    return H, masses, coords


def vibrations(frequencies=(50.0, 1600.0, 3700.0)):
    H, masses, coords = triatomic(frequencies)
    freqs, modes = nms.normal_mode_analysis(H, masses, coords)
    mask = freqs > nms.ZERO_MODE_THRESHOLD_CM
    return freqs[mask], modes[:, mask], masses, coords


def test_model_frequencies():
    freqs, _, _, _ = vibrations()
    assert freqs == pytest.approx([50.0, 1600.0, 3700.0], rel=1e-6)


def test_energy_targets_are_uniform_in_the_range():
    freqs, modes, masses, _ = vibrations((800.0, 1600.0, 3700.0))
    low, high = 0.5 * KCAL, 9.0 * KCAL
    disp, energies, q_vectors, n_rej, n_drop = nms.sample_normal_modes(
        freqs,
        modes,
        masses,
        20000,
        distribution="energy",
        energy_range=(low, high),
        amplitude_cap=3.0,
        rng=np.random.default_rng(1),
    )
    assert len(disp) == 20000 and n_rej == 0 and n_drop == 0
    energies = np.array(energies)
    assert energies.min() >= low and energies.max() <= high
    assert energies.mean() == pytest.approx(0.5 * (low + high), rel=0.02)
    # uniform: a quarter of the targets in each quarter of the range
    counts, _ = np.histogram(energies, bins=4, range=(low, high))
    assert counts / len(energies) == pytest.approx([0.25] * 4, abs=0.02)

    # the reported energy is the harmonic energy of the reported displacement
    omega = 2.0 * np.pi * nms._C_CM_S * freqs
    q = np.array(q_vectors) / nms._Q_SI_TO_AMU_HALF_ANG  # SI
    e = 0.5 * np.sum(omega**2 * q**2, axis=1) * nms._N_AVOGADRO / 1000.0
    assert e == pytest.approx(energies, rel=1e-9)

    # equipartition shape: on average the same share of the energy in every mode
    share = (0.5 * omega**2 * q**2 * nms._N_AVOGADRO / 1000.0) / e[:, None]
    assert share.mean(axis=0) == pytest.approx([1 / 3] * 3, abs=0.01)


def test_energy_targeting_needs_a_valid_range():
    freqs, modes, masses, _ = vibrations()
    with pytest.raises(ValueError, match="energy range"):
        nms.sample_normal_modes(freqs, modes, masses, 1, distribution="energy")
    with pytest.raises(ValueError, match="0 < low <= high"):
        nms.sample_normal_modes(
            freqs, modes, masses, 1, distribution="energy", energy_range=(2.0, 1.0)
        )


def test_the_guard_limits_every_accepted_sample():
    # stiff modes only: with a 50 cm^-1 mode most equipartition draws move an atom
    # far (why the recipe leaves out modes below 100 cm^-1)
    freqs, modes, masses, coords = vibrations((800.0, 1600.0, 3700.0))
    guard = nms.geometry_guard(coords, [(0, 1), (1, 2)], max_displacement=0.10)
    disp, energies, _, n_rej, n_drop = nms.sample_normal_modes(
        freqs,
        modes,
        masses,
        200,
        distribution="energy",
        energy_range=(2.0, 37.0),
        guard=guard,
        max_draws=None,
        rng=np.random.default_rng(2),
    )
    assert len(disp) == 200 and n_drop == 0 and n_rej > 0
    assert max(np.max(np.linalg.norm(d, axis=1)) for d in disp) <= 0.10


def test_samples_are_dropped_after_max_draws():
    freqs, modes, masses, coords = vibrations()
    guard = nms.geometry_guard(coords, max_displacement=1.0e-6)  # nothing passes
    disp, _, _, n_rej, n_drop = nms.sample_normal_modes(
        freqs,
        modes,
        masses,
        7,
        distribution="energy",
        energy_range=(2.0, 37.0),
        guard=guard,
        max_draws=5,
        rng=np.random.default_rng(3),
    )
    assert disp == [] and n_drop == 7 and n_rej == 35


def test_geometry_guard_reasons():
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    bonds = [(0, 1), (1, 2)]
    assert nms.geometry_guard(x0, bonds) is None  # no limits, no guard
    guard = nms.geometry_guard(
        x0, bonds, max_displacement=0.3, max_bond_change=0.2, max_angle_change=15.0
    )
    assert guard(np.zeros((3, 3))) is None
    dx = np.zeros((3, 3))
    dx[0, 0] = -0.35  # moves 0.35 Å
    assert "atom moved" in guard(dx)
    dx = np.zeros((3, 3))
    dx[0, 0], dx[1, 0] = -0.125, 0.125  # bond 0-1 0.25 Å longer, 0.125 Å moves
    assert "bond changed" in guard(dx)
    dx = np.zeros((3, 3))
    dx[2, 0] = 0.29  # the 90° angle opens by ~16°
    assert "angle changed" in guard(dx)
    dx[2, 0] = 0.20  # ~11°
    assert guard(dx) is None


def test_cutoff_and_count_helpers():
    P = NormalModeSamplingParameters().values_to_dict()
    assert NormalModeSampling._number_of_samples(P, 33) == 20
    P["number of samples"] = "in proportion to the modes"
    # round(1/2 (3N - 6)) between 10 and 40, rounding halves up
    assert NormalModeSampling._number_of_samples(P, 3) == 10
    assert NormalModeSampling._number_of_samples(P, 33) == 17
    assert NormalModeSampling._number_of_samples(P, 51) == 26
    assert NormalModeSampling._number_of_samples(P, 300) == 40


def test_seed_is_recorded():
    seed = NormalModeSampling._seed_of("random")
    assert isinstance(seed, int) and 0 <= seed < 2**63
    assert NormalModeSampling._seed_of("42") == 42
    assert NormalModeSampling._seed_of(7) == 7


def test_new_parameters():
    P = NormalModeSamplingParameters()
    assert "energy-targeted (equipartition)" in P["distribution"].enumeration
    for key, default in (
        ("minimum target energy", 0.5),
        ("maximum target energy", 9.0),
        ("lowest frequency", "none"),
        ("maximum atom displacement", "none"),
        ("maximum bond change", "none"),
        ("maximum angle change", "none"),
        ("maximum draws", 5),
        ("samples per mode", 0.5),
        ("minimum number of samples", 10),
        ("maximum number of samples", 40),
    ):
        assert P[key].default == default, key
    assert NormalModeSampling._distribution_key("energy-targeted (equipartition)") == (
        "energy"
    )


def test_guards_do_not_bias_the_target_energies():
    """A sample keeps its target through redraws, so a guard that rejects many
    high-energy shapes still leaves the accepted targets uniform."""
    freqs, modes, masses, coords = vibrations((800.0, 1600.0, 3700.0))
    low, high = 0.5 * KCAL, 9.0 * KCAL
    guard = nms.geometry_guard(coords, [(0, 1), (1, 2)], max_displacement=0.12)
    _, energies, _, n_rej, n_drop = nms.sample_normal_modes(
        freqs,
        modes,
        masses,
        4000,
        distribution="energy",
        energy_range=(low, high),
        guard=guard,
        max_draws=None,
        rng=np.random.default_rng(5),
    )
    assert n_rej > 400 and n_drop == 0
    assert np.mean(energies) == pytest.approx(0.5 * (low + high), rel=0.03)
