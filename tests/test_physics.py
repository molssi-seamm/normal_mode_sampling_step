# -*- coding: utf-8 -*-

"""Tests for the pure-physics helpers of the Normal Mode Sampling step.

These functions are deliberately free of any flowchart state, so they can be
checked directly against closed-form harmonic-oscillator values.
"""

import numpy as np
import pytest

from normal_mode_sampling_step import normal_mode_sampling as nms


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def diatomic(k=0.5, mass=1.0, bond=0.74):
    """A single-mode 'diatomic': coupling only along z, equal masses.

    Returns (hessian_au, masses_amu, coords_ang) and the analytic vibrational
    frequency (cm^-1) for the reduced mass mu = mass / 2.
    """
    H = np.zeros((6, 6))
    H[2, 2] = k
    H[5, 5] = k
    H[2, 5] = -k
    H[5, 2] = -k
    masses = np.array([mass, mass])
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]])

    mu = mass / 2.0
    # Magnitude of the vibrational frequency (a negative k gives an imaginary
    # mode of the same magnitude).
    nu = np.sqrt((abs(k) / mu) * nms._LAMBDA_AU_TO_SI) / (2.0 * np.pi * nms._C_CM_S)
    return H, masses, coords, nu


def recover_normal_coordinate(displacements_ang, mode_mw, masses_amu):
    """Project sampled Cartesian displacements back onto a normal coordinate."""
    arr = np.asarray(displacements_ang, dtype=float)
    disp = arr.reshape(arr.shape[0], -1)
    m3_kg = np.repeat(masses_amu, 3) * nms._AMU_KG
    dxi = (disp * 1.0e-10) * np.sqrt(m3_kg)  # mass-weighted, sqrt(kg) m
    return dxi @ mode_mw


# --------------------------------------------------------------------------- #
# normal_mode_analysis
# --------------------------------------------------------------------------- #
def test_diatomic_frequency_matches_analytic():
    H, masses, coords, nu = diatomic()
    freqs, modes = nms.normal_mode_analysis(H, masses, coords)

    vib = freqs[freqs > nms.ZERO_MODE_THRESHOLD_CM]
    assert vib.size == 1
    assert vib[0] == pytest.approx(nu, rel=1.0e-9)
    # A linear molecule has 5 external (translation/rotation) modes.
    assert int(np.sum(np.abs(freqs) <= nms.ZERO_MODE_THRESHOLD_CM)) == 5
    # Eigenvectors are returned as columns, orthonormal.
    assert modes.shape == (6, 6)
    assert np.allclose(modes.T @ modes, np.eye(6), atol=1.0e-10)


def test_imaginary_mode_is_negative_frequency():
    # Flip the sign of the force constant -> the vibrational eigenvalue is
    # negative, which must come back as a negative (imaginary) frequency.
    H, masses, coords, nu = diatomic(k=-0.5)
    freqs, modes = nms.normal_mode_analysis(H, masses, coords)

    imaginary = freqs[freqs < -nms.ZERO_MODE_THRESHOLD_CM]
    assert imaginary.size == 1
    assert imaginary[0] == pytest.approx(-nu, rel=1.0e-9)


@pytest.mark.parametrize(
    "coords, n_external",
    [
        (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]), 5),  # linear
        (np.array([[0.0, 0.0, 0.0], [0.0, 0.96, 0.0], [0.93, -0.24, 0.0]]), 6),  # bent
    ],
)
def test_translation_rotation_projection_counts(coords, n_external):
    # With the mass-weighted Hessian equal to the identity (H = diag(m3)), the
    # projected external subspace gives exactly n_external zero eigenvalues and
    # the rest are equal, nonzero frequencies -- a clean, deterministic check.
    masses = np.array([16.0, 1.0, 1.0])
    m3 = np.repeat(masses, 3)
    H = np.diag(m3)  # mass-weighted -> identity

    freqs, modes = nms.normal_mode_analysis(H, masses, coords)

    n_zero = int(np.sum(np.abs(freqs) <= nms.ZERO_MODE_THRESHOLD_CM))
    assert n_zero == n_external
    vib = freqs[freqs > nms.ZERO_MODE_THRESHOLD_CM]
    assert vib.size == 9 - n_external
    # All the internal modes share the same (identity) frequency.
    assert np.allclose(vib, vib[0], rtol=1.0e-6)


def test_translation_rotation_basis_orthonormal():
    masses = np.array([16.0, 1.0, 1.0])
    bent = np.array([[0.0, 0.0, 0.0], [0.0, 0.96, 0.0], [0.93, -0.24, 0.0]])
    linear = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])

    B_bent = nms._translation_rotation_basis(masses, bent)
    assert B_bent.shape == (9, 6)
    assert np.allclose(B_bent.T @ B_bent, np.eye(6), atol=1.0e-10)

    B_lin = nms._translation_rotation_basis(masses, linear)
    assert B_lin.shape == (9, 5)


# --------------------------------------------------------------------------- #
# zero_point_energy
# --------------------------------------------------------------------------- #
def test_zero_point_energy_single_mode():
    nu = 3000.0  # cm^-1
    zpe = nms.zero_point_energy([nu])
    h = 2.0 * np.pi * nms._HBAR_J_S
    expected = 0.5 * h * nms._C_CM_S * nu * nms._N_AVOGADRO / 1000.0
    assert zpe == pytest.approx(expected, rel=1.0e-12)


def test_zero_point_energy_ignores_nonpositive():
    # Imaginary / zero frequencies must not contribute.
    assert nms.zero_point_energy([-100.0, 0.0]) == pytest.approx(0.0)
    both = nms.zero_point_energy([-100.0, 2000.0])
    only = nms.zero_point_energy([2000.0])
    assert both == pytest.approx(only)


# --------------------------------------------------------------------------- #
# _sigma_squared: the three distributions
# --------------------------------------------------------------------------- #
def test_sigma_ordering_for_a_stiff_mode():
    # 5000 cm^-1 is very stiff: hbar*omega/2kT >> 1 at 300 K.
    omega = 2.0 * np.pi * nms._C_CM_S * 5000.0
    wigner = nms._sigma_squared(omega, "wigner", 300.0)
    ground = nms._sigma_squared(omega, "ground", 300.0)
    classical = nms._sigma_squared(omega, "classical", 300.0)

    # Quantum width >= zero-point width, and both >> classical for a stiff mode.
    assert wigner >= ground
    assert wigner > classical
    # coth(x) -> 1 for large x, so Wigner collapses onto the ground state.
    assert wigner == pytest.approx(ground, rel=1.0e-3)


def test_sigma_classical_matches_high_temperature_wigner():
    # For a soft mode at high T, hbar*omega/2kT << 1 and Wigner -> classical.
    omega = 2.0 * np.pi * nms._C_CM_S * 50.0
    wigner = nms._sigma_squared(omega, "wigner", 1000.0)
    classical = nms._sigma_squared(omega, "classical", 1000.0)
    assert wigner == pytest.approx(classical, rel=1.0e-2)


# --------------------------------------------------------------------------- #
# sample_normal_modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("distribution", ["ground", "wigner", "classical"])
def test_sampling_width_matches_sigma(distribution):
    H, masses, coords, nu = diatomic()
    freqs, modes = nms.normal_mode_analysis(H, masses, coords)
    mask = freqs > nms.ZERO_MODE_THRESHOLD_CM
    vib, vmodes = freqs[mask], modes[:, mask]

    rng = np.random.default_rng(12345)
    disp, energies, q_vectors, n_rej = nms.sample_normal_modes(
        vib,
        vmodes,
        masses,
        40000,
        distribution=distribution,
        temperature=300.0,
        rng=rng,
    )
    assert len(disp) == 40000
    assert n_rej == 0

    q = recover_normal_coordinate(disp, vmodes[:, 0], masses)
    omega = 2.0 * np.pi * nms._C_CM_S * vib[0]
    sigma_expected = np.sqrt(nms._sigma_squared(omega, distribution, 300.0))
    assert q.std() == pytest.approx(sigma_expected, rel=0.05)
    assert q.mean() == pytest.approx(0.0, abs=0.1 * sigma_expected)

    # The reported per-mode normal coordinate (amu^1/2 Angstrom) matches the
    # geometric displacement recovered from the Cartesian coordinates.
    assert len(q_vectors) == 40000
    assert q_vectors[0].shape == (1,)
    reported = np.array([qv[0] for qv in q_vectors])
    assert reported.std() == pytest.approx(
        sigma_expected * nms._Q_SI_TO_AMU_HALF_ANG, rel=0.05
    )
    assert reported * (1.0 / nms._Q_SI_TO_AMU_HALF_ANG) == pytest.approx(q, rel=1.0e-6)


def test_amplitude_cap_clips_draws():
    H, masses, coords, nu = diatomic()
    freqs, modes = nms.normal_mode_analysis(H, masses, coords)
    mask = freqs > nms.ZERO_MODE_THRESHOLD_CM
    vib, vmodes = freqs[mask], modes[:, mask]

    rng = np.random.default_rng(7)
    disp, _, _, _ = nms.sample_normal_modes(
        vib, vmodes, masses, 20000, distribution="wigner", amplitude_cap=1.0, rng=rng
    )
    q = recover_normal_coordinate(disp, vmodes[:, 0], masses)
    omega = 2.0 * np.pi * nms._C_CM_S * vib[0]
    sigma = np.sqrt(nms._sigma_squared(omega, "wigner", 300.0))
    assert np.abs(q).max() <= sigma * 1.0 * (1.0 + 1.0e-9)


def test_energy_ceiling_rejects_and_bounds_energy():
    H, masses, coords, nu = diatomic()
    freqs, modes = nms.normal_mode_analysis(H, masses, coords)
    mask = freqs > nms.ZERO_MODE_THRESHOLD_CM
    vib, vmodes = freqs[mask], modes[:, mask]

    ceiling = 10.0  # kJ/mol; mean ground-state harmonic energy is ~15 kJ/mol here
    rng = np.random.default_rng(3)
    disp, energies, _, n_rej = nms.sample_normal_modes(
        vib,
        vmodes,
        masses,
        200,
        distribution="ground",
        energy_ceiling=ceiling,
        rng=rng,
    )
    assert len(disp) == 200
    assert n_rej > 0
    assert max(energies) <= ceiling


def test_sampling_is_reproducible_with_seed():
    H, masses, coords, nu = diatomic()
    freqs, modes = nms.normal_mode_analysis(H, masses, coords)
    mask = freqs > nms.ZERO_MODE_THRESHOLD_CM
    vib, vmodes = freqs[mask], modes[:, mask]

    a, _, _, _ = nms.sample_normal_modes(
        vib, vmodes, masses, 10, rng=np.random.default_rng(99)
    )
    b, _, _, _ = nms.sample_normal_modes(
        vib, vmodes, masses, 10, rng=np.random.default_rng(99)
    )
    for da, db in zip(a, b):
        assert np.array_equal(da, db)
