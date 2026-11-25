# -*- coding: utf-8 -*-

"""Non-graphical part of the Normal Mode Sampling step in a SEAMM flowchart

The step draws an ensemble of displaced geometries by Wigner (or classical /
ground-state) normal-mode sampling of the Hessian, and writes them as
configurations of a new system for downstream single-point labelling.

The Hessian is obtained from the Model Chemistry published upstream (the
``_model_chemistry`` workspace variable), driven over MDI by ``seamm_mdi``:

* if the engine advertises the custom ``<HESSIAN`` command
  (``MDIEngine.supports("<HESSIAN")`` -- true only when it has a genuine analytic
  Hessian for the method), the analytic Hessian is used directly;
* otherwise the Hessian is built by central finite differences of the forces
  (``<FORCES``) over the *warm* engine -- cheap because the engine stays
  resident, and code-agnostic.

The physics helpers (``normal_mode_analysis``, ``sample_normal_modes``,
``zero_point_energy``) are module-level and free of any flowchart state so they
can be unit-tested directly.
"""

import logging
from pathlib import Path
import importlib.resources
import pprint  # noqa: F401

import numpy as np

import normal_mode_sampling_step
import molsystem
import seamm
from seamm_util import ureg, Q_  # noqa: F401
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("Normal Mode Sampling")

# Add this module's properties to the standard properties
path = importlib.resources.files("normal_mode_sampling_step") / "data"
csv_file = path / "properties.csv"
if path.exists():
    molsystem.add_properties_from_file(csv_file)

# Physical constants (CODATA 2018), for turning a Hessian in atomic units and
# masses in amu into frequencies, amplitudes, and energies in SI.
_HARTREE_J = 4.3597447222071e-18  # J
_BOHR_M = 5.29177210903e-11  # m
_AMU_KG = 1.66053906660e-27  # kg
_HBAR_J_S = 1.054571817e-34  # J s
_KB_J_K = 1.380649e-23  # J / K
_C_CM_S = 2.99792458e10  # cm / s
_N_AVOGADRO = 6.02214076e23  # 1 / mol

# Hessian eigenvalue (hartree / (bohr^2 amu)) -> angular-frequency^2 (1/s^2).
_LAMBDA_AU_TO_SI = _HARTREE_J / (_BOHR_M**2 * _AMU_KG)

# Mass-weighted normal coordinate: internal SI (sqrt(kg) m) -> the conventional
# amu^1/2 Angstrom used to report normal-coordinate displacements.
_Q_SI_TO_AMU_HALF_ANG = 1.0 / (_AMU_KG**0.5 * 1.0e-10)

# Modes with |frequency| at or below this are treated as translation/rotation or
# numerical zeros and are not sampled; more negative than this is "imaginary".
ZERO_MODE_THRESHOLD_CM = 10.0

# Per-configuration properties stored on each generated structure. A fixed suffix
# (not "{model}") keeps the names stable so downstream steps can reliably filter
# on them, mirroring the dimer_builder "#DimerBuilder#scan" convention.
_HARMONIC_ENERGY_PROPERTY = "harmonic energy#NormalModeSampling#sampling"
_DISPLACEMENT_PROPERTY = "normal mode displacement#NormalModeSampling#sampling"


# --------------------------------------------------------------------------- #
# Pure-physics helpers (no flowchart state -- unit-testable on their own)
# --------------------------------------------------------------------------- #
def _translation_rotation_basis(masses_amu, coords_ang):
    """Orthonormal mass-weighted translation/rotation vectors (Eckart).

    Returns an ``(3N, k)`` array whose columns span the ``k`` (=6, or 5 for a
    linear molecule) infinitesimal translations and rotations in mass-weighted
    Cartesian coordinates, orthonormalized. Projecting these out of the
    mass-weighted Hessian removes the external degrees of freedom before
    diagonalization.
    """
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_ang, dtype=float).reshape(-1, 3)
    n = masses.size
    sqrt_m = np.sqrt(masses)

    # Center of mass, so the rotations are about it.
    com = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    rel = coords - com

    vectors = []
    # Three translations: atom a moves by e_k; mass-weighted -> sqrt(m_a) e_k.
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = sqrt_m
        vectors.append(v.ravel())
    # Three rotations: delta x_a = e_k x rel_a; mass-weighted -> sqrt(m_a)(...).
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        v = sqrt_m[:, None] * np.cross(axis, rel)
        vectors.append(v.ravel())

    # Gram-Schmidt, dropping near-null vectors (the missing rotation of a linear
    # molecule falls out here).
    basis = []
    for v in vectors:
        for b in basis:
            v = v - np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1.0e-6:
            basis.append(v / norm)
    return np.array(basis).T if basis else np.zeros((3 * n, 0))


def normal_mode_analysis(hessian_au, masses_amu, coords_ang):
    """Diagonalize the (projected, mass-weighted) Hessian.

    Parameters
    ----------
    hessian_au : (3N, 3N) array
        The Cartesian Hessian in atomic units (hartree / bohr^2).
    masses_amu : (N,) array
        Atomic masses in amu.
    coords_ang : (N, 3) array
        Reference coordinates in Angstrom (for the rotation projection).

    Returns
    -------
    frequencies_cm : (3N,) ndarray
        Frequencies in cm^-1, ascending, **signed** (negative == imaginary).
    modes_mw : (3N, 3N) ndarray
        The corresponding orthonormal mass-weighted eigenvectors, as columns in
        the same order as ``frequencies_cm``.
    """
    masses = np.asarray(masses_amu, dtype=float)
    n = masses.size
    H = np.asarray(hessian_au, dtype=float).reshape(3 * n, 3 * n)
    H = 0.5 * (H + H.T)

    # Mass-weight: H_mw[i,j] = H[i,j] / sqrt(m_i m_j).
    m3 = np.repeat(masses, 3)
    inv_sqrt = 1.0 / np.sqrt(m3)
    H_mw = H * np.outer(inv_sqrt, inv_sqrt)

    # Project out translations and rotations.
    B = _translation_rotation_basis(masses, coords_ang)
    if B.shape[1] > 0:
        P = np.eye(3 * n) - B @ B.T
        H_mw = P.T @ H_mw @ P

    eigvals, eigvecs = np.linalg.eigh(H_mw)

    lam_si = eigvals * _LAMBDA_AU_TO_SI  # angular-frequency^2, 1/s^2
    with np.errstate(invalid="ignore"):
        freq_cm = np.sign(lam_si) * np.sqrt(np.abs(lam_si)) / (2.0 * np.pi * _C_CM_S)

    order = np.argsort(freq_cm)
    return freq_cm[order], eigvecs[:, order]


def _sigma_squared(omega, distribution, temperature):
    """Variance of the normal coordinate (mass-weighted, SI: kg m^2).

    ``omega`` is the angular frequency (1/s). ``distribution`` is one of
    ``wigner``, ``classical``, ``ground`` or ``energy``. For ``energy`` (the
    energy-targeted distribution) only the shape matters, since each draw is
    scaled to its target energy afterwards: the classical (equipartition) shape,
    sigma proportional to 1/omega, the same expected energy in every mode.
    """
    if distribution == "classical":
        # <Q^2> = kT / omega^2 for a unit-mass harmonic oscillator.
        return _KB_J_K * temperature / omega**2
    if distribution == "energy":
        return 1.0 / omega**2
    zpe_var = _HBAR_J_S / (2.0 * omega)  # ground-state (T -> 0) width
    if distribution == "ground":
        return zpe_var
    # Wigner (quantum) width at finite T.
    x = _HBAR_J_S * omega / (2.0 * _KB_J_K * temperature)
    return zpe_var / np.tanh(x)


def _angles_of(bonds, n_atoms):
    """The bond angles (i, j, k), j the centre, of a list of bonds (i, j)."""
    neighbours = [set() for _ in range(n_atoms)]
    for i, j in bonds:
        neighbours[i].add(j)
        neighbours[j].add(i)
    angles = []
    for j in range(n_atoms):
        nb = sorted(neighbours[j])
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                angles.append((nb[a], j, nb[b]))
    return angles


def _angle_deg(x, i, j, k):
    u, v = x[i] - x[j], x[k] - x[j]
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def geometry_guard(
    coords_ang,
    bonds=(),
    max_displacement=None,
    max_bond_change=None,
    max_angle_change=None,
):
    """A test of displaced geometries against limits on how far they move.

    Parameters
    ----------
    coords_ang : (N, 3) array
        The reference coordinates (Angstrom).
    bonds : sequence of (i, j)
        The bonds (0-based) whose lengths, and the angles between them, are
        checked.
    max_displacement : float or None
        The largest distance any atom may move (Angstrom).
    max_bond_change : float or None
        The largest change of any bond length (Angstrom).
    max_angle_change : float or None
        The largest change of any bond angle (degrees).

    Returns
    -------
    callable or None
        ``guard(dx)`` -> None if the displacement ``dx`` (N, 3) passes, else
        the reason, as text; None when no limit is set.
    """
    if (
        max_displacement is None
        and max_bond_change is None
        and (max_angle_change is None)
    ):
        return None
    x0 = np.asarray(coords_ang, dtype=float).reshape(-1, 3)
    bonds = [(int(i), int(j)) for i, j in bonds]
    r0 = {b: np.linalg.norm(x0[b[0]] - x0[b[1]]) for b in bonds}
    angles = _angles_of(bonds, len(x0)) if max_angle_change is not None else []
    a0 = {a: _angle_deg(x0, *a) for a in angles}

    def guard(dx):
        dx = np.asarray(dx, dtype=float).reshape(-1, 3)
        if max_displacement is not None:
            d = float(np.max(np.linalg.norm(dx, axis=1)))
            if d > max_displacement:
                return f"an atom moved {d:.3f} Å"
        x = x0 + dx
        if max_bond_change is not None:
            for (i, j), r in r0.items():
                d = abs(np.linalg.norm(x[i] - x[j]) - r)
                if d > max_bond_change:
                    return f"a bond changed by {d:.3f} Å"
        if max_angle_change is not None:
            for a, value in a0.items():
                d = abs(_angle_deg(x, *a) - value)
                if d > max_angle_change:
                    return f"an angle changed by {d:.1f}°"
        return None

    return guard


def sample_normal_modes(
    frequencies_cm,
    modes_mw,
    masses_amu,
    n_samples,
    *,
    distribution="wigner",
    temperature=300.0,
    amplitude_cap=None,
    energy_ceiling=None,
    energy_range=None,
    guard=None,
    max_draws=None,
    rng=None,
    max_tries_factor=100,
):
    """Draw displaced geometries by normal-mode sampling.

    Parameters
    ----------
    frequencies_cm : (M,) array
        Frequencies (cm^-1, positive) of the modes to sample.
    modes_mw : (3N, M) array
        The matching orthonormal mass-weighted eigenvectors, as columns.
    masses_amu : (N,) array
        Atomic masses (amu).
    n_samples : int
        Number of geometries requested.
    distribution : {"wigner", "classical", "ground", "energy"}
        "energy" is energy-targeted sampling: the equipartition shape (sigma
        proportional to 1/omega), each draw then scaled so that its harmonic
        energy is a target drawn uniformly from ``energy_range``. A sample keeps
        its target through redraws after guard failures.
    temperature : float
        Temperature (K); used only by the Wigner and classical distributions.
    amplitude_cap : float or None
        Clamp each normal-coordinate draw to this many standard deviations (for
        "energy", applied to the shape before the scaling).
    energy_ceiling : float or None
        Reject (and redraw) samples whose harmonic energy above the reference
        (kJ/mol) exceeds this.
    energy_range : (float, float) or None
        The range (kJ/mol) of the target energies, for "energy".
    guard : callable or None
        ``guard(dx)`` -> None to accept the Cartesian displacement dx (N, 3,
        Angstrom), or the reason to redraw (see :func:`geometry_guard`).
    max_draws : int or None
        With a guard: the most draws for one sample. When all of them fail the
        guard, that sample is dropped. None: redraw until accepted (bounded by
        ``max_tries_factor``).
    rng : numpy.random.Generator or None
    max_tries_factor : int
        Safety bound on redraws (per requested sample) before giving up.

    Returns
    -------
    displacements : list[(N, 3) ndarray]
        Cartesian displacements (Angstrom) to add to the reference geometry.
    energies_kJ : list[float]
        Harmonic energy of each accepted sample above the reference (kJ/mol).
    q_vectors : list[(M,) ndarray]
        The normal-coordinate displacement per sampled mode for each accepted
        sample, in amu^1/2 Angstrom (same mode order as ``frequencies_cm``).
    n_rejected : int
        Number of draws rejected by the energy ceiling or the guard.
    n_dropped : int
        Number of samples dropped because every one of their ``max_draws``
        draws failed the guard.
    """
    if rng is None:
        rng = np.random.default_rng()
    if distribution == "energy":
        if energy_range is None:
            raise ValueError("Energy-targeted sampling needs an energy range.")
        e_low, e_high = (float(v) for v in energy_range)
        if not 0.0 < e_low <= e_high:
            raise ValueError(
                f"The target energies must satisfy 0 < low <= high: {energy_range}"
            )

    freqs = np.asarray(frequencies_cm, dtype=float)
    modes = np.asarray(modes_mw, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    n_modes = freqs.size

    omega = 2.0 * np.pi * _C_CM_S * freqs  # angular frequency, 1/s
    sigma = np.sqrt(
        np.array([_sigma_squared(w, distribution, temperature) for w in omega])
    )  # sqrt(kg) m, per mode (for "energy", the shape only)

    inv_sqrt_m3 = 1.0 / np.sqrt(np.repeat(masses, 3) * _AMU_KG)  # 1/sqrt(kg)
    to_kJ = _N_AVOGADRO / 1000.0  # J per molecule -> kJ/mol

    displacements = []
    energies_kJ = []
    q_vectors = []
    n_rejected = 0
    n_dropped = 0
    max_tries = max(1, max_tries_factor) * max(1, n_samples)
    tries = 0
    draws = 0  # draws for the current sample
    while len(displacements) + n_dropped < n_samples and tries < max_tries:
        tries += 1
        draws += 1
        q = rng.standard_normal(n_modes) * sigma  # normal coords, sqrt(kg) m
        if amplitude_cap is not None:
            q = np.clip(q, -amplitude_cap * sigma, amplitude_cap * sigma)

        # Harmonic energy E = 1/2 sum omega^2 q^2 (unit mass), J -> kJ/mol.
        energy_kJ = 0.5 * np.sum(omega**2 * q**2) * to_kJ
        if distribution == "energy":
            # One target per sample: a redraw after a guard failure draws a new
            # shape at the same target, so the guards do not bias the accepted
            # energies toward the bottom of the range.
            if draws == 1:
                target = rng.uniform(e_low, e_high)
            q = q * np.sqrt(target / energy_kJ)
            energy_kJ = target
        if energy_ceiling is not None and energy_kJ > energy_ceiling:
            n_rejected += 1
            draws = 0  # a fresh draw, with a new target
            continue

        # Mass-weighted Cartesian displacement, then de-weight to metres -> Ang.
        dxi = modes @ q  # (3N,), sqrt(kg) m
        dx_m = dxi * inv_sqrt_m3  # (3N,), m
        dx_ang = (dx_m / 1.0e-10).reshape(-1, 3)

        if guard is not None and guard(dx_ang) is not None:
            n_rejected += 1
            if max_draws is not None and draws >= max_draws:
                n_dropped += 1
                draws = 0
            continue

        draws = 0
        displacements.append(dx_ang)
        energies_kJ.append(float(energy_kJ))
        q_vectors.append(q * _Q_SI_TO_AMU_HALF_ANG)

    return displacements, energies_kJ, q_vectors, n_rejected, n_dropped


def zero_point_energy(frequencies_cm):
    """Harmonic zero-point energy (kJ/mol) from positive frequencies (cm^-1)."""
    freqs = np.asarray(frequencies_cm, dtype=float)
    freqs = freqs[freqs > 0.0]
    # E_ZPE = 1/2 sum h c nu-tilde ; nu-tilde in cm^-1 -> multiply by c[cm/s].
    e_j = 0.5 * np.sum(2.0 * np.pi * _HBAR_J_S * _C_CM_S * freqs)
    return e_j * _N_AVOGADRO / 1000.0


class NormalModeSampling(seamm.Node):
    """The non-graphical part of a Normal Mode Sampling step in a flowchart."""

    def __init__(
        self,
        flowchart=None,
        title="Normal Mode Sampling",
        extension=None,
        logger=logger,
    ):
        """A step for Normal Mode Sampling in a SEAMM flowchart."""
        logger.debug(f"Creating Normal Mode Sampling {self}")

        super().__init__(
            flowchart=flowchart,
            title="Normal Mode Sampling",
            extension=extension,
            module=__name__,
            logger=logger,
        )

        self._metadata = normal_mode_sampling_step.metadata
        self.parameters = normal_mode_sampling_step.NormalModeSamplingParameters()

    @property
    def version(self):
        """The semantic version of this module."""
        return normal_mode_sampling_step.__version__

    @property
    def git_revision(self):
        """The git version of this module."""
        return normal_mode_sampling_step.__git_revision__

    def description_text(self, P=None):
        """Create the text description of what this step will do."""
        if not P:
            P = self.parameters.values_to_dict()

        distribution = P["distribution"]
        n = P["number of samples"]
        if str(n).strip().lower() == "in proportion to the modes":
            count = (
                f"{P['samples per mode']} displaced geometries per sampled mode "
                f"(at least {P['minimum number of samples']}, at most "
                f"{P['maximum number of samples']})"
            )
        else:
            count = f"{n} displaced geometr" + ("y" if str(n) == "1" else "ies")

        key = self._distribution_key(distribution)
        if key == "energy":
            how = (
                "with the equipartition shape, each scaled to a harmonic energy "
                f"drawn uniformly between {P['minimum target energy']} and "
                f"{P['maximum target energy']} above the reference"
            )
        elif key == "ground":
            how = (
                f"from the '{distribution}' normal-mode distribution at 0 K "
                "(zero-point motion only)"
            )
        else:
            how = (
                f"from the '{distribution}' normal-mode distribution at "
                f"{P['temperature']}"
            )

        text = (
            f"Sampling {count} per reference structure {how}, using the Hessian "
            "from the model chemistry defined earlier in the flowchart. "
            "Translational, rotational, and imaginary modes are excluded"
        )
        if not self._is_none(P["lowest frequency"]):
            text += f", as are modes below {P['lowest frequency']}"
        text += "."
        limits = [
            (
                f"no atom moves more than {P['maximum atom displacement']}"
                if not self._is_none(P["maximum atom displacement"])
                else None
            ),
            (
                f"no bond changes by more than {P['maximum bond change']}"
                if not self._is_none(P["maximum bond change"])
                else None
            ),
            (
                f"no bond angle changes by more than {P['maximum angle change']}"
                if not self._is_none(P["maximum angle change"])
                else None
            ),
        ]
        limits = [x for x in limits if x is not None]
        if limits:
            if str(P["maximum draws"]).strip().lower() == "no limit":
                tries = "redrawn until it passes"
            else:
                tries = f"redrawn up to {P['maximum draws']} times in all, then dropped"
            text += (
                " A sample is accepted only if "
                + (", ".join(limits[:-1]) + " and " if len(limits) > 1 else "")
                + limits[-1]
                + f"; otherwise it is {tries}."
            )
        text += (
            f" The structures are written to the system '{P['system name']}' for "
            "downstream single-point evaluation."
        )

        return self.header + "\n" + __(text, indent=4 * " ").__str__()

    def run(self):
        """Run a Normal Mode Sampling step."""
        next_node = super().run(printer)
        P = self.parameters.current_values_to_dict(
            context=seamm.flowchart_variables._data
        )

        printer.important(__(self.description_text(P), indent=self.indent))
        printer.important("")

        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)

        system_db = self.get_variable("_system_db")
        seed = self._seed_of(P["random seed"])
        rng = np.random.default_rng(seed)

        distribution = self._distribution_key(P["distribution"])
        energy_range = None
        if distribution == "energy":
            energy_range = (
                self._as_float(P["minimum target energy"], "kJ/mol"),
                self._as_float(P["maximum target energy"], "kJ/mol"),
            )
        lowest = (
            None
            if self._is_none(P["lowest frequency"])
            else self._as_float(P["lowest frequency"], "1/cm")
        )
        limits = {
            key: (None if self._is_none(P[key]) else self._as_float(P[key], units))
            for key, units in (
                ("maximum atom displacement", "Å"),
                ("maximum bond change", "Å"),
                ("maximum angle change", "degree"),
            )
        }
        max_draws = (
            None
            if str(P["maximum draws"]).strip().lower() == "no limit"
            else int(P["maximum draws"])
        )
        temperature = self._as_float(P["temperature"], "K")
        cap = None if self._is_none(P["amplitude cap"]) else float(P["amplitude cap"])
        ceiling = (
            None
            if self._is_none(P["energy ceiling"])
            else self._as_float(P["energy ceiling"], "kJ/mol")
        )
        if distribution == "energy" and ceiling is not None:
            # The targets set every sample's energy, so a ceiling has no meaning;
            # the GUI hides it, but a hand-edited flowchart may still carry one.
            printer.important(
                __(
                    "The energy ceiling is ignored for energy-targeted sampling: "
                    "the target energies set each sample's energy.",
                    indent=4 * " ",
                )
            )
            ceiling = None

        references = self._reference_pool(P, system_db)
        if len(references) == 0:
            raise ValueError(
                "Normal Mode Sampling found no reference configuration to sample."
            )

        # Build the output system from the first reference's topology.
        out_name = self._output_system_name(P, references[0])
        out_sys = system_db.create_combined_system([references[0]], name=out_name)
        base = out_sys.configuration

        total = 0
        n_rejected_total = 0
        n_dropped_total = 0
        first_freqs = None
        first_zpe = None
        first_counts = None
        energies_all = []

        for ref_index, reference in enumerate(references, start=1):
            hessian_au = self._hessian(reference)
            masses = np.asarray(reference.atoms.atomic_masses, dtype=float)
            coords = reference.atoms.get_coordinates(fractionals=False, as_array=True)
            coords = np.asarray(coords, dtype=float)

            freqs_cm, modes_mw = normal_mode_analysis(hessian_au, masses, coords)

            imaginary = freqs_cm < -ZERO_MODE_THRESHOLD_CM
            low = np.abs(freqs_cm) <= ZERO_MODE_THRESHOLD_CM
            vibrational = freqs_cm > ZERO_MODE_THRESHOLD_CM

            if imaginary.any():
                worst = freqs_cm[imaginary]
                printer.important(
                    __(
                        f"Warning: reference {ref_index} has "
                        f"{int(imaginary.sum())} imaginary frequenc"
                        + ("y" if imaginary.sum() == 1 else "ies")
                        + f" (down to {worst.min():.1f}i cm^-1). These are not "
                        "sampled; the geometry is not a minimum (fine for a "
                        "transition state).",
                        indent=4 * " ",
                    )
                )

            vib_freqs = freqs_cm[vibrational]
            vib_modes = modes_mw[:, vibrational]

            keep = self._parse_mode_selection(P["modes"], vib_freqs.size)
            if lowest is not None:
                keep &= vib_freqs >= lowest
            sel_freqs = vib_freqs[keep]
            sel_modes = vib_modes[:, keep]
            if sel_freqs.size == 0:
                raise ValueError(
                    f"Reference {ref_index} has no vibrational modes to sample "
                    "after applying the mode selection."
                )

            guard = None
            if any(v is not None for v in limits.values()):
                guard = geometry_guard(
                    coords,
                    self._bonds_of(reference),
                    max_displacement=limits["maximum atom displacement"],
                    max_bond_change=limits["maximum bond change"],
                    max_angle_change=limits["maximum angle change"],
                )

            (
                displacements,
                energies,
                q_vectors,
                n_rejected,
                n_dropped,
            ) = sample_normal_modes(
                sel_freqs,
                sel_modes,
                masses,
                self._number_of_samples(P, sel_freqs.size),
                distribution=distribution,
                temperature=temperature,
                amplitude_cap=cap,
                energy_ceiling=ceiling,
                energy_range=energy_range,
                guard=guard,
                max_draws=max_draws,
                rng=rng,
            )
            n_rejected_total += n_rejected
            n_dropped_total += n_dropped
            energies_all.extend(energies)

            for i, dx in enumerate(displacements):
                conf = base if total == 0 else out_sys.copy_configuration(base)
                conf.atoms.set_coordinates(coords + dx, fractionals=False)
                total += 1
                conf.name = self._configuration_name(
                    P["configuration name"], ref_index, total, len(displacements)
                )
                # Per-structure provenance, so downstream steps can filter on the
                # predicted energy and inspect the normal-mode displacement.
                self._put_property(
                    conf, _HARMONIC_ENERGY_PROPERTY, float(energies[i]), "kJ/mol"
                )
                self._put_property(
                    conf,
                    _DISPLACEMENT_PROPERTY,
                    [float(x) for x in q_vectors[i]],
                    "Å*u^1/2",
                    _type="json",
                )

            if ref_index == 1:
                first_freqs = vib_freqs
                first_zpe = zero_point_energy(vib_freqs)
                first_counts = {
                    "vibrational": int(vibrational.sum()),
                    "sampled": int(sel_freqs.size),
                    "imaginary": int(imaginary.sum()),
                    "low": int(low.sum()),
                }

        # Make the new system and its first configuration current.
        system_db.system = out_sys
        out_sys.configuration = out_sys.configurations[0].id

        # Cite the method. The plug-in's own reference is added by the base class,
        # which has already loaded data/references.bib into self._bibliography.
        if distribution in ("wigner", "ground") and "wigner1932" in self._bibliography:
            self.references.cite(
                raw=self._bibliography["wigner1932"],
                alias="wigner1932",
                module="normal_mode_sampling_step",
                level=1,
                note="The quantum (Wigner) phase-space distribution used to set "
                "the normal-mode sampling amplitudes.",
            )

        data = {
            "number of samples": total,
            "number of rejected samples": n_rejected_total,
            "number of dropped samples": n_dropped_total,
            "random seed": seed,
        }
        if first_freqs is not None:
            data["frequencies"] = [float(v) for v in first_freqs]
            data["zero point energy"] = float(first_zpe)
            data["number of vibrational modes"] = first_counts["vibrational"]
            data["number of sampled modes"] = first_counts["sampled"]
            data["number of imaginary modes"] = first_counts["imaginary"]
            data["number of low modes"] = first_counts["low"]
        if energies_all:
            data["mean harmonic energy"] = float(np.mean(energies_all))
            data["maximum harmonic energy"] = float(np.max(energies_all))

        self.analyze(P=P, data=data, out_name=out_name, n_refs=len(references))
        self.store_results(configuration=base, data=data)

        return next_node

    # ------------------------------------------------------------------ #
    # Hessian acquisition (Model Chemistry over MDI)
    # ------------------------------------------------------------------ #
    def _hessian(self, configuration):
        """Return the Cartesian Hessian (hartree/bohr^2) for a configuration.

        Uses the ``_model_chemistry`` published upstream, through the shared
        driver in seamm_exec.hessian (also used by the Thermochemistry and
        Vibrational Analysis steps): analytic over the MDI engine when the
        program offers one, otherwise central finite differences of the forces
        over the warm engine or as tasks where the job runs them.
        """
        from seamm_exec.hessian import HessianDriver

        driver = HessianDriver(
            self,
            who="Normal Mode Sampling",
            print_to=printer,
        )
        if configuration.periodicity != 0:
            raise ValueError(
                "Normal Mode Sampling supports only non-periodic (molecular) "
                "systems."
            )
        _, H, _ = driver.energy_and_hessian(configuration)
        return H

    def _reference_pool(self, P, system_db):
        """Resolve the 'structure' parameters to a list of configurations."""
        spec = P["structure"]
        if isinstance(spec, (list, tuple)):
            return list(spec)

        if isinstance(spec, str) and (spec == "" or spec.lower() == "current"):
            system = system_db.system
        else:
            system = system_db.get_system(spec)

        how = P["structure configurations"]
        name = P["structure configuration name"]
        if how == "current":
            return [system.configuration]
        return self._select_configurations(system, how, name)

    @staticmethod
    def _select_configurations(system, how, name):
        """Pick configurations from a system, mirroring the loop/dimer steps."""
        import fnmatch
        import re

        configurations = system.configurations
        if how == "all":
            return configurations
        if how == "last":
            return [configurations[-1]]
        if how == "first":
            return [configurations[0]]
        if how == "name is":
            return [c for c in configurations if c.name == name]
        if how == "name matches":
            return [c for c in configurations if fnmatch.fnmatch(c.name, name)]
        if how == "name regexp":
            pattern = re.compile(name)
            return [c for c in configurations if pattern.search(c.name)]
        raise ValueError(f"Unknown configuration selector '{how}'.")

    def _output_system_name(self, P, reference):
        name = P["system name"]
        if name == "from structure" or name.strip() == "":
            base = reference.system.name or "structure"
            return f"{base} - NMS"
        return name

    @staticmethod
    def _configuration_name(naming, ref_index, running_index, per_ref):
        if naming == "sample number":
            within = (running_index - 1) % per_ref + 1
            return str(within)
        if naming == "reference,sample":
            within = (running_index - 1) % per_ref + 1
            return f"{ref_index},{within}"
        return str(running_index)

    def _put_property(self, conf, name, value, units, _type="float"):
        """Store a property on a configuration (defining it if not registered).

        The properties are normally pre-registered from data/properties.csv; the
        fallback definition here covers a configuration whose database lacks that
        registration.
        """
        properties = conf.properties
        if not properties.exists(name):
            properties.add(name, _type, units=units, noerror=True)
        properties.put(name, value)

    @staticmethod
    def _seed_of(seed):
        """The integer seed: the given one, or a fresh one for 'random', so that
        the ensemble can be reproduced from the recorded seed."""
        if isinstance(seed, str):
            if seed.strip() == "" or seed.strip().lower() == "random":
                return int(np.random.SeedSequence().entropy % 2**63)
            seed = int(seed)
        return int(seed)

    @staticmethod
    def _number_of_samples(P, n_modes):
        """The number of samples for a reference with n_modes sampled modes."""
        n = P["number of samples"]
        if str(n).strip().lower() != "in proportion to the modes":
            return int(n)
        wanted = int(np.floor(float(P["samples per mode"]) * n_modes + 0.5))
        low, high = int(P["minimum number of samples"]), int(
            P["maximum number of samples"]
        )
        return max(low, min(high, wanted))

    @staticmethod
    def _bonds_of(configuration):
        """The bonds (0-based index pairs) of a configuration: its bond table,
        or, when it has none, bonds perceived from the distances (covalent
        radii x 1.3, RDKit's radii)."""
        ids = list(configuration.atoms.ids)
        index = {aid: k for k, aid in enumerate(ids)}
        if configuration.bonds.n_bonds > 0:
            data = configuration.bonds.get_as_dict()
            return [(index[i], index[j]) for i, j in zip(data["i"], data["j"])]
        from rdkit import Chem

        table = Chem.GetPeriodicTable()
        symbols = list(configuration.atoms.symbols)
        radii = np.array([table.GetRcovalent(s) for s in symbols])
        x = np.asarray(
            configuration.atoms.get_coordinates(fractionals=False), dtype=float
        )
        bonds = []
        for i in range(len(symbols)):
            d = np.linalg.norm(x[i + 1 :] - x[i], axis=1)
            for k in np.nonzero(d < 1.3 * (radii[i + 1 :] + radii[i]))[0]:
                bonds.append((i, i + 1 + int(k)))
        return bonds

    def _make_rng(self, seed):
        if isinstance(seed, str):
            if seed.strip() == "" or seed.strip().lower() == "random":
                return np.random.default_rng()
            seed = int(seed)
        return np.random.default_rng(int(seed))

    @staticmethod
    def _distribution_key(value):
        v = str(value).lower()
        if "energy" in v:
            return "energy"
        if "classical" in v:
            return "classical"
        if "ground" in v:
            return "ground"
        return "wigner"

    @staticmethod
    def _is_none(value):
        return isinstance(value, str) and value.strip().lower() in ("none", "")

    @staticmethod
    def _as_float(value, units):
        """Coerce a parameter (possibly a pint quantity) to a float in units."""
        try:
            return value.m_as(units)
        except AttributeError:
            return float(value)

    @staticmethod
    def _parse_mode_selection(spec, n_modes):
        """Return a boolean mask over the ``n_modes`` vibrational modes.

        ``spec`` is 'all' or a comma/space list of 1-based indices and ranges
        (e.g. '1-3, 7'), indexing the vibrational modes by increasing frequency.
        """
        if spec is None or str(spec).strip().lower() in ("all", ""):
            return np.ones(n_modes, dtype=bool)
        mask = np.zeros(n_modes, dtype=bool)
        for token in str(spec).replace(",", " ").split():
            if "-" in token:
                lo, hi = token.split("-", 1)
                lo, hi = int(lo), int(hi)
            else:
                lo = hi = int(token)
            for i in range(lo, hi + 1):
                if 1 <= i <= n_modes:
                    mask[i - 1] = True
        return mask

    def analyze(self, P=None, data=None, out_name="", n_refs=1, indent="", **kwargs):
        """Summarize what was generated to the step's output."""
        if data is None:
            return

        text = (
            f"Created {data['number of samples']} configurations in the system "
            f"'{out_name}' from {n_refs} reference "
            + ("structure" if n_refs == 1 else "structures")
            + "."
        )
        printer.important(__(text, indent=4 * " "))

        if "number of vibrational modes" in data:
            text = (
                f"The first reference has {data['number of vibrational modes']} "
                f"vibrational modes ({data['number of sampled modes']} sampled, "
                f"{data['number of imaginary modes']} imaginary), with a harmonic "
                f"zero-point energy of {data['zero point energy']:.1f} kJ/mol."
            )
            printer.important(__(text, indent=4 * " "))

        if "mean harmonic energy" in data:
            text = (
                f"The sampled ensemble has a mean harmonic energy of "
                f"{data['mean harmonic energy']:.1f} kJ/mol above the reference "
                f"(maximum {data['maximum harmonic energy']:.1f} kJ/mol)"
                + (
                    f"; {data['number of rejected samples']} draws were rejected "
                    "by the energy ceiling or the limits on the geometry."
                    if data.get("number of rejected samples")
                    else "."
                )
            )
            printer.important(__(text, indent=4 * " "))
        if data.get("number of dropped samples"):
            text = (
                f"{data['number of dropped samples']} samples were dropped because "
                "every draw moved the structure more than the limits allow."
            )
            printer.important(__(text, indent=4 * " "))
        if "random seed" in data:
            printer.important(
                __(f"The random seed was {data['random seed']}.", indent=4 * " ")
            )
        printer.important("")
