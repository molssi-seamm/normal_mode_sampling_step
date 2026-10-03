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
    ``wigner``, ``classical``, ``ground``.
    """
    if distribution == "classical":
        # <Q^2> = kT / omega^2 for a unit-mass harmonic oscillator.
        return _KB_J_K * temperature / omega**2
    zpe_var = _HBAR_J_S / (2.0 * omega)  # ground-state (T -> 0) width
    if distribution == "ground":
        return zpe_var
    # Wigner (quantum) width at finite T.
    x = _HBAR_J_S * omega / (2.0 * _KB_J_K * temperature)
    return zpe_var / np.tanh(x)


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
        Number of accepted geometries to return.
    distribution : {"wigner", "classical", "ground"}
    temperature : float
        Temperature (K); ignored for the ground-state distribution.
    amplitude_cap : float or None
        Clamp each normal-coordinate draw to this many standard deviations.
    energy_ceiling : float or None
        Reject (and redraw) samples whose harmonic energy above the reference
        (kJ/mol) exceeds this.
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
        Number of draws rejected by the energy ceiling.
    """
    if rng is None:
        rng = np.random.default_rng()

    freqs = np.asarray(frequencies_cm, dtype=float)
    modes = np.asarray(modes_mw, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    n_modes = freqs.size

    omega = 2.0 * np.pi * _C_CM_S * freqs  # angular frequency, 1/s
    sigma = np.sqrt(
        np.array([_sigma_squared(w, distribution, temperature) for w in omega])
    )  # sqrt(kg) m, per mode

    inv_sqrt_m3 = 1.0 / np.sqrt(np.repeat(masses, 3) * _AMU_KG)  # 1/sqrt(kg)

    displacements = []
    energies_kJ = []
    q_vectors = []
    n_rejected = 0
    max_tries = max(1, max_tries_factor) * max(1, n_samples)
    tries = 0
    while len(displacements) < n_samples and tries < max_tries:
        tries += 1
        q = rng.standard_normal(n_modes) * sigma  # normal coords, sqrt(kg) m
        if amplitude_cap is not None:
            q = np.clip(q, -amplitude_cap * sigma, amplitude_cap * sigma)

        # Harmonic energy E = 1/2 sum omega^2 q^2 (unit mass), J -> kJ/mol.
        energy_kJ = 0.5 * np.sum(omega**2 * q**2) * _N_AVOGADRO / 1000.0
        if energy_ceiling is not None and energy_kJ > energy_ceiling:
            n_rejected += 1
            continue

        # Mass-weighted Cartesian displacement, then de-weight to metres -> Ang.
        dxi = modes @ q  # (3N,), sqrt(kg) m
        dx_m = dxi * inv_sqrt_m3  # (3N,), m
        dx_ang = (dx_m / 1.0e-10).reshape(-1, 3)

        displacements.append(dx_ang)
        energies_kJ.append(energy_kJ)
        q_vectors.append(q * _Q_SI_TO_AMU_HALF_ANG)

    return displacements, energies_kJ, q_vectors, n_rejected


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
        if isinstance(distribution, str) and "ground" in distribution:
            temp = "0 K (zero-point motion only)"
        else:
            temp = f"{P['temperature']}"

        text = (
            f"Sampling {n} displaced geometr"
            + ("y" if str(n) == "1" else "ies")
            + f" per reference structure from the '{distribution}' normal-mode "
            f"distribution at {temp}, using the Hessian from the model chemistry "
            "defined earlier in the flowchart. Translational, rotational, and "
            "imaginary modes are excluded. The structures are written to the "
            f"system '{P['system name']}' for downstream single-point evaluation."
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
        rng = self._make_rng(P["random seed"])

        distribution = self._distribution_key(P["distribution"])
        temperature = self._as_float(P["temperature"], "K")
        cap = None if self._is_none(P["amplitude cap"]) else float(P["amplitude cap"])
        ceiling = (
            None
            if self._is_none(P["energy ceiling"])
            else self._as_float(P["energy ceiling"], "kJ/mol")
        )

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
            sel_freqs = vib_freqs[keep]
            sel_modes = vib_modes[:, keep]
            if sel_freqs.size == 0:
                raise ValueError(
                    f"Reference {ref_index} has no vibrational modes to sample "
                    "after applying the mode selection."
                )

            displacements, energies, q_vectors, n_rejected = sample_normal_modes(
                sel_freqs,
                sel_modes,
                masses,
                int(P["number of samples"]),
                distribution=distribution,
                temperature=temperature,
                amplitude_cap=cap,
                energy_ceiling=ceiling,
                rng=rng,
            )
            n_rejected_total += n_rejected
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
        if "wigner1932" in self._bibliography:
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

        Uses the ``_model_chemistry`` published upstream, driven over MDI:
        analytic if the engine exposes a ``hessian`` method (a future custom
        ``<HESSIAN`` command), otherwise central finite differences of the
        forces over the resident engine.
        """
        if not self.variable_exists("_model_chemistry"):
            raise ValueError(
                "Normal Mode Sampling needs a Model Chemistry: add a 'Model "
                "Chemistry' step before it to choose the method for the Hessian."
            )
        if configuration.periodicity != 0:
            raise ValueError(
                "Normal Mode Sampling supports only non-periodic (molecular) "
                "systems."
            )

        mc = self.get_variable("_model_chemistry")
        options = mc["options"]
        from seamm_exec import Evaluator

        try:
            evaluator = Evaluator(self, mc, properties=("energy", "gradients"))
        except ValueError:
            raise ValueError(
                f"The model chemistry '{mc['level']}' can be evaluated neither over "
                "MDI nor as separate calculations, so it cannot give the Hessian. "
                "Choose a model chemistry with an MDI engine or with tasks (e.g. "
                "MOPAC, xTB or ORCA)."
            )

        step = self.flowchart.plugin_manager.get(mc["step"])
        # The engine's real keyword (an ORCA functional's alias undone) and the
        # user's basis, for programs that take one; MOPAC, xTB and MLFFs take
        # a method alone.
        method = options.get("mdi_method_arg") or mc["method"]
        basis = options.get("mdi_basis_arg")

        def build_argv(hostname, port):
            kwargs = {}
            if basis is not None:
                kwargs["basis"] = basis
            return step.get_mdi_engine_command(
                self.flowchart.executor,
                self.global_options,
                method=method,
                port=port,
                hostname=hostname,
                charge=configuration.charge,
                multiplicity=configuration.spin_multiplicity,
                n_atoms=configuration.n_atoms,
                **kwargs,
            )

        elements = list(configuration.atoms.atomic_numbers)
        coords_ang = np.asarray(
            configuration.atoms.get_coordinates(fractionals=False, as_array=True),
            dtype=float,
        )

        n = len(elements)
        if options.get("mdi_capable", False):
            from seamm_mdi import MDIEngine  # only for the MDI routes

            try:
                eng = MDIEngine(build_argv, elements, name="SEAMM", logger=self.logger)
                eng.start()
            except Exception as e:
                # No engine here (e.g. the code is installed only on the job's
                # cluster): the finite difference as tasks, if the program can.
                if not hasattr(evaluator.provider, "get_task"):
                    raise
                self.logger.warning(
                    f"Could not start the MDI engine ({e}); computing the Hessian "
                    "by finite differences as separate calculations."
                )
                return self._fd_hessian_tasks(evaluator, configuration, coords_ang)
            with eng:
                eng.set_coordinates(coords_ang, units="Å")
                # The engine advertises <HESSIAN only when it has a genuine
                # analytic Hessian for this method, so supports() is the truthful
                # capability check: use the analytic Hessian when offered.
                if eng.supports("<HESSIAN"):
                    # Analytic Hessian: (3N, 3N) in hartree/bohr^2.
                    return np.asarray(eng.hessian(), dtype=float).reshape(3 * n, 3 * n)
                # Otherwise finite-difference the forces: over the warm engine,
                # unless the evaluator runs this model chemistry as tasks.
                if evaluator.path != "batch":
                    return self._fd_hessian(eng, coords_ang, n)
        return self._fd_hessian_tasks(evaluator, configuration, coords_ang)

    def _fd_hessian_tasks(self, evaluator, configuration, coords_ang, delta_bohr=0.01):
        """Central-difference Hessian (hartree/bohr^2) from the gradients of the
        6N displaced structures, computed together as tasks (concurrently, or on
        the job's queue; a rerun keeps the finished ones)."""
        from seamm_exec import Geometry

        elements = list(configuration.atoms.atomic_numbers)
        x0 = np.asarray(coords_ang, dtype=float).reshape(-1)
        delta = Q_(delta_bohr, "bohr").m_as("Å")
        ndof = x0.size
        prefix = f"c{getattr(configuration, 'id', 0)}-fd"
        for j in range(ndof):
            for sign, tag in ((1.0, "p"), (-1.0, "m")):
                x = x0.copy()
                x[j] += sign * delta
                evaluator.submit(
                    Geometry(
                        elements,
                        x.reshape(-1, 3),
                        configuration.charge,
                        configuration.spin_multiplicity,
                    ),
                    key=f"{prefix}{j:04d}{tag}",
                )
        gradients = {}
        for result in evaluator.results():
            if not result.ok:
                raise RuntimeError(
                    f"A displaced structure for the Hessian failed ({result.key}): "
                    f"{result.reason}"
                )
            gradients[result.key] = (
                Q_(np.asarray(result.gradients, dtype=float), "kJ/mol/Å")
                .m_as("hartree/bohr")
                .reshape(-1)
            )
        H = np.zeros((ndof, ndof))
        for j in range(ndof):
            gp = gradients[f"{prefix}{j:04d}p"]
            gm = gradients[f"{prefix}{j:04d}m"]
            H[:, j] = (gp - gm) / (2.0 * delta_bohr)
        return 0.5 * (H + H.T)

    def _fd_hessian(self, engine, coords_ang, n_atoms, delta_bohr=0.01):
        """Central-difference Hessian (hartree/bohr^2) from MDI forces."""
        x0 = Q_(coords_ang, "Å").m_as("bohr").reshape(-1)
        ndof = 3 * n_atoms
        H = np.zeros((ndof, ndof))
        for j in range(ndof):
            xp = x0.copy()
            xp[j] += delta_bohr
            engine.set_coordinates(xp, units="bohr")
            fp = np.asarray(engine.forces(units="hartree/bohr"), dtype=float).reshape(
                -1
            )

            xm = x0.copy()
            xm[j] -= delta_bohr
            engine.set_coordinates(xm, units="bohr")
            fm = np.asarray(engine.forces(units="hartree/bohr"), dtype=float).reshape(
                -1
            )

            # Force = -dE/dx, so d^2E/dx_i dx_j = -(f_i(+) - f_i(-)) / 2 delta.
            H[:, j] = -(fp - fm) / (2.0 * delta_bohr)
        return 0.5 * (H + H.T)

    # ------------------------------------------------------------------ #
    # Input selection, naming, and small parameter helpers
    # ------------------------------------------------------------------ #
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

    def _make_rng(self, seed):
        if isinstance(seed, str):
            if seed.strip() == "" or seed.strip().lower() == "random":
                return np.random.default_rng()
            seed = int(seed)
        return np.random.default_rng(int(seed))

    @staticmethod
    def _distribution_key(value):
        v = str(value).lower()
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
                    "by the energy ceiling."
                    if data.get("number of rejected samples")
                    else "."
                )
            )
            printer.important(__(text, indent=4 * " "))
        printer.important("")
