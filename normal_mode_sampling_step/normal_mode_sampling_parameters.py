# -*- coding: utf-8 -*-
"""
Control parameters for the Normal Mode Sampling step in a SEAMM flowchart
"""

import logging
import seamm

logger = logging.getLogger(__name__)


class NormalModeSamplingParameters(seamm.Parameters):
    """
    The control parameters for Normal Mode Sampling.

    The keys are the parameters for this plug-in; each value is a dictionary
    describing that parameter (default, kind, units, enumeration, format,
    description, help). The Hessian is obtained at run time from the Model
    Chemistry published by an upstream Model Chemistry step (the
    ``_model_chemistry`` workspace variable) -- there is deliberately no
    method/basis parameter here.

    parameters : {str: {str: str}}
        A dictionary containing the parameters for the current step.
    """

    parameters = {
        # ------------------------------------------------------------------ #
        # Input: the reference structure(s) to sample around
        # ------------------------------------------------------------------ #
        "structure": {
            "default": "current",
            "kind": "string",
            "default_units": "",
            "enumeration": ("current",),
            "format_string": "",
            "description": "Reference structure:",
            "help_text": (
                "The source of the reference structure(s) whose normal modes are "
                "sampled: 'current' for the current system, a system name, or a "
                "variable ($name) holding a list of configurations. The geometry "
                "should be a stationary point (a minimum, or a transition state) "
                "of the chosen Model Chemistry."
            ),
        },
        "structure configurations": {
            "default": "current",
            "kind": "string",
            "default_units": "",
            "enumeration": (
                "current",
                "all",
                "last",
                "first",
                "name is",
                "name matches",
                "name regexp",
            ),
            "format_string": "",
            "description": "using configurations:",
            "help_text": (
                "Which configuration(s) of the structure system to sample around. "
                "Each selected configuration is treated as an independent "
                "reference and gets its own set of samples. Ignored when the "
                "structure is a variable holding a list of configurations (all of "
                "them are used)."
            ),
        },
        "structure configuration name": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "matching:",
            "help_text": (
                "The configuration name or pattern, used with 'name is', 'name "
                "matches', or 'name regexp'."
            ),
        },
        # ------------------------------------------------------------------ #
        # Sampling: how many, and with what amplitude distribution
        # ------------------------------------------------------------------ #
        "number of samples": {
            "default": 20,
            "kind": "integer",
            "default_units": "",
            "enumeration": ("in proportion to the modes",),
            "format_string": "",
            "description": "Number of samples:",
            "help_text": (
                "How many displaced geometries to generate per reference "
                "structure: a number, or 'in proportion to the modes' for the "
                "number of sampled modes times 'samples per mode', kept between "
                "the minimum and maximum below."
            ),
        },
        "samples per mode": {
            "default": 0.5,
            "kind": "float",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": ".2f",
            "description": "Samples per mode:",
            "help_text": (
                "With 'in proportion to the modes': the number of samples per "
                "sampled vibrational mode, rounded."
            ),
        },
        "minimum number of samples": {
            "default": 10,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "At least:",
            "help_text": "With 'in proportion to the modes': the fewest samples.",
        },
        "maximum number of samples": {
            "default": 40,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "At most:",
            "help_text": "With 'in proportion to the modes': the most samples.",
        },
        "distribution": {
            "default": "Wigner (quantum)",
            "kind": "enum",
            "default_units": "",
            "enumeration": (
                "Wigner (quantum)",
                "classical (thermal)",
                "ground state (0 K)",
                "energy-targeted (equipartition)",
            ),
            "format_string": "",
            "description": "Amplitude distribution:",
            "help_text": (
                "The distribution of normal-coordinate displacements. 'Wigner "
                "(quantum)' uses the quantum harmonic-oscillator width "
                "sigma^2 = (hbar/2 omega) coth(hbar omega / 2 kT), which captures "
                "the zero-point spread of stiff modes that classical sampling "
                "misses. 'classical (thermal)' uses the Boltzmann width "
                "sigma^2 = kT / omega^2. 'ground state (0 K)' is the Wigner "
                "distribution at T=0, i.e. pure zero-point motion. "
                "'energy-targeted (equipartition)' draws the classical shape "
                "(the same expected energy in every mode, sigma proportional to "
                "1/omega) and scales each draw so that its harmonic energy is a "
                "target drawn uniformly between the minimum and maximum target "
                "energies: every sample lies in a chosen energy window, whatever "
                "the size of the molecule, and no temperature enters."
            ),
        },
        "minimum target energy": {
            "default": 0.5,
            "kind": "float",
            "default_units": "kcal/mol",
            "enumeration": tuple(),
            "format_string": ".2f",
            "description": "Target energies from:",
            "help_text": (
                "For energy-targeted sampling: the lowest harmonic energy above "
                "the reference that a sample is scaled to."
            ),
        },
        "maximum target energy": {
            "default": 9.0,
            "kind": "float",
            "default_units": "kcal/mol",
            "enumeration": tuple(),
            "format_string": ".2f",
            "description": "to:",
            "help_text": (
                "For energy-targeted sampling: the highest harmonic energy above "
                "the reference that a sample is scaled to."
            ),
        },
        "temperature": {
            "default": 300.0,
            "kind": "float",
            "default_units": "K",
            "enumeration": tuple(),
            "format_string": ".1f",
            "description": "Temperature:",
            "help_text": (
                "The temperature entering the amplitude distribution. Not used "
                "for the 'ground state (0 K)' distribution."
            ),
        },
        "amplitude cap": {
            "default": 4.0,
            "kind": "float",
            "default_units": "",
            "enumeration": ("none",),
            "format_string": ".1f",
            "description": "Amplitude cap (in sigma):",
            "help_text": (
                "Clamp each normal-coordinate displacement to this many standard "
                "deviations, so no sample wanders toward dissociation. 'none' "
                "leaves the Gaussian tails uncapped."
            ),
        },
        "energy ceiling": {
            "default": "none",
            "kind": "float",
            "default_units": "kJ/mol",
            "enumeration": ("none",),
            "format_string": ".1f",
            "description": "Reject harmonic energy above:",
            "help_text": (
                "Reject (and redraw) any sample whose harmonic energy above the "
                "reference exceeds this ceiling. 'none' accepts every draw."
            ),
        },
        "modes": {
            "default": "all",
            "kind": "string",
            "default_units": "",
            "enumeration": ("all",),
            "format_string": "",
            "description": "Modes to sample:",
            "help_text": (
                "Which vibrational modes to displace along: 'all', or a list / "
                "range of 1-based mode indices ordered by increasing frequency "
                "(e.g. '1-3, 7'). Translational, rotational, and imaginary modes "
                "are always excluded, so a transition state is sampled along its "
                "real modes with the reaction coordinate left alone."
            ),
        },
        "lowest frequency": {
            "default": "none",
            "kind": "float",
            "default_units": "1/cm",
            "enumeration": ("none",),
            "format_string": ".1f",
            "description": "Skip modes below:",
            "help_text": (
                "Do not displace along vibrational modes with a lower frequency "
                "than this, e.g. 100 cm^-1 to leave soft torsions and other "
                "large-amplitude motions to a torsion scan. 'none' samples every "
                "selected mode."
            ),
        },
        "maximum atom displacement": {
            "default": "none",
            "kind": "float",
            "default_units": "Å",
            "enumeration": ("none",),
            "format_string": ".3f",
            "description": "Largest atom displacement:",
            "help_text": (
                "Redraw a sample in which any atom moves farther than this from "
                "the reference. 'none' sets no limit."
            ),
        },
        "maximum bond change": {
            "default": "none",
            "kind": "float",
            "default_units": "Å",
            "enumeration": ("none",),
            "format_string": ".3f",
            "description": "Largest bond-length change:",
            "help_text": (
                "Redraw a sample in which any bond length changes by more than "
                "this. The bonds are the reference's (perceived from the "
                "distances when it has none). 'none' sets no limit."
            ),
        },
        "maximum angle change": {
            "default": "none",
            "kind": "float",
            "default_units": "degree",
            "enumeration": ("none",),
            "format_string": ".1f",
            "description": "Largest bond-angle change:",
            "help_text": (
                "Redraw a sample in which any angle between two bonds changes by "
                "more than this. 'none' sets no limit."
            ),
        },
        "maximum draws": {
            "default": 5,
            "kind": "integer",
            "default_units": "",
            "enumeration": ("no limit",),
            "format_string": "",
            "description": "Draws per sample:",
            "help_text": (
                "With any of the limits above: how many times a sample is drawn "
                "before it is dropped (and counted as dropped). 'no limit' "
                "redraws until a sample passes."
            ),
        },
        "random seed": {
            "default": "random",
            "kind": "string",
            "default_units": "",
            "enumeration": ("random",),
            "format_string": "",
            "description": "Random seed:",
            "help_text": (
                "The seed for the random-number generator. Use 'random' for a "
                "fresh, non-reproducible seed, or an integer for a reproducible "
                "ensemble."
            ),
        },
        # ------------------------------------------------------------------ #
        # Output: a new system of sampled configurations
        # ------------------------------------------------------------------ #
        "system name": {
            "default": "from structure",
            "kind": "string",
            "default_units": "",
            "enumeration": ("from structure",),
            "format_string": "",
            "description": "Name the sampled system:",
            "help_text": (
                "The name for the new system holding the sampled configurations. "
                "'from structure' derives it from the reference system's name; "
                "otherwise the literal text is used."
            ),
        },
        "configuration name": {
            "default": "sequential",
            "kind": "string",
            "default_units": "",
            "enumeration": ("sequential", "sample number", "reference,sample"),
            "format_string": "",
            "description": "Name the configurations:",
            "help_text": (
                "How to name each generated configuration. 'sequential' numbers "
                "them 1, 2, ... across all references; 'sample number' numbers "
                "them within each reference; 'reference,sample' labels them by "
                "reference and sample index (1,1  1,2  ...) so the samples of one "
                "reference group together. A comma (not '/') is used since '/' "
                "separates system and configuration names elsewhere."
            ),
        },
        "results": {
            "default": {},
            "kind": "dictionary",
            "default_units": None,
            "enumeration": tuple(),
            "format_string": "",
            "description": "results",
            "help_text": "The results to save to variables or in tables.",
        },
    }

    def __init__(self, defaults={}, data=None):
        """
        Initialize the parameters, by default with the parameters defined above

        Parameters
        ----------
        defaults: dict
            A dictionary of parameters to initialize. The parameters
            above are used first and any given will override/add to them.
        data: dict
            A dictionary of keys and a subdictionary with value and units
            for updating the current, default values.

        Returns
        -------
        None
        """

        logger.debug("NormalModeSamplingParameters.__init__")

        super().__init__(
            defaults={
                **NormalModeSamplingParameters.parameters,
                **defaults,
            },
            data=data,
        )
