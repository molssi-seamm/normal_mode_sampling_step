# -*- coding: utf-8 -*-

"""This file contains metadata describing the results from NormalModeSampling"""

metadata = {}

"""Properties that Normal Mode Sampling produces.

`metadata["results"]` describes the results that this step can produce. It is a
dictionary where the keys are the internal names of the results within this step,
and the values are a dictionary describing the result. See the SEAMM developer
documentation (or e.g. the Thermochemistry step) for the full field reference.

Normal Mode Sampling does not carry a computational model or keywords of its own
-- the energy/Hessian come from the Model Chemistry published upstream -- so only
``results`` is populated here. The storable scalar (the zero-point energy) is
defined in ``data/properties.csv``; ``{model}`` is filled from the model
chemistry at run time. The frequencies are exposed as a retrievable result (to a
variable or table) but not stored as a per-configuration property.
"""

metadata["results"] = {
    "number of samples": {
        "description": "Number of sampled configurations generated",
        "dimensionality": "scalar",
        "type": "integer",
    },
    "number of vibrational modes": {
        "description": "Number of real vibrational modes available to sample",
        "dimensionality": "scalar",
        "type": "integer",
    },
    "number of sampled modes": {
        "description": "Number of modes actually displaced along",
        "dimensionality": "scalar",
        "type": "integer",
    },
    "number of imaginary modes": {
        "description": (
            "Number of imaginary-frequency modes found (0 for a minimum, 1 for a "
            "transition state); these are reported but not sampled"
        ),
        "dimensionality": "scalar",
        "type": "integer",
    },
    "number of low modes": {
        "description": (
            "Number of translational, rotational, and near-zero modes projected "
            "out before sampling"
        ),
        "dimensionality": "scalar",
        "type": "integer",
    },
    "frequencies": {
        "description": "The vibrational frequencies used for sampling",
        "dimensionality": "[n_vibrational_modes]",
        "type": "float",
        "units": "cm^-1",
    },
    "zero point energy": {
        "description": "The harmonic zero-point vibrational energy",
        "dimensionality": "scalar",
        "property": "zero point energy#NormalModeSampling#{model}",
        "type": "float",
        "units": "kJ/mol",
        "format": ".3f",
    },
    "mean harmonic energy": {
        "description": (
            "Mean harmonic energy of the sampled ensemble, above the reference"
        ),
        "dimensionality": "scalar",
        "type": "float",
        "units": "kJ/mol",
        "format": ".3f",
    },
    "maximum harmonic energy": {
        "description": (
            "Largest harmonic energy above the reference among the accepted " "samples"
        ),
        "dimensionality": "scalar",
        "type": "float",
        "units": "kJ/mol",
        "format": ".3f",
    },
    "number of rejected samples": {
        "description": ("Number of draws rejected by the energy ceiling and redrawn"),
        "dimensionality": "scalar",
        "type": "integer",
    },
}
