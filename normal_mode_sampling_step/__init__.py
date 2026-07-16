# -*- coding: utf-8 -*-

"""
normal_mode_sampling_step
A SEAMM plug-in for Wigner/thermal normal-mode sampling of the Hessian, to
generate displaced structures (e.g. for MLFF training sets).
"""

# Bring up the classes so that they appear to be directly in
# the normal_mode_sampling_step package.

from .normal_mode_sampling import NormalModeSampling  # noqa: F401, E501
from .normal_mode_sampling_parameters import NormalModeSamplingParameters  # noqa: F401
from .normal_mode_sampling_step import NormalModeSamplingStep  # noqa: F401, E501
from .tk_normal_mode_sampling import TkNormalModeSampling  # noqa: F401, E501

from .metadata import metadata  # noqa: F401

# Handle versioneer
from ._version import get_versions

__author__ = "Paul Saxe"
__email__ = "psaxe@molssi.org"
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
