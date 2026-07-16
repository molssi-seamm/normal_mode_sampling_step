2026-07-15 Normal Mode Sampling -- initial development
======================================================

Build the **Normal Mode Sampling** plug-in: given a molecule and its Hessian,
draw an ensemble of displaced geometries by **Wigner (quantum) normal-mode
sampling** and write them as configurations for downstream single-point
labelling. The immediate driver is the internal-degrees-of-freedom (1-body)
layer of a machine-learned-force-field (MLFF) training set.

This is the SEAMM *implementation* of decision #2 of the water |rarr| electrolyte
MLFF training-set design. That campaign's **science** home is the ``~/Sites``
lab notebook -- the design document
``mlff-training/2026-07-15_water-mlff-training-plan/`` -- **not** this
repository. Keep science decisions there and implementation notes here, with
light cross-links (see *scope* for the back-pointers).

.. |rarr| unicode:: U+2192

Contents:

.. toctree::
   :glob:
   :maxdepth: 2

   *scope*
   NOTES_*
