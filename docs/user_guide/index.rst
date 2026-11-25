.. _user-guide:

**********
User Guide
**********

..
    <remove the dots above and this line and unindent the toctree to expose it>
    Contents:

    .. toctree::
       :glob:
       :maxdepth: 2
       :titlesonly:

       *

How the Hessian is computed
===========================

The normal modes come from the Hessian of the structure with the model chemistry chosen
by a *Model Chemistry* step earlier in the flowchart. The step picks the best route the
model chemistry offers:

- **On a cluster queue, finite differences there.** When the job's calculations go to
  a cluster queue and the program can run separate calculations (ORCA, MOPAC), the
  Hessian is the finite difference of the gradients from the 6N structures displaced by
  ±0.01 bohr along each Cartesian coordinate, run as separate calculations batched on
  the cluster. Nothing is started on the job's own machine,
  where the program may not be installed.
- **Otherwise analytic**, when the program's MDI engine provides second derivatives
  (ORCA for HF, most DFT functionals and MP2).
- **Otherwise finite differences here.** For programs that run separate calculations
  (ORCA) the 6N calculations run concurrently on this machine; for programs with a
  resident engine (MOPAC, xTB, MLFFs) they are evaluated one after another over the
  warm MDI connection. If the program is not installed on this machine at all, the
  separate calculations are used, and the output says so.

Separate calculations are kept in ``tasks/`` in the step's directory, and a rerun of the
job in the same directory reuses the finished ones.

The routes are those of the Thermochemistry and Vibrational Analysis steps, which share
the same code. The MDI engine runs with the program's own environment (for example the
modules it is installed with, or ORCA's MPI runtime) and on the cores the job has.

For HF/def2-SVP water the analytic and finite-difference Hessians agree to better than 1 cm⁻¹ in the frequencies.

Energy-targeted sampling
========================

Thermal and Wigner sampling put each sample's energy where the temperature (or the
zero-point motion) puts it, and that grows with the size of the molecule: a Wigner
sample's mean harmonic energy is about half the zero-point energy, some 14 kcal/mol
for methane and 35 kcal/mol for a 19-atom organic molecule. For training data in a
chosen energy window, use the **energy-targeted (equipartition)** distribution:

1. The shape of each sample is drawn from the classical distribution, which gives every
   mode the same expected energy (the normal-coordinate width is proportional to
   1/ω), capped at the amplitude cap.
2. A target energy E* is drawn uniformly between the **minimum** and **maximum target
   energy**.
3. The displacement is scaled so that its harmonic energy, ½ Σ ω² Q², is exactly E*.

No temperature enters. The energy ceiling does not apply, since every sample's energy is
set.

**Skip modes below** leaves out the soft modes, whose large amplitudes are better
covered by a torsion scan (100 cm⁻¹ is a common choice).

**Limits on the geometry.** A sample in which an atom moves farther than the largest
atom displacement, or a bond length or a bond angle changes by more than its limit, is
drawn again, keeping its target energy so that the limits do not bias the energies.
After the given number of draws it is dropped, and the output reports how many were.
The bonds are the structure's own, or perceived from the distances when it has none.

**Number of samples** can be a number or *in proportion to the modes*: the number of
sampled modes times the samples per mode, rounded, between a minimum and a maximum.

The random seed is reported and saved with the results, so an ensemble drawn with
'random' can be reproduced.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
