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

For HF/def2-SVP water the analytic and finite-difference Hessians agree to better than 1 cm⁻¹ in the frequencies.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
