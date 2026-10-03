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

- **Analytic**, when the program's MDI engine provides second derivatives (ORCA for HF,
  most DFT functionals and MP2).
- **Finite differences of the gradients**, from the 6N structures displaced by ±0.01
  bohr along each Cartesian coordinate, otherwise. For programs that run separate
  calculations (ORCA, or any model chemistry when the job's tasks go to a cluster queue)
  the 6N calculations run together -- concurrently on this machine, or batched on the
  cluster -- and a rerun of the job in the same directory reuses the finished ones. For
  programs with a resident engine (MOPAC, xTB, MLFFs) they are evaluated one after
  another over the warm MDI connection.

For HF/def2-SVP water the two routes agree to better than 1 cm⁻¹ in the frequencies.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
