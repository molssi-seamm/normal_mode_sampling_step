Normal Mode Sampling -- scope and design
========================================

Goal
----

A new **Normal Mode Sampling** plug-in (``normal_mode_sampling_step``) that,
given a molecule and its Hessian, draws an ensemble of *N* displaced geometries
by **Wigner (quantum) normal-mode sampling** and writes them as configurations
in a new system, ready for a downstream loop of single-point calculations.

The step covers the **intramolecular (1-body)** layer of a machine-learned
force-field (MLFF) training set: it broadens each optimized monomer along its
own vibrational modes so the fit sees the internal distortions a production
simulation will actually visit. It is the normal-mode sibling of
``dimer_builder_step`` (the 2-body layer) and feeds the same counterpoise-
corrected labelling pipeline (see ``orca_step`` BSSE / the general ``bsse_step``).

Where this fits (cross-project)
-------------------------------

The science lives in the ``~/Sites`` lab notebook, **not** in this repository:

* **Design / roadmap:**
  ``~/Sites/mlff-training/2026-07-15_water-mlff-training-plan/index.html`` --
  the water |rarr| electrolyte (Li\ :sup:`+`/BF\ :sub:`4`\ :sup:`-`) MLFF
  training-set design. This plug-in implements its **decision #2, "Internal
  sampling = Wigner normal-mode sampling."**

Keep the layers separate: **science/campaign decisions in** ``~/Sites``;
**implementation and dev notes here.** When this step satisfies a campaign
requirement, record the *implementation* here and leave a back-pointer in the
``~/Sites`` plan.

The science: why Wigner, not classical
---------------------------------------

At an optimized geometry the potential is, to second order, a set of independent
harmonic oscillators (the normal modes). We sample each mode :math:`i` as a
Gaussian in its normal coordinate with variance

.. math::

   \sigma_i^2 = \frac{\hbar}{2\omega_i}\,
                \coth\!\left(\frac{\hbar\omega_i}{2 k_B T}\right)

-- the **quantum (Wigner) distribution** of a harmonic oscillator, *not* the
classical Boltzmann width :math:`\sigma_i^2 = k_B T / \omega_i^2`.

This distinction is the whole point. The O--H stretch is very stiff
(:math:`\hbar\omega/k_B \approx 5300` K), so at 300 K classical sampling barely
moves it -- yet zero-point motion gives it a real, wide spread. The
:math:`\coth` factor captures that width (as :math:`T\to 0` it reduces to the
pure zero-point amplitude :math:`\sigma_i^2 = \hbar/2\omega_i`), while soft
bends and librations broaden thermally as expected. Getting the stiff stretch
right is exactly what classical MD of a force field would miss, and it is the
single most important internal coordinate for this training set. The same
treatment covers BF\ :sub:`4`\ :sup:`-`'s stiff B--F modes when the campaign
reaches the ions.

How it works
------------

**Input.** A configuration (selected with the same selectors
``dimer_builder_step`` uses -- current / first / last / by name / glob / regex),
its charge and multiplicity taken from the configuration, and a **Model
Chemistry** that defines the QM (or ML) method used to obtain the Hessian and,
implicitly, the surface the samples sit on.

**Reference geometry -- warn but allow.** The harmonic picture assumes a
stationary point. The step does **not** require a minimum: it reports any
imaginary or near-zero modes and *warns* if the geometry does not look like a
clean minimum, but proceeds. This is deliberate -- sampling around a **transition
state** (one imaginary mode) is a legitimate and useful case. Imaginary and
near-zero modes are **not** Wigner-displaced (the :math:`\coth` amplitude is
undefined for imaginary :math:`\omega`); by default they are skipped, so a TS is
sampled along its real modes with the reaction coordinate left alone.

**Hessian over MDI.** Consistent with the workspace's shift to Model Chemistry +
MDI (a subflowchart is overkill for so simple a setup, and a resident MDI engine
removes per-call startup cost), the step gets the Hessian from the model
chemistry's MDI engine. MDI carries only energy/forces/stress today, so the
driver discovers capability at **run time**:

* if the engine supports a (new, custom) ``<HESSIAN`` command, pull the
  **analytic** Hessian directly -- far better where available;
* otherwise **finite-difference the gradients** (``<FORCES``) over the *warm*
  connection -- code-agnostic, and cheap precisely because the engine stays
  resident.

This is seamless to the user: no capability flag on the Model Chemistry, no GUI
branch. (Adding ORCA's analytic second derivative -- ``! AnFreq`` + ``.hess`` --
and wiring it to ``<HESSIAN`` is a separate, parallel task in ``orca_step``.)

**Modes.** Mass-weight the Hessian, diagonalize, convert eigenvalues to
frequencies, and project out the 6 (5 for a linear molecule) translation/rotation
modes. Retain the real vibrational modes for sampling; report imaginary/near-zero
ones.

**Sampling.** For each of *N* samples, draw a displacement in each retained
normal coordinate :math:`q_i \sim \mathcal{N}(0, \sigma_i^2)` (optionally the
matching momenta, unused here), transform back to mass-weighted Cartesians, and
add to the reference geometry. Controls:

* **distribution** -- Wigner (quantum, default) / classical-thermal /
  ground-state (:math:`T\to 0`, pure ZPE);
* **temperature** *T*;
* **amplitude cap** -- clamp \|q\ :sub:`i`\ \| to a multiple of :math:`\sigma_i`
  (no wandering toward dissociation);
* **energy-outlier rejection** -- reject a draw whose harmonic energy estimate
  exceeds a ceiling, and redraw;
* **mode selection** -- include/exclude specific modes (e.g. sample only the
  stiff stretches);
* **random seed** -- for reproducible ensembles.

**Output.** *N* configurations in a **new system** (the ``dimer_builder_step``
idiom: read ``_system_db``, create the system, add configurations, make it
current). A downstream **Loop** iterates the configurations into single-point
QM/CP labelling. **No file dump** here -- writing SDF / XYZ / EXTXYZ is the job
of a following Write Structure step.

Parameters (draft)
------------------

``model chemistry``, ``configuration`` selector, ``number of samples``,
``distribution`` (Wigner / classical / ground-state), ``temperature``,
``amplitude cap`` (in units of :math:`\sigma`), ``energy ceiling`` for outlier
rejection, ``modes to sample`` (all / list / exclude), ``random seed``, and the
``system name`` for the output.

GUI (prevent invalid combinations)
----------------------------------

Following the workspace rule -- make invalid states unpickable, don't just catch
them at run time:

* hide **temperature** when the distribution is *ground-state* (:math:`T=0`);
* expose the **Model Chemistry** picker; caps and ceilings as plain fields.

Run-time checks in the headless node remain the backstop for hand-edited or
scripted flowcharts.

Completeness (not just the count)
---------------------------------

Ship the plug-in complete, per the workspace standard:

* **Citations** -- Wigner (1932) for the quantum phase-space distribution; the
  normal-mode-sampling methodology; and whatever method/basis/code the chosen
  Model Chemistry resolves to (harvested the usual way, not hard-coded).
* **Results & properties** -- number of configurations generated, the
  frequencies/normal modes used, counts of imaginary/near-zero modes, and
  per-sample harmonic-energy statistics; stored through ``store_results`` so
  they are available as variables/tables.

Out of scope (v1)
-----------------

* **Elevated-temperature MD** for correlated multi-mode distortions -- a separate
  MD step; NMS is purely harmonic normal-mode.
* **File output** (SDF/XYZ/EXTXYZ) -- delegated to a downstream Write Structure
  step.
* **Many-body / cluster sampling** -- that is the dimer/cluster + bootstrapped
  active-learning machinery, not this step.
* A custom ``<HESSIAN`` MDI command and ORCA analytic second derivatives -- real
  and needed, but tracked as separate tasks; NMS runs on the finite-difference
  fallback until they land.

Open questions
--------------

* Amplitude-cap default and the energy-outlier ceiling -- pick values that keep
  geometries physical without truncating the genuine zero-point width.
* Whether the harmonic energy estimate is a good enough outlier filter, or the
  actual single-point energy should gate acceptance (couples sampling to the
  labelling loop).
* How to present transition-state sampling in the GUI without inviting misuse.

References
----------

* E. Wigner, "On the Quantum Correction For Thermodynamic Equilibrium,"
  *Phys. Rev.* **40**, 749 (1932).

.. |rarr| unicode:: U+2192
.. |larr| unicode:: U+2190
