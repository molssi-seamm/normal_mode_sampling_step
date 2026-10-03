=======
History
=======
2026.10.3.1 -- The finite-difference Hessian as separate calculations, on this machine or a cluster
    * When the job's calculations go to a cluster queue and the program can run
      separate calculations (ORCA, MOPAC), the Hessian is the finite difference of the
      gradients, run there as 6N calculations; nothing is started on the job's own
      machine, where the program may not be installed.
    * Otherwise the analytic Hessian is used when the program's MDI engine has one, and
      ORCA's model chemistry now says so without ORCA being started to ask. Failing
      that, ORCA's finite-difference calculations run several at a time on this
      machine; MOPAC, xTB and MLFFs use the warm MDI engine as before.
    * Rerunning the job reuses finished finite-difference calculations.
    * If the program is not installed on this machine, the separate calculations are
      used and the output says so; other failures to start the engine stop the step
      with the reason rather than quietly switching method.
    * Requires seamm-exec 2026.10.3 or later.

2026.10.3 -- Bugfix: the Hessian used the program's method name and default basis
    * The MDI engine for the Hessian was launched with the model chemistry's
      method name alone, so ORCA ran def2-SVP whatever basis was chosen, and a
      functional whose name the Model Chemistry step had to alter was not
      recognized. It now gets the program's own keyword and the chosen basis
      (with model_chemistry_step 2026.10.3).
    * The shared CI now runs on uv: ``devtools/conda-envs/test_env.yaml`` is
      removed, so ``requirements.txt`` is the one dependency list.

2026.7.15 -- Initial release
    * Generates an ensemble of displaced structures by normal-mode sampling of a
      molecule's Hessian, for building machine-learned-force-field training sets
      and similar uses.
    * Amplitudes follow the quantum (Wigner) distribution by default, so stiff
      modes such as O-H stretches get the real zero-point spread that classical
      300 K sampling misses; classical-thermal and ground-state (0 K)
      distributions are also available.
    * The Hessian is obtained from the Model Chemistry defined earlier in the
      flowchart, over MDI: the analytic Hessian when the engine provides one,
      otherwise a finite-difference of the forces over the resident engine.
    * A temperature, per-mode selection, amplitude cap, and harmonic-energy
      outlier rejection are all controllable, and the random seed can be fixed
      for reproducible ensembles.
    * Off-minimum geometries are allowed with a warning, so a transition state is
      sampled along its real modes while its imaginary reaction coordinate is
      left alone.
    * Each generated structure records its predicted harmonic energy and the
      per-mode normal-coordinate displacement as properties, for downstream
      filtering and analysis.
