=======
History
=======

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
