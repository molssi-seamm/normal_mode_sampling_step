# -*- coding: utf-8 -*-

"""The MDI engine is launched with the engine's keyword and the user's basis."""

import types

import numpy as np
import pytest

import normal_mode_sampling_step.normal_mode_sampling as nms


class _Engine:
    """Stand-in for seamm_mdi.MDIEngine: records the argv, offers a Hessian."""

    argv = None

    def __init__(self, build_argv, elements, **kwargs):
        _Engine.argv = build_argv("localhost", 1234)
        self.n = len(elements)

    def start(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def set_coordinates(self, xyz, units=None):
        pass

    def supports(self, command):
        return True

    def hessian(self):
        return np.eye(3 * self.n)


class _OrcaStep:
    def __init__(self):
        self.kwargs = None

    def get_mdi_engine_command(self, executor, seamm_options, **kwargs):
        self.kwargs = kwargs
        return ["orca_mdi"]


def test_hessian_engine_gets_the_engine_keyword_and_users_basis(monkeypatch):
    import seamm_mdi

    monkeypatch.setattr(seamm_mdi, "MDIEngine", _Engine)
    step = _OrcaStep()
    mc = {
        "level": "ORCA:DFT@wB97X-D3/def2-TZVP",
        "method": "wB97X-D3",
        "basis": "def2-TZVP",
        "step": "orca-step",
        "options": {
            "mdi_capable": True,
            "mdi_method_arg": "WB97X-D3",
            "mdi_basis_arg": "def2-TZVP",
        },
    }
    me = types.SimpleNamespace(
        variable_exists=lambda name: True,
        get_variable=lambda name: mc,
        global_options={},
        logger=None,
        flowchart=types.SimpleNamespace(
            executor="local",
            plugin_manager=types.SimpleNamespace(get=lambda name: step),
        ),
    )
    configuration = types.SimpleNamespace(
        periodicity=0,
        charge=0,
        spin_multiplicity=1,
        n_atoms=1,
        atoms=types.SimpleNamespace(
            atomic_numbers=[1],
            get_coordinates=lambda fractionals, as_array: [[0.0, 0.0, 0.0]],
        ),
    )
    nms.NormalModeSampling._hessian(me, configuration)
    assert step.kwargs["method"] == "WB97X-D3"
    assert step.kwargs["basis"] == "def2-TZVP"

    # MOPAC: no basis argument at all
    mc["options"] = {"mdi_capable": True, "mdi_method_arg": "PM6"}
    nms.NormalModeSampling._hessian(me, configuration)
    assert step.kwargs["method"] == "PM6" and "basis" not in step.kwargs


def test_fd_hessian_tasks_reproduces_a_quadratic_potential():
    """The task-based finite difference assembles the Hessian correctly: for
    E = 1/2 x.K.x (x in bohr) the gradients are K.x, so the FD Hessian is K."""
    from seamm_exec import EvaluatorResult
    from seamm_util import Q_

    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    K = A @ A.T  # hartree/bohr^2

    class FakeEvaluator:
        def __init__(self):
            self.submitted = {}

        def submit(self, geometry, key):
            self.submitted[key] = geometry
            return key

        def results(self):
            for key, geometry in self.submitted.items():
                x = (
                    Q_(geometry.atoms.get_coordinates(as_array=True), "Å")
                    .m_as("bohr")
                    .reshape(-1)
                )
                g = Q_(K @ x, "hartree/bohr").m_as("kJ/mol/Å").reshape(-1, 3)
                yield EvaluatorResult(key=key, ok=True, energy=0.0, gradients=g)

    configuration = types.SimpleNamespace(
        id=7,
        charge=0,
        spin_multiplicity=1,
        atoms=types.SimpleNamespace(atomic_numbers=[1, 1]),
    )
    me = types.SimpleNamespace()
    evaluator = FakeEvaluator()
    H = nms.NormalModeSampling._fd_hessian_tasks(
        me, evaluator, configuration, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    )
    assert len(evaluator.submitted) == 12  # 6N displacements, N = 2
    assert np.allclose(H, K, atol=1e-8)


def _me_and_configuration(mc, step, target=None):
    me = types.SimpleNamespace(
        variable_exists=lambda name: True,
        get_variable=lambda name: mc,
        global_options={},
        indent="",
        logger=types.SimpleNamespace(warning=lambda *a, **k: None),
        flowchart=types.SimpleNamespace(
            executor="local",
            plugin_manager=types.SimpleNamespace(get=lambda name: step),
        ),
    )
    me._code_is_here = lambda step: nms.NormalModeSampling._code_is_here(me, step)
    called = {}

    def fd_tasks(evaluator, configuration, coords):
        called["fd tasks"] = evaluator.target
        return np.eye(3)

    me._fd_hessian_tasks = fd_tasks
    configuration = types.SimpleNamespace(
        periodicity=0,
        charge=0,
        spin_multiplicity=1,
        n_atoms=1,
        atoms=types.SimpleNamespace(
            atomic_numbers=[1],
            get_coordinates=lambda fractionals, as_array: [[0.0, 0.0, 0.0]],
        ),
    )
    return me, configuration, called


class _TaskProvider(_OrcaStep):
    get_task = staticmethod(lambda *a, **k: None)
    analyze_task = staticmethod(lambda *a, **k: None)


_MC = {
    "level": "ORCA:DFT@B3LYP/def2-SVP",
    "method": "B3LYP",
    "basis": "def2-SVP",
    "step": "orca-step",
    "options": {"mdi_capable": True, "mdi_method_arg": "B3LYP"},
}


def test_no_code_here_means_finite_differences_as_tasks(monkeypatch):
    """The code is installed only where the job's tasks run: no engine is
    started; the Hessian is the finite difference as tasks."""
    import seamm_mdi

    class _Exploding:
        def __init__(self, *a, **k):
            raise AssertionError("no engine may be started")

    monkeypatch.setattr(seamm_mdi, "MDIEngine", _Exploding)

    class _NotHere(_TaskProvider):
        @staticmethod
        def get_executor_config(executor, options):
            raise RuntimeError("Could not find the 'orca' executable")

    printed = []
    monkeypatch.setattr(
        nms.printer, "important", lambda text: printed.append(str(text))
    )
    me, configuration, called = _me_and_configuration(_MC, _NotHere())
    H = nms.NormalModeSampling._hessian(me, configuration)
    assert "fd tasks" in called and H.shape == (3, 3)
    assert "not installed on this machine" in printed[0]


def test_a_queue_target_wins(monkeypatch, tmp_path):
    """On a queue target the finite difference runs as tasks on the cluster;
    no local engine is started even to ask for an analytic Hessian."""
    import seamm_mdi
    import seamm_exec.targets
    from seamm_scheduler import TargetSection

    class _Exploding:
        def __init__(self, *a, **k):
            raise AssertionError("no engine may be started")

    monkeypatch.setattr(seamm_mdi, "MDIEngine", _Exploding)
    section = TargetSection(
        name="arc", transport="ssh", host="tc", type="local", tasks="queue"
    )
    monkeypatch.setattr(seamm_exec.targets, "find_target", lambda *a, **k: section)
    me, configuration, called = _me_and_configuration(_MC, _TaskProvider())
    nms.NormalModeSampling._hessian(me, configuration)
    assert called["fd tasks"] is section


def test_a_transient_engine_failure_is_not_hidden(monkeypatch):
    """With the code here, an engine that fails to start is an error, not a
    silent switch from the analytic Hessian to 6N finite-difference tasks."""
    import seamm_mdi

    class _Flaky:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            raise RuntimeError("port in use")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(seamm_mdi, "MDIEngine", _Flaky)

    class _Here(_TaskProvider):
        @staticmethod
        def get_executor_config(executor, options):
            return {"code": "/opt/orca/orca"}

    me, configuration, called = _me_and_configuration(_MC, _Here())
    with pytest.raises(RuntimeError, match="port in use"):
        nms.NormalModeSampling._hessian(me, configuration)
    assert called == {}
