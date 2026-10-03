# -*- coding: utf-8 -*-

"""The MDI engine is launched with the engine's keyword and the user's basis."""

import types

import numpy as np

import normal_mode_sampling_step.normal_mode_sampling as nms


class _Engine:
    """Stand-in for seamm_mdi.MDIEngine: records the argv, offers a Hessian."""

    argv = None

    def __init__(self, build_argv, elements, **kwargs):
        _Engine.argv = build_argv("localhost", 1234)
        self.n = len(elements)

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
    monkeypatch.setattr(nms, "MDIEngine", _Engine)
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
