# -*- coding: utf-8 -*-

"""Smoke test of the Tk dialog: create it and re-lay it out for every choice
that drives the layout. Skipped when no display is available."""

import pytest


@pytest.fixture()
def tk_node():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    import Pmw
    import seamm

    Pmw.initialise(root)
    flowchart = seamm.Flowchart(namespace="org.molssi.seamm", directory=".")
    tk_flowchart = seamm.TkFlowchart(
        master=root, flowchart=flowchart, namespace="org.molssi.seamm.tk"
    )
    node = flowchart.create_node("Normal Mode Sampling")
    flowchart.add_node(node)
    plugin = tk_flowchart.plugin_manager.get("Normal Mode Sampling")
    tk_node = plugin.create_tk_node(
        tk_flowchart=tk_flowchart, node=node, canvas=tk_flowchart.canvas, x=100, y=100
    )
    yield tk_node
    root.destroy()


def shown(tk_node, key):
    return tk_node[key].grid_info() != {}


def test_dialog_layouts(tk_node):
    tk_node.create_dialog()
    tk_node.reset_dialog()

    # Wigner: temperature and ceiling, no target energies
    tk_node["distribution"].set("Wigner (quantum)")
    tk_node.reset_dialog()
    assert shown(tk_node, "temperature") and shown(tk_node, "energy ceiling")
    assert not shown(tk_node, "minimum target energy")

    # ground state: no temperature
    tk_node["distribution"].set("ground state (0 K)")
    tk_node.reset_dialog()
    assert not shown(tk_node, "temperature")

    # energy-targeted: the target energies replace the temperature and ceiling
    tk_node["distribution"].set("energy-targeted (equipartition)")
    tk_node.reset_dialog()
    assert shown(tk_node, "minimum target energy")
    assert shown(tk_node, "maximum target energy")
    assert not shown(tk_node, "temperature")
    assert not shown(tk_node, "energy ceiling")

    # the per-mode count appears only for 'in proportion to the modes'
    assert not shown(tk_node, "samples per mode")
    tk_node["number of samples"].set("in proportion to the modes")
    tk_node.reset_dialog()
    for key in (
        "samples per mode",
        "minimum number of samples",
        "maximum number of samples",
    ):
        assert shown(tk_node, key)
    tk_node["number of samples"].set("20")
    tk_node.reset_dialog()
    assert not shown(tk_node, "samples per mode")

    # 'Draws per sample' appears only when a limit is set
    assert not shown(tk_node, "maximum draws")
    for key in (
        "maximum atom displacement",
        "maximum bond change",
        "maximum angle change",
    ):
        tk_node[key].set("0.3")
        tk_node.reset_dialog()
        assert shown(tk_node, "maximum draws"), key
        tk_node[key].set("none")
        tk_node.reset_dialog()
        assert not shown(tk_node, "maximum draws"), key

    for how in ("name is", "all", "current"):
        tk_node["structure configurations"].set(how)
        tk_node.reset_dialog()
