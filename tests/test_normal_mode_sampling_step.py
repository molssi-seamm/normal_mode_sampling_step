#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `normal_mode_sampling_step` package."""

import pytest  # noqa: F401
import normal_mode_sampling_step  # noqa: F401


def test_construction():
    """Just create an object and test its type."""
    result = normal_mode_sampling_step.NormalModeSampling()
    assert (
        str(type(result))
        == "<class 'normal_mode_sampling_step.normal_mode_sampling.NormalModeSampling'>"
    )
