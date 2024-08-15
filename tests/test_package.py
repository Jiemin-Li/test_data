from __future__ import annotations

import importlib.metadata

import test_data as m


def test_version():
    assert importlib.metadata.version("test_data") == m.__version__
