import os
from typing import Dict
from unittest.mock import patch

import dask
import pytest
from dask import compute  # type: ignore[attr-defined]

from karabo.util.dask import DaskHandler, DaskHandlerSlurm

_EnvVarsType = Dict[str, str]


@pytest.fixture
def env_vars() -> Dict[str, str]:
    return {
        "SLURM_JOB_NODELIST": "nid0[4397-4406]",
        "SLURM_JOB_NUM_NODES": "10",
        "SLURMD_NODENAME": "nid04397",
        "SLURM_JOB_ID": "123456",
    }


def test_get_min_max_of_node_id(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm._get_min_max_of_node_id() == (4397, 4406)


def test_get_lowest_node_id(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm._get_lowest_node_id() == 4397


def test_get_base_string_node_list(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm._get_base_string_node_list() == "nid0"


def test_get_lowest_node_name(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm._get_lowest_node_name() == "nid04397"


def test_get_number_of_nodes(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm.get_number_of_nodes() == 10


def test_get_node_id(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm.get_node_id() == 4397


def test_get_node_name(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm.get_node_name() == "nid04397"


def test_is_first_node(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm.is_first_node() is True


def test_is_on_slurm_cluster(env_vars: _EnvVarsType) -> None:
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm.is_on_slurm_cluster() is True


# repeat the tests for other values of environment variables


def test_multiple_nodes_and_ranges() -> None:
    env_vars = {
        "SLURM_JOB_NODELIST": "nid0[2780-2781,4715]",
        "SLURM_JOB_NUM_NODES": "3",
        "SLURMD_NODENAME": "nid02780",
        "SLURM_JOB_ID": "123456",
    }
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm._get_min_max_of_node_id() == (2780, 4715)
        assert DaskHandlerSlurm._get_lowest_node_id() == 2780
        assert DaskHandlerSlurm._get_base_string_node_list() == "nid0"
        assert DaskHandlerSlurm._get_lowest_node_name() == "nid02780"
        assert DaskHandlerSlurm.get_number_of_nodes() == 3
        assert DaskHandlerSlurm.get_node_id() == 2780
        assert DaskHandlerSlurm.get_node_name() == "nid02780"
        assert DaskHandlerSlurm.is_first_node() is True
        assert DaskHandlerSlurm.is_on_slurm_cluster() is True

    # test for a different node
    env_vars["SLURMD_NODENAME"] = "nid04715"

    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm._get_min_max_of_node_id() == (2780, 4715)
        assert DaskHandlerSlurm._get_lowest_node_id() == 2780
        assert DaskHandlerSlurm._get_base_string_node_list() == "nid0"
        assert DaskHandlerSlurm._get_lowest_node_name() == "nid02780"
        assert DaskHandlerSlurm.get_number_of_nodes() == 3
        assert DaskHandlerSlurm.get_node_id() == 4715
        assert DaskHandlerSlurm.get_node_name() == "nid04715"
        assert DaskHandlerSlurm.is_first_node() is False
        assert DaskHandlerSlurm.is_on_slurm_cluster() is True

    # test for a different node
    env_vars["SLURMD_NODENAME"] = "nid02781"

    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm._get_min_max_of_node_id() == (2780, 4715)
        assert DaskHandlerSlurm._get_lowest_node_id() == 2780
        assert DaskHandlerSlurm._get_base_string_node_list() == "nid0"
        assert DaskHandlerSlurm._get_lowest_node_name() == "nid02780"
        assert DaskHandlerSlurm.get_number_of_nodes() == 3
        assert DaskHandlerSlurm.get_node_id() == 2781
        assert DaskHandlerSlurm.get_node_name() == "nid02781"
        assert DaskHandlerSlurm.is_first_node() is False
        assert DaskHandlerSlurm.is_on_slurm_cluster() is True


def test_extreme_range_of_nodes() -> None:
    env_vars = {
        "SLURM_JOB_NODELIST": "nid0[2780-2781,3213-4313,4441,4443,4715]",
        "SLURM_JOB_NUM_NODES": "1106",
        "SLURMD_NODENAME": "nid03333",
        "SLURM_JOB_ID": "123456",
    }
    with patch.dict(os.environ, env_vars):
        assert DaskHandlerSlurm._get_min_max_of_node_id() == (2780, 4715)
        assert DaskHandlerSlurm._get_lowest_node_id() == 2780
        assert DaskHandlerSlurm._get_base_string_node_list() == "nid0"
        assert DaskHandlerSlurm._get_lowest_node_name() == "nid02780"
        assert DaskHandlerSlurm.get_number_of_nodes() == 1106
        assert DaskHandlerSlurm.get_node_id() == 3333
        assert DaskHandlerSlurm.get_node_name() == "nid03333"
        assert DaskHandlerSlurm.is_first_node() is False
        assert DaskHandlerSlurm.is_on_slurm_cluster() is True
        assert len(DaskHandlerSlurm._extract_node_ids_from_node_list()) == 1106


def test_single_node() -> None:
    env_vars = {
        "SLURM_JOB_NODELIST": "nid03038",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURMD_NODENAME": "nid03038",
    }
    with patch.dict(os.environ, env_vars):
        min_node_id, max_node_id = DaskHandlerSlurm._get_min_max_of_node_id()
        assert min_node_id == 3038
        assert max_node_id == 3038
        assert DaskHandlerSlurm._get_base_string_node_list() == "nid"


def test_dask_job() -> None:
    DaskHandler.setup()
    client = DaskHandler.get_dask_client()

    assert client is not None

    @dask.delayed
    def inc(x):
        return x + 1

    @dask.delayed
    def double(x):
        return x * 2

    @dask.delayed
    def add(x, y):
        return x + y

    data = [1, 2, 3, 4, 5]

    output = []
    for x in data:
        a = inc(x)
        b = double(x)
        c = add(a, b)
        output.append(c)

    result = compute(*output, scheduler="distributed")

    assert result == (4, 7, 10, 13, 16)
    assert sum(result) == 50


def simple_function(x: int, multiplier: int = 1) -> int:
    return x * multiplier


def test_parallelize_with_dask() -> None:
    iterable = [1, 2, 3, 4, 5]
    results = DaskHandler.parallelize_with_dask(simple_function, iterable, multiplier=2)
    expected_results = tuple([x * 2 for x in iterable])
    assert results == expected_results
