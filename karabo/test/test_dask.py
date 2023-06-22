import os
from unittest.mock import patch

import pytest

from karabo.util.dask import (  # replace `your_module` with your actual module name
    get_base_string_node_list,
    get_lowest_node_id,
    get_lowest_node_name,
    get_min_max_of_node_id,
    get_node_id,
    get_node_name,
    get_number_of_nodes,
    is_first_node,
    is_on_slurm_cluster,
)


@pytest.fixture
def env_vars():
    return {
        "SLURM_JOB_NODELIST": "nid0[4397-4406]",
        "SLURM_JOB_NUM_NODES": "10",
        "SLURMD_NODENAME": "nid04397",
        "SLURM_JOB_ID": "123456",
    }


def test_get_min_max_of_node_id(env_vars):
    with patch.dict(os.environ, env_vars):
        assert get_min_max_of_node_id() == (4397, 4406)


def test_get_lowest_node_id(env_vars):
    with patch.dict(os.environ, env_vars):
        assert get_lowest_node_id() == 4397


def test_get_base_string_node_list(env_vars):
    with patch.dict(os.environ, env_vars):
        assert get_base_string_node_list() == "nid0"


def test_get_lowest_node_name(env_vars):
    with patch.dict(os.environ, env_vars):
        assert get_lowest_node_name() == "nid04397"


def test_get_number_of_nodes(env_vars):
    with patch.dict(os.environ, env_vars):
        assert get_number_of_nodes() == 10


def test_get_node_id(env_vars):
    with patch.dict(os.environ, env_vars):
        assert get_node_id() == 4397


def test_get_node_name(env_vars):
    with patch.dict(os.environ, env_vars):
        assert get_node_name() == "nid04397"


def test_is_first_node(env_vars):
    with patch.dict(os.environ, env_vars):
        assert is_first_node() is True


def test_is_on_slurm_cluster(env_vars):
    with patch.dict(os.environ, env_vars):
        assert is_on_slurm_cluster() is True


# repeat the tests for other values of environment variables

def test_multiple_nodes_and_ranges():
    env_vars = {
        "SLURM_JOB_NODELIST": "nid0[2780-2781,4715]",
        "SLURM_JOB_NUM_NODES": "3",
        "SLURMD_NODENAME": "nid02780",
        "SLURM_JOB_ID": "123456"
    }
    with patch.dict(os.environ, env_vars):
        assert get_min_max_of_node_id() == (2780, 4715)
        assert get_lowest_node_id() == 2780
        assert get_base_string_node_list() == "nid0"
        assert get_lowest_node_name() == "nid02780"
        assert get_number_of_nodes() == 3
        assert get_node_id() == 2780
        assert get_node_name() == "nid02780"
        assert is_first_node() is True
        assert is_on_slurm_cluster() is True


def test_single_node():
    env_vars = {
        "SLURM_JOB_NODELIST": "nid03038",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURMD_NODENAME": "nid03038"
    }
    with patch.dict(os.environ, env_vars):
        min_node_id, max_node_id = get_min_max_of_node_id()
        assert min_node_id == 3038
        assert max_node_id == 3038
        