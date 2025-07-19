import warnings
import requests
import os   

import pytest
import ccmlib.cluster as ccmlib_cluster
from pytriton.client import ModelClient

TEST_MODEL_NAME = "summarization_model"
TRITON_SERVER_HEALTHY_ADDRESS = os.getenv("TRITON_SERVER_HEALTHY_ADDRESS",
                                          "http://localhost:8005/v2/health/")


class TritonServerConnectionWarning(Warning):
    """Warning raised when Triton server connection fails"""
    pass


@pytest.fixture(scope="session")
def get_test_client():
    test_client = ModelClient(
                    url=os.getenv("TRITON_SERVER_URL", "grpc://127.0.0.1:8001"),
                    model_name=TEST_MODEL_NAME,
                    lazy_init=True,
                    ensure_model_is_ready=False,
                    inference_timeout_s=180
                )
    yield test_client


@pytest.fixture(scope="session")
def triton_server_status():
    try:
        response_live = requests.get(TRITON_SERVER_HEALTHY_ADDRESS + "live",
                                     timeout=2.0)

        response_ready = requests.get(TRITON_SERVER_HEALTHY_ADDRESS + "ready",
                                      timeout=2.0)
        return (response_live.status_code == 200
                and response_ready.status_code == 200)
    except requests.exceptions.RequestException:
        warnings.warn("Triton server is not reachable or not healthy",
                      TritonServerConnectionWarning)
        return False


@pytest.fixture(scope="session")
def cassandra_cluster(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("ccm")
    ccm_cluster = ccmlib_cluster.Cluster(path=str(workdir),
                                         name="test",
                                         cassandra_dir="/usr/sbin/cassandra")
    ccm_cluster.populate(1).start(wait_for_binary_proto=True)
    yield ccm_cluster
    ccm_cluster.stop()
    ccm_cluster.remove()
