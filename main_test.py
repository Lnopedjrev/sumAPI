import importlib.util as iutil
from typing import List
from functools import partial

import pytest
import numpy as np
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from microServiceAPI.utils.fixtures import get_test_client, triton_server_status, cassandra_cluster
from .main import app as fastapi_app
from .db.dbmanager import get_cassandra_manager, get_test_cassandra_manager
from .models import get_test_custom_article


fastapi_client = TestClient(fastapi_app)


@pytest.mark.parametrize(("input_texts", "expected_output_format"), [
    (["Summarize: Test Article 1", ], np.ndarray),  # single input
    (["Summarize: Test Article 1",
      "Summarize: Test Article 2"], np.ndarray),  # multiple inputs
])
def test_pytorch_server_inference(get_test_client, triton_server_status, input_texts, expected_output_format):
    """Test pytorch server inference endpoint with both single and batched input"""
    if not triton_server_status:
        pytest.skip("Triton server is not running or not healthy")
    client = get_test_client
    preprocessed_texts = np.char.encode(input_texts, 'utf-8').reshape(-1, 1)
    summaries = client.infer_batch(preprocessed_texts)
    print("Summaries:", summaries)
    assert isinstance(summaries['output_text'], expected_output_format)


@pytest.mark.parametrize(("input_data", "expected_output_format"), [
    ([get_test_custom_article(title="Test Article 1"), ], List[str]),  # single input
    ([get_test_custom_article(title="Test Article 1"),
      get_test_custom_article(title="Test Article 2")], List[str]),  # multiple inputs
])
@pytest.mark.usefixtures("cassandra_cluster")
@pytest.mark.anyio
async def test_fastapi_summarization_endpoint(triton_server_status, input_data, expected_output_format):
    """Test fastapi endpoint with both single and batched input"""
    if not triton_server_status:
        pytest.skip("Triton server is not running or not healthy")

    fastapi_app.dependency_overrides[get_cassandra_manager] = get_test_cassandra_manager
    serialized_data = [article.json() for article in input_data]

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as fastapi_client:
        response = await fastapi_client.post("/summarize",
                                             json=serialized_data)
        response_data = response.json()

    assert response.status_code == 200
    assert isinstance(response_data, expected_output_format)
    assert len(response_data) == len(input_data)

    manager = await get_test_cassandra_manager()
    db_summaries = await manager.get_all_summaries().all()

    assert len(db_summaries) == len(input_data)
    for i, record in enumerate(db_summaries):
        resource_text = f"Summarize: {input_data[i].title} \n {input_data[i].content}"
        assert record.original_text == resource_text
        assert record.summary != ""

    fastapi_app.dependency_overrides[get_cassandra_manager] = {}
