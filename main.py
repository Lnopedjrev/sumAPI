from typing import Union, Optional, List, Annotated
import uuid
import os

from fastapi import FastAPI, Depends
import uvicorn
import numpy as np
from pytriton.client import AsyncioModelClient
from cassandra.query import PreparedStatement
from dotenv import load_dotenv

from microServiceAPI.models import CustomArticle
from microServiceAPI.db.dbmanager import CassandraManager, get_prepared_write_summaries, get_cassandra_manager

app = FastAPI()
client_config = None
DatabaseManager = Annotated[CassandraManager, Depends(get_cassandra_manager)]


@app.on_event("startup")
async def startup():
    print("Starting up the application...")
    global client_config
    client_config = AsyncioModelClient(
                    url=TRITON_SERVER_URL,
                    model_name="summarization_model",
                    lazy_init=False,
                    ensure_model_is_ready=True,
                    inference_timeout_s=180
                )
    print("Pytrtion Client is connected to server")

    # If not set, the infer_sample defaults to HTTP client, because prefix grpc:// is stripped away
    client_config._url = TRITON_SERVER_URL


@app.post("/summarize")
async def get_sum(resources: List[CustomArticle],
                  db_manager: DatabaseManager,
                  write_summaries_statement: PreparedStatement = Depends(get_prepared_write_summaries)):
    global client_config
    async with AsyncioModelClient.from_existing_client(client_config) as request_client:
        resources_texts = [f"Summarize: {resource.title} \n {resource.content}" for resource in resources]
        resources_encoded = np.char.encode(resources_texts, 'utf-8').reshape(-1, 1)
        summaries = await request_client.infer_batch(resources_encoded)
    summaries_decoded = summaries['output_text'].astype("U").flatten()
    for index, summary in enumerate(summaries_decoded):
        resource = resources[index]
        bound_stm = write_summaries_statement.bind((resource.user_id,
                                                    uuid.uuid4(),
                                                    resources_texts[index],
                                                    resource.categories,
                                                    summary,
                                                    -1))
        await db_manager.session.aexecute(bound_stm)

    return summary['output_text'].tolist()


@app.on_event("shutdown")
async def shutdown():
    await client_config.close()


if __name__ == "__main__":
    load_dotenv()
    API_PORT = os.getenv("API_PORT", 8090)
    API_HOST = os.getenv("API_HOST", '127.0.0.1')
    TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "grpc://127.0.0.1:8001")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
