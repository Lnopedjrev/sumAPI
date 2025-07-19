import os
import uuid
from typing import List, Callable, Annotated
from functools import partial

from cassandra_asyncio.cluster import Cluster
from dotenv import load_dotenv
from fastapi import Depends

from microServiceAPI.utils.decorators import preparable


class CassandraManager:
    def __init__(self):
        self.cluster = None
        self.session = None
        self.prepared = False

    def connect(self, main_node_ip: str = "127.0.0.1", cql_port: int = 9042):
        self.cluster = Cluster([main_node_ip, ],
                               port=cql_port,
                               connect_timeout=10)
        self.session = self.cluster.connect()
        return self.session

    async def setup(self, keyspace: str = "api_keyspace"):
        if self.session is not None:
            await self.session.aexecute(f"CREATE KEYSPACE IF NOT EXISTS {keyspace}" + " WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 2}")
            self.session.set_keyspace(keyspace)

            await self.session.aexecute("""CREATE TABLE IF NOT EXISTS summaries_by_user_id(
                                            user_id int,
                                            article_id uuid,
                                            original_text text,
                                            categories list<text>,
                                            summary text,
                                            user_review int,
                                            PRIMARY KEY ((user_id), article_id)                                                                
                                        )""")
        return self.session

    def prepare(self):
        self.prepared = True
        return self

    def unprepare(self):
        self.prepared = False
        return self

    @preparable
    async def write_summaries(self, *, user_id: int, original_text: str, categories: List[str], summary: str, user_review: int = -1):
        if self.prepared and self.session is not None:
            prepared_statement = self.session.prepare(
                "INSERT INTO summaries_by_user_id (user_id, article_id, original_text, categories, summary, user_review) VALUES (?, ?, ?, ?, ?, ?)"
            )
            return prepared_statement
        elif self.session is not None:
            await self.session.aexecute(
                "INSERT INTO summaries_by_user_id (user_id, article_id, original_text, categories, summary, user_review) VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, uuid.uuid4(),
                 original_text, categories,
                 summary, user_review)
            )
            return True
        return False

    async def get_all_summaries(self):
        query_statement = "SELECT * FROM summaries_by_user_id;"
        rows = await self.session.aexecute(
            query_statement
        )
        return rows


async def get_cassandra_manager(keyspace: str = "api_keyspace"):
    load_dotenv()
    main_node_ip = os.getenv("CASSANDRA_MAIN_NODE_IP", "127.0.0.1")
    cql_port = int(os.getenv("CASSANDRA_CQL_PORT", 9042))
    manager = CassandraManager()
    manager.connect(main_node_ip, cql_port)
    await manager.setup()
    return manager


async def get_test_cassandra_manager():
    return await get_cassandra_manager("test_keyspace")


async def get_prepared_statement(method: Callable, db_manager: Annotated[CassandraManager, Depends(get_cassandra_manager)]):
    if not db_manager.prepared:
        db_manager.prepare()
    return await method(db_manager)


async def get_prepared_write_summaries(db_manager: Annotated[CassandraManager, Depends(get_cassandra_manager)]):
    return await get_prepared_statement(CassandraManager.write_summaries, db_manager)
