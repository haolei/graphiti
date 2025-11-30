"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

from .connection import AsyncNebulaPool, AsyncSession
from .graph_operations import NebulaGraphOperations
from .milvus_adapter import MilvusAdapter
from .query_builder import QueryBuilder
from .schema_manager import SchemaManager
from .search import NebulaSearch

logger = logging.getLogger(__name__)


class NebulaDriverSession(GraphDriverSession):
    def __init__(self, pool: AsyncNebulaPool, username: str, password: str, space: str):
        self.provider = GraphProvider.NEBULA
        self._session = AsyncSession(pool, username, password, space)

    async def __aenter__(self):
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._session.__aexit__(exc_type, exc, tb)

    async def run(self, query: str, **kwargs: Any) -> Any:
        formatted_query = query
        if kwargs:
            for k, v in kwargs.items():
                val_str = QueryBuilder._format_value(v)
                formatted_query = formatted_query.replace(f'${k}', val_str)

        resp = await self._session.execute(formatted_query)
        if not resp.is_succeeded():
            raise RuntimeError(f'Query failed: {resp.error_msg()}')

        keys = []
        if resp.col_names:
            keys = [c.decode('utf-8') for c in resp.col_names]

        records = []
        if resp.rows:
            for row in resp.rows:
                record = {}
                for i, val in enumerate(row.values):
                    key = keys[i] if i < len(keys) else str(i)
                    record[key] = self._convert_nebula_value(val)
                records.append(record)

        return records, None, keys

    def _convert_nebula_value(self, val):
        # val is nebula3.common.ttypes.Value
        # Use get_* methods instead of direct attribute access
        if val.getType() == val.NVAL:  # Null
            return None
        if val.getType() == val.BVAL:
            return val.get_bVal()
        if val.getType() == val.IVAL:
            return val.get_iVal()
        if val.getType() == val.FVAL:
            return val.get_fVal()
        if val.getType() == val.SVAL:
            return val.get_sVal().decode('utf-8')
        if val.getType() == val.DVAL:  # Date
            return str(val.get_dVal())
        if val.getType() == val.TVAL:  # Time
            return str(val.get_tVal())
        if val.getType() == val.DTVAL:  # DateTime
            return str(val.get_dtVal())
        if val.getType() == val.VVAL:  # Vertex
            vVal = val.get_vVal()
            node = {'uuid': vVal.vid.get_sVal().decode('utf-8')}
            if vVal.tags:
                for tag in vVal.tags:
                    if tag.props:
                        for k, v in tag.props.items():
                            node[k.decode('utf-8')] = self._convert_nebula_value(v)
            return node
        if val.getType() == val.EVAL:  # Edge
            eVal = val.get_eVal()
            edge = {
                'source_node_uuid': eVal.src.get_sVal().decode('utf-8'),
                'target_node_uuid': eVal.dst.get_sVal().decode('utf-8'),
                'ranking': eVal.ranking,
            }
            if eVal.props:
                for k, v in eVal.props.items():
                    edge[k.decode('utf-8')] = self._convert_nebula_value(v)
            return edge

        # Handle List (lVal)
        if val.getType() == val.LVAL:
            return [self._convert_nebula_value(v) for v in val.get_lVal().values]

        return str(val)

    async def close(self):
        pass

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)


class NebulaDriver(GraphDriver):
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        space: str,
        milvus_uri: str,
        milvus_token: str = '',
    ):
        self.provider = GraphProvider.NEBULA
        self._database = space
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._space = space

        self.pool = AsyncNebulaPool(host, port)
        self.milvus = MilvusAdapter(milvus_uri, milvus_token)
        self.schema_manager = SchemaManager(self.pool, username, password, space)
        self.query_builder = QueryBuilder()

        self.graph_operations_interface = NebulaGraphOperations()
        self.search_interface = NebulaSearch()

        # Space-isolated collection names for Milvus
        self._nodes_collection = f'{space}_Graphiti_Nodes'
        self._edges_collection = f'{space}_Graphiti_Edges'
        self._episodes_collection = f'{space}_Graphiti_Episodes'

    @property
    def nodes_collection(self) -> str:
        """Get the space-isolated nodes collection name."""
        return self._nodes_collection

    @property
    def edges_collection(self) -> str:
        """Get the space-isolated edges collection name."""
        return self._edges_collection

    @property
    def episodes_collection(self) -> str:
        """Get the space-isolated episodes collection name."""
        return self._episodes_collection

    async def connect(self):
        await self.pool.initialize()
        await self.milvus.connect()
        await self.milvus.ensure_collection(self._nodes_collection)
        await self.milvus.ensure_collection(self._edges_collection)
        await self.milvus.ensure_collection(self._episodes_collection)
        await self.schema_manager.initialize()

    async def execute_query(self, query: str, **kwargs: Any) -> Any:
        async with self.session() as session:
            return await session.run(query, **kwargs)

    def session(self, database: str | None = None) -> NebulaDriverSession:
        space = database if database else self._space
        return NebulaDriverSession(self.pool, self._username, self._password, space)

    async def close(self):
        await self.pool.close()
        await self.milvus.close()

    async def delete_all_indexes(self):
        if self.milvus.client:
            # We don't want to crash if collection doesn't exist
            try:
                await self.milvus.client.drop_collection(self._nodes_collection)
            except Exception:
                pass
            try:
                await self.milvus.client.drop_collection(self._edges_collection)
            except Exception:
                pass
            try:
                await self.milvus.client.drop_collection(self._episodes_collection)
            except Exception:
                pass

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.delete_all_indexes()
        await self.milvus.ensure_collection(self._nodes_collection)
        await self.milvus.ensure_collection(self._edges_collection)
        await self.milvus.ensure_collection(self._episodes_collection)
