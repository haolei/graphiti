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

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymilvus import AsyncMilvusClient as AsyncMilvusClientType
else:
    AsyncMilvusClientType = Any

try:
    from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
except ImportError:
    AsyncMilvusClient = None  # type: ignore[misc, assignment]
    DataType = None  # type: ignore[misc, assignment]
    Function = None  # type: ignore[misc, assignment]
    FunctionType = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


class MilvusAdapter:
    def __init__(
        self, uri: str, token: str = '', collection_name: str = 'Graphiti_Nodes', dim: int = 1536
    ):
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dim = dim
        self.client: AsyncMilvusClientType | None = None
        self._ensured_collections: set[str] = set()  # Local cache for ensured collections

    async def connect(self):
        if AsyncMilvusClient is None:
            raise ImportError(
                'pymilvus is not installed. Please install it with `pip install pymilvus`.'
            )
        self.client = AsyncMilvusClient(self.uri, token=self.token)

    async def ensure_collection(self, collection_name: str = 'Graphiti_Nodes'):
        # Fast path: check local cache first
        if collection_name in self._ensured_collections:
            return

        if not self.client:
            await self.connect()

        if self.client is None or DataType is None:
            raise RuntimeError('Failed to connect to Milvus or DataType not available')

        if await self.client.has_collection(collection_name):
            self._ensured_collections.add(collection_name)
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            description=f'Graphiti Embeddings for {collection_name}',
        )
        schema.add_field('id', DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field('embedding', DataType.FLOAT_VECTOR, dim=self.dim)
        # Add text field with analyzer for BM25 full-text search
        schema.add_field('text', DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        # Add sparse vector field for BM25 embeddings
        schema.add_field('sparse', DataType.SPARSE_FLOAT_VECTOR)

        # Add BM25 function to convert text to sparse vectors
        if Function is not None and FunctionType is not None:
            bm25_function = Function(
                name='text_bm25_emb',
                input_field_names=['text'],
                output_field_names=['sparse'],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        # Dense vector index for similarity search
        index_params.add_index('embedding', index_type='AUTOINDEX', metric_type='COSINE')
        # Sparse vector index for BM25 full-text search
        index_params.add_index(
            'sparse',
            index_type='SPARSE_INVERTED_INDEX',
            metric_type='BM25',
            params={'inverted_index_algo': 'DAAT_MAXSCORE', 'bm25_k1': 1.2, 'bm25_b': 0.75},
        )

        await self.client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

        self._ensured_collections.add(collection_name)

    async def _ensure_client(self) -> None:
        """Ensure client is connected, raise if not possible."""
        if not self.client:
            await self.connect()
        if self.client is None:
            raise RuntimeError('Failed to connect to Milvus')

    async def upsert_vectors(self, data: list[dict], collection_name: str = 'Graphiti_Nodes'):
        await self._ensure_client()
        assert self.client is not None  # For type checker
        await self.client.upsert(collection_name=collection_name, data=data)

    async def search(
        self,
        vector: list[float],
        limit: int = 10,
        filter_expr: str = '',
        collection_name: str = 'Graphiti_Nodes',
    ) -> list[dict]:
        await self._ensure_client()
        assert self.client is not None  # For type checker

        res = await self.client.search(
            collection_name=collection_name,
            data=[vector],
            limit=limit,
            filter=filter_expr,
            output_fields=['id', 'created_at'],
        )

        if not res:
            return []

        return res[0]

    async def bm25_search(
        self,
        query: str,
        limit: int = 10,
        filter_expr: str = '',
        collection_name: str = 'Graphiti_Nodes',
    ) -> list[dict]:
        """
        Perform BM25 full-text search using Milvus sparse vector index.

        Args:
            query: The raw text query to search for
            limit: Maximum number of results to return
            filter_expr: Optional Milvus filter expression
            collection_name: The collection to search in

        Returns:
            List of search results with id and score
        """
        await self._ensure_client()
        assert self.client is not None  # For type checker

        if not query or not query.strip():
            return []

        search_params = {
            'params': {'drop_ratio_search': 0.2},
        }

        res = await self.client.search(
            collection_name=collection_name,
            data=[query],  # Raw text query - Milvus BM25 function handles conversion
            anns_field='sparse',  # Search on the sparse vector field
            limit=limit,
            filter=filter_expr,
            output_fields=['id'],
            search_params=search_params,
        )

        if not res:
            return []

        return res[0]

    async def delete(self, ids: list[str], collection_name: str = 'Graphiti_Nodes'):
        await self._ensure_client()
        assert self.client is not None  # For type checker
        await self.client.delete(collection_name=collection_name, pks=ids)

    async def get_vectors(
        self, ids: list[str], collection_name: str = 'Graphiti_Nodes'
    ) -> list[dict]:
        await self._ensure_client()
        assert self.client is not None  # For type checker

        res = await self.client.query(
            collection_name=collection_name,
            filter=f'id in {json.dumps(ids)}',
            output_fields=['id', 'embedding'],
        )
        return res

    async def close(self):
        if self.client:
            await self.client.close()
