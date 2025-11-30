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

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

from graphiti_core.driver.driver import GraphProvider

load_dotenv()

NEBULA_HOST = os.getenv('NEBULA_HOST', 'localhost')
NEBULA_PORT = int(os.getenv('NEBULA_PORT', '9669'))
NEBULA_USER = os.getenv('NEBULA_USER', 'root')
NEBULA_PASSWORD = os.getenv('NEBULA_PASSWORD', 'nebula')
NEBULA_SPACE = os.getenv('NEBULA_SPACE', 'graphiti')
MILVUS_URI = os.getenv('MILVUS_URI', 'http://localhost:19530')
MILVUS_TOKEN = os.getenv('MILVUS_TOKEN', '')

# Try to import Nebula driver if available
try:
    from graphiti_core.driver.nebula import NebulaDriver
    from graphiti_core.driver.nebula.connection import AsyncNebulaPool, AsyncSession
    from graphiti_core.driver.nebula.driver import NebulaDriverSession
    from graphiti_core.driver.nebula.milvus_adapter import MilvusAdapter
    from graphiti_core.driver.nebula.query_builder import QueryBuilder
    from graphiti_core.driver.nebula.schema_manager import SchemaManager

    HAS_NEBULA = True
except ImportError:
    NebulaDriver = None
    NebulaDriverSession = None
    QueryBuilder = None
    MilvusAdapter = None
    SchemaManager = None
    AsyncNebulaPool = None
    AsyncSession = None
    HAS_NEBULA = False


@pytest.mark.skipif(not HAS_NEBULA, reason='Nebula driver not installed')
class TestNebulaDriver:
    """Comprehensive test suite for Nebula driver."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        with (
            patch('graphiti_core.driver.nebula.driver.AsyncNebulaPool') as mock_pool_class,
            patch('graphiti_core.driver.nebula.driver.MilvusAdapter') as mock_milvus_class,
            patch('graphiti_core.driver.nebula.driver.SchemaManager') as mock_schema_class,
        ):
            self.mock_pool = MagicMock()
            self.mock_milvus = MagicMock()
            self.mock_schema_manager = MagicMock()

            mock_pool_class.return_value = self.mock_pool
            mock_milvus_class.return_value = self.mock_milvus
            mock_schema_class.return_value = self.mock_schema_manager

            self.driver = NebulaDriver(
                host=NEBULA_HOST,
                port=NEBULA_PORT,
                username=NEBULA_USER,
                password=NEBULA_PASSWORD,
                space=NEBULA_SPACE,
                milvus_uri=MILVUS_URI,
                milvus_token=MILVUS_TOKEN,
            )

            # Replace mocks after init
            self.driver.pool = self.mock_pool
            self.driver.milvus = self.mock_milvus
            self.driver.schema_manager = self.mock_schema_manager

    def test_init_with_connection_params(self):
        """Test initialization with connection parameters."""
        with (
            patch('graphiti_core.driver.nebula.driver.AsyncNebulaPool') as mock_pool_class,
            patch('graphiti_core.driver.nebula.driver.MilvusAdapter') as mock_milvus_class,
            patch('graphiti_core.driver.nebula.driver.SchemaManager') as mock_schema_class,
        ):
            driver = NebulaDriver(
                host='test-host',
                port=9669,
                username='test-user',
                password='test-pass',
                space='test-space',
                milvus_uri='http://milvus:19530',
                milvus_token='test-token',
            )

            assert driver.provider == GraphProvider.NEBULA
            assert driver._database == 'test-space'
            assert driver._host == 'test-host'
            assert driver._port == 9669
            assert driver._username == 'test-user'
            assert driver._password == 'test-pass'
            assert driver._space == 'test-space'

            mock_pool_class.assert_called_once_with('test-host', 9669)
            mock_milvus_class.assert_called_once_with('http://milvus:19530', 'test-token')
            mock_schema_class.assert_called_once()

    def test_provider(self):
        """Test driver provider identification."""
        assert self.driver.provider == GraphProvider.NEBULA

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test driver connect method."""
        self.mock_pool.initialize = AsyncMock()
        self.mock_milvus.connect = AsyncMock()
        self.mock_milvus.ensure_collection = AsyncMock()
        self.mock_schema_manager.initialize = AsyncMock()

        await self.driver.connect()

        self.mock_pool.initialize.assert_called_once()
        self.mock_milvus.connect.assert_called_once()
        assert self.mock_milvus.ensure_collection.call_count == 3  # nodes, edges, episodes
        self.mock_schema_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_success(self):
        """Test successful query execution."""
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=([{'result': 'value'}], None, ['result']))

        with patch.object(self.driver, 'session', return_value=mock_session):
            result = await self.driver.execute_query('MATCH (n) RETURN n', param1='value1')

            mock_session.run.assert_called_once()
            records, _, keys = result
            assert records == [{'result': 'value'}]
            assert keys == ['result']

    def test_session_creation(self):
        """Test session creation."""
        session = self.driver.session()
        assert isinstance(session, NebulaDriverSession)

    def test_session_creation_with_database(self):
        """Test session creation with specific database/space."""
        session = self.driver.session(database='custom_space')
        assert isinstance(session, NebulaDriverSession)

    @pytest.mark.asyncio
    async def test_close(self):
        """Test driver close method."""
        self.mock_pool.close = AsyncMock()
        self.mock_milvus.close = AsyncMock()

        await self.driver.close()

        self.mock_pool.close.assert_called_once()
        self.mock_milvus.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_all_indexes(self):
        """Test delete_all_indexes method."""
        self.mock_milvus.client = MagicMock()
        self.mock_milvus.client.drop_collection = AsyncMock()

        await self.driver.delete_all_indexes()

        assert self.mock_milvus.client.drop_collection.call_count == 3  # nodes, edges, episodes

    @pytest.mark.asyncio
    async def test_build_indices_and_constraints(self):
        """Test build_indices_and_constraints method."""
        self.mock_milvus.ensure_collection = AsyncMock()

        await self.driver.build_indices_and_constraints()

        assert self.mock_milvus.ensure_collection.call_count == 3  # nodes, edges, episodes

    @pytest.mark.asyncio
    async def test_build_indices_and_constraints_delete_existing(self):
        """Test build_indices_and_constraints with delete_existing=True."""
        self.mock_milvus.client = MagicMock()
        self.mock_milvus.client.drop_collection = AsyncMock()
        self.mock_milvus.ensure_collection = AsyncMock()

        await self.driver.build_indices_and_constraints(delete_existing=True)

        # Should drop and recreate collections (nodes, edges, episodes)
        assert self.mock_milvus.client.drop_collection.call_count == 3
        assert self.mock_milvus.ensure_collection.call_count == 3


@pytest.mark.skipif(not HAS_NEBULA, reason='Nebula driver not installed')
class TestNebulaDriverSession:
    """Test Nebula driver session functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_pool = MagicMock()
        self.session = NebulaDriverSession(
            pool=self.mock_pool,
            username=NEBULA_USER,
            password=NEBULA_PASSWORD,
            space=NEBULA_SPACE,
        )

    @pytest.mark.asyncio
    async def test_session_async_context_manager(self):
        """Test session can be used as async context manager."""
        mock_inner_session = MagicMock()
        mock_inner_session.__aenter__ = AsyncMock(return_value=mock_inner_session)
        mock_inner_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(
            self.session._session, '__aenter__', AsyncMock(return_value=mock_inner_session)
        ):
            with patch.object(self.session._session, '__aexit__', AsyncMock(return_value=None)):
                async with self.session as s:
                    assert s is self.session

    @pytest.mark.asyncio
    async def test_close_method(self):
        """Test session close method doesn't raise exceptions."""
        await self.session.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_execute_write_passes_session_and_args(self):
        """Test execute_write method passes session and arguments correctly."""

        async def test_func(session, *args, **kwargs):
            assert session is self.session
            assert args == ('arg1', 'arg2')
            assert kwargs == {'key': 'value'}
            return 'result'

        result = await self.session.execute_write(test_func, 'arg1', 'arg2', key='value')
        assert result == 'result'


@pytest.mark.skipif(not HAS_NEBULA, reason='Nebula driver not installed')
class TestNebulaValueConversion:
    """Test Nebula value conversion in session."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_pool = MagicMock()
        self.session = NebulaDriverSession(
            pool=self.mock_pool,
            username=NEBULA_USER,
            password=NEBULA_PASSWORD,
            space=NEBULA_SPACE,
        )

    def test_convert_null_value(self):
        """Test conversion of null values."""
        mock_val = MagicMock()
        mock_val.getType.return_value = mock_val.NVAL

        result = self.session._convert_nebula_value(mock_val)
        assert result is None

    def test_convert_bool_value(self):
        """Test conversion of boolean values."""
        mock_val = MagicMock()
        mock_val.getType.return_value = mock_val.BVAL
        mock_val.get_bVal.return_value = True

        result = self.session._convert_nebula_value(mock_val)
        assert result is True

    def test_convert_int_value(self):
        """Test conversion of integer values."""
        mock_val = MagicMock()
        mock_val.getType.return_value = mock_val.IVAL
        mock_val.get_iVal.return_value = 42

        result = self.session._convert_nebula_value(mock_val)
        assert result == 42

    def test_convert_float_value(self):
        """Test conversion of float values."""
        mock_val = MagicMock()
        mock_val.getType.return_value = mock_val.FVAL
        mock_val.get_fVal.return_value = 3.14

        result = self.session._convert_nebula_value(mock_val)
        assert result == 3.14

    def test_convert_string_value(self):
        """Test conversion of string values."""
        mock_val = MagicMock()
        mock_val.getType.return_value = mock_val.SVAL
        mock_val.get_sVal.return_value = b'test_string'

        result = self.session._convert_nebula_value(mock_val)
        assert result == 'test_string'

    def test_convert_list_value(self):
        """Test conversion of list values."""
        mock_inner_val = MagicMock()
        mock_inner_val.getType.return_value = mock_inner_val.IVAL
        mock_inner_val.get_iVal.return_value = 1

        mock_list = MagicMock()
        mock_list.values = [mock_inner_val]

        mock_val = MagicMock()
        mock_val.getType.return_value = mock_val.LVAL
        mock_val.get_lVal.return_value = mock_list

        result = self.session._convert_nebula_value(mock_val)
        assert result == [1]


@pytest.mark.skipif(not HAS_NEBULA, reason='Nebula driver not installed')
class TestQueryBuilder:
    """Test QueryBuilder functionality."""

    def test_format_value_string(self):
        """Test _format_value for strings."""
        result = QueryBuilder._format_value('test')
        assert result == '"test"'

    def test_format_value_string_with_quotes(self):
        """Test _format_value for strings with quotes."""
        result = QueryBuilder._format_value('test "quoted" value')
        assert result == '"test \\"quoted\\" value"'

    def test_format_value_int(self):
        """Test _format_value for integers."""
        result = QueryBuilder._format_value(42)
        assert result == '42'

    def test_format_value_float(self):
        """Test _format_value for floats."""
        result = QueryBuilder._format_value(3.14)
        assert result == '3.14'

    def test_format_value_bool_true(self):
        """Test _format_value for True."""
        result = QueryBuilder._format_value(True)
        assert result == 'true'

    def test_format_value_bool_false(self):
        """Test _format_value for False."""
        result = QueryBuilder._format_value(False)
        assert result == 'false'

    def test_format_value_none(self):
        """Test _format_value for None."""
        result = QueryBuilder._format_value(None)
        assert result == 'NULL'

    def test_format_value_list(self):
        """Test _format_value for lists."""
        result = QueryBuilder._format_value([1, 2, 3])
        assert '[1, 2, 3]' in result or '"[1, 2, 3]"' in result

    def test_format_value_dict(self):
        """Test _format_value for dicts."""
        result = QueryBuilder._format_value({'key': 'value'})
        assert 'key' in result

    def test_build_upsert_vertex(self):
        """Test build_upsert_vertex method."""
        props = {'name': 'test_node', 'group_id': 'test_group', 'summary': 'test summary'}
        result = QueryBuilder.build_upsert_vertex('Entity', 'uuid123', props)

        assert 'UPSERT VERTEX ON' in result
        assert '`Entity`' in result
        assert '"uuid123"' in result
        assert 'SET' in result
        assert '`name`' in result
        assert '`group_id`' in result

    def test_build_insert_edge(self):
        """Test build_insert_edge method."""
        props = {'uuid': 'edge123', 'fact': 'test fact', 'group_id': 'test_group'}
        result = QueryBuilder.build_insert_edge(
            'RELATED_TO', 'src_uuid', 'dst_uuid', 1234567890, props
        )

        assert 'INSERT EDGE' in result
        assert '`RELATED_TO`' in result
        assert '"src_uuid"' in result
        assert '"dst_uuid"' in result
        assert '@1234567890' in result


@pytest.mark.skipif(not HAS_NEBULA, reason='Nebula driver not installed')
class TestMilvusAdapter:
    """Test MilvusAdapter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = MilvusAdapter(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            collection_name='test_collection',
            dim=1536,
        )

    def test_init(self):
        """Test MilvusAdapter initialization."""
        assert self.adapter.uri == MILVUS_URI
        assert self.adapter.token == MILVUS_TOKEN
        assert self.adapter.collection_name == 'test_collection'
        assert self.adapter.dim == 1536
        assert self.adapter.client is None

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test MilvusAdapter connect method."""
        with patch(
            'graphiti_core.driver.nebula.milvus_adapter.AsyncMilvusClient'
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            await self.adapter.connect()

            mock_client_class.assert_called_once_with(MILVUS_URI, token=MILVUS_TOKEN)
            assert self.adapter.client is mock_client

    @pytest.mark.asyncio
    async def test_ensure_collection_already_exists(self):
        """Test ensure_collection when collection already exists."""
        mock_client = MagicMock()
        mock_client.has_collection = AsyncMock(return_value=True)
        self.adapter.client = mock_client

        await self.adapter.ensure_collection('test_collection')

        mock_client.has_collection.assert_called_once_with('test_collection')

    @pytest.mark.asyncio
    async def test_upsert_vectors(self):
        """Test upsert_vectors method."""
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock()
        self.adapter.client = mock_client

        data = [{'id': 'test1', 'embedding': [0.1] * 1536}]
        await self.adapter.upsert_vectors(data, 'test_collection')

        mock_client.upsert.assert_called_once_with(collection_name='test_collection', data=data)

    @pytest.mark.asyncio
    async def test_search(self):
        """Test search method."""
        mock_client = MagicMock()
        mock_results = [[MagicMock(id='test1', score=0.9)]]
        mock_client.search = AsyncMock(return_value=mock_results)
        self.adapter.client = mock_client

        vector = [0.1] * 1536
        results = await self.adapter.search(vector, limit=10, collection_name='test_collection')

        mock_client.search.assert_called_once()
        assert results == mock_results[0]

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete method."""
        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        self.adapter.client = mock_client

        await self.adapter.delete(['id1', 'id2'], 'test_collection')

        mock_client.delete.assert_called_once_with(
            collection_name='test_collection', pks=['id1', 'id2']
        )

    @pytest.mark.asyncio
    async def test_get_vectors(self):
        """Test get_vectors method."""
        mock_client = MagicMock()
        mock_results = [{'id': 'test1', 'embedding': [0.1] * 1536}]
        mock_client.query = AsyncMock(return_value=mock_results)
        self.adapter.client = mock_client

        results = await self.adapter.get_vectors(['test1'], 'test_collection')

        mock_client.query.assert_called_once()
        assert results == mock_results

    @pytest.mark.asyncio
    async def test_close(self):
        """Test close method."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        self.adapter.client = mock_client

        await self.adapter.close()

        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_bm25_search(self):
        """Test bm25_search method for full-text search."""
        mock_client = MagicMock()
        mock_results = [[MagicMock(id='test1', score=0.85)]]
        mock_client.search = AsyncMock(return_value=mock_results)
        self.adapter.client = mock_client

        results = await self.adapter.bm25_search(
            query='test query', limit=10, collection_name='test_collection'
        )

        mock_client.search.assert_called_once()
        # Verify search was called with text query and sparse field
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs['data'] == ['test query']
        assert call_kwargs['anns_field'] == 'sparse'
        assert call_kwargs['collection_name'] == 'test_collection'
        assert results == mock_results[0]

    @pytest.mark.asyncio
    async def test_bm25_search_empty_query(self):
        """Test bm25_search with empty query returns empty list."""
        mock_client = MagicMock()
        self.adapter.client = mock_client

        results = await self.adapter.bm25_search(query='', collection_name='test_collection')
        assert results == []

        results = await self.adapter.bm25_search(query='   ', collection_name='test_collection')
        assert results == []

    @pytest.mark.asyncio
    async def test_bm25_search_with_filter(self):
        """Test bm25_search with filter expression."""
        mock_client = MagicMock()
        mock_results = [[MagicMock(id='test1', score=0.9)]]
        mock_client.search = AsyncMock(return_value=mock_results)
        self.adapter.client = mock_client

        results = await self.adapter.bm25_search(
            query='test query',
            filter_expr='group_id in ["group1"]',
            collection_name='test_collection',
        )

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs['filter'] == 'group_id in ["group1"]'


@pytest.mark.skipif(not HAS_NEBULA, reason='Nebula driver not installed')
class TestSchemaManager:
    """Test SchemaManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_pool = MagicMock()
        self.schema_manager = SchemaManager(
            pool=self.mock_pool,
            username=NEBULA_USER,
            password=NEBULA_PASSWORD,
            space=NEBULA_SPACE,
        )

    def test_get_nebula_type_bool(self):
        """Test _get_nebula_type for boolean."""
        result = self.schema_manager._get_nebula_type(True)
        assert result == 'BOOL'

    def test_get_nebula_type_int(self):
        """Test _get_nebula_type for integer."""
        result = self.schema_manager._get_nebula_type(42)
        assert result == 'INT64'

    def test_get_nebula_type_float(self):
        """Test _get_nebula_type for float."""
        result = self.schema_manager._get_nebula_type(3.14)
        assert result == 'DOUBLE'

    def test_get_nebula_type_string(self):
        """Test _get_nebula_type for string."""
        result = self.schema_manager._get_nebula_type('test')
        assert result == 'STRING'

    def test_get_nebula_type_list(self):
        """Test _get_nebula_type for list (defaults to STRING)."""
        result = self.schema_manager._get_nebula_type([1, 2, 3])
        assert result == 'STRING'


@pytest.mark.skipif(not HAS_NEBULA, reason='Nebula driver not installed')
class TestAsyncNebulaPool:
    """Test AsyncNebulaPool functionality."""

    def test_init(self):
        """Test AsyncNebulaPool initialization."""
        pool = AsyncNebulaPool(
            host=NEBULA_HOST, port=NEBULA_PORT, min_size=2, max_size=20, timeout=10
        )

        assert pool._host == NEBULA_HOST
        assert pool._port == NEBULA_PORT
        assert pool._min_size == 2
        assert pool._max_size == 20
        assert pool._timeout == 10
        assert pool._current_size == 0
        assert pool._closed is False

    @pytest.mark.asyncio
    async def test_acquire_when_closed(self):
        """Test acquire raises error when pool is closed."""
        pool = AsyncNebulaPool(host=NEBULA_HOST, port=NEBULA_PORT)
        pool._closed = True

        with pytest.raises(RuntimeError, match='Connection pool is closed'):
            await pool.acquire()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test pool close method."""
        pool = AsyncNebulaPool(host=NEBULA_HOST, port=NEBULA_PORT)
        await pool.close()
        assert pool._closed is True


# Try to import NebulaSearch if available
try:
    from graphiti_core.driver.nebula.search import NebulaSearch

    HAS_NEBULA_SEARCH = True
except ImportError:
    NebulaSearch = None
    HAS_NEBULA_SEARCH = False


@pytest.mark.skipif(not HAS_NEBULA_SEARCH, reason='Nebula search not installed')
class TestNebulaSearch:
    """Test NebulaSearch functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.search = NebulaSearch()

    @pytest.mark.asyncio
    async def test_node_fulltext_search_empty_query(self):
        """Test node_fulltext_search returns empty for blank query."""
        mock_driver = MagicMock()

        result = await self.search.node_fulltext_search(
            driver=mock_driver, query='', search_filter=None
        )
        assert result == []

        result = await self.search.node_fulltext_search(
            driver=mock_driver, query='   ', search_filter=None
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_node_fulltext_search_success(self):
        """Test node_fulltext_search with valid query."""
        mock_driver = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = 'test-uuid'
        mock_driver.milvus.bm25_search = AsyncMock(return_value=[mock_hit])
        mock_driver.nodes_collection = 'test_nodes'

        with patch(
            'graphiti_core.driver.nebula.search.EntityNode.get_by_uuids', new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = [MagicMock(uuid='test-uuid')]

            result = await self.search.node_fulltext_search(
                driver=mock_driver, query='test query', search_filter=None
            )

            mock_driver.milvus.bm25_search.assert_called_once()
            mock_get.assert_called_once_with(mock_driver, ['test-uuid'])

    @pytest.mark.asyncio
    async def test_edge_fulltext_search_empty_query(self):
        """Test edge_fulltext_search returns empty for blank query."""
        mock_driver = MagicMock()

        result = await self.search.edge_fulltext_search(
            driver=mock_driver, query='', search_filter=None
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_edge_fulltext_search_success(self):
        """Test edge_fulltext_search with valid query."""
        mock_driver = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = 'edge-uuid'
        mock_driver.milvus.bm25_search = AsyncMock(return_value=[mock_hit])
        mock_driver.edges_collection = 'test_edges'

        with patch(
            'graphiti_core.driver.nebula.search.EntityEdge.get_by_uuids', new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = [MagicMock(uuid='edge-uuid')]

            result = await self.search.edge_fulltext_search(
                driver=mock_driver, query='test query', search_filter=None
            )

            mock_driver.milvus.bm25_search.assert_called_once()
            mock_get.assert_called_once_with(mock_driver, ['edge-uuid'])

    @pytest.mark.asyncio
    async def test_episode_fulltext_search_empty_query(self):
        """Test episode_fulltext_search returns empty for blank query."""
        mock_driver = MagicMock()

        result = await self.search.episode_fulltext_search(
            driver=mock_driver, query='', search_filter=None
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_episode_fulltext_search_success(self):
        """Test episode_fulltext_search with valid query."""
        mock_driver = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = 'episode-uuid'
        mock_driver.milvus.bm25_search = AsyncMock(return_value=[mock_hit])
        mock_driver.episodes_collection = 'test_episodes'

        with patch(
            'graphiti_core.driver.nebula.search.EpisodicNode.get_by_uuids', new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = [MagicMock(uuid='episode-uuid')]

            result = await self.search.episode_fulltext_search(
                driver=mock_driver, query='test query', search_filter=None
            )

            mock_driver.milvus.bm25_search.assert_called_once()
            mock_get.assert_called_once_with(mock_driver, ['episode-uuid'])

    @pytest.mark.asyncio
    async def test_fulltext_search_with_group_ids(self):
        """Test fulltext search builds correct filter for group_ids."""
        mock_driver = MagicMock()
        mock_driver.milvus.bm25_search = AsyncMock(return_value=[])
        mock_driver.nodes_collection = 'test_nodes'

        await self.search.node_fulltext_search(
            driver=mock_driver, query='test', search_filter=None, group_ids=['group1', 'group2']
        )

        call_kwargs = mock_driver.milvus.bm25_search.call_args[1]
        assert 'group1' in call_kwargs['filter_expr']
        assert 'group2' in call_kwargs['filter_expr']


# Integration test
@pytest.mark.skipif(not HAS_NEBULA, reason='Nebula driver not installed')
class TestNebulaDriverIntegration:
    """Integration tests for Nebula driver."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_basic_integration_with_real_nebula(self):
        """Basic integration test with real Nebula instance."""
        try:
            driver = NebulaDriver(
                host=NEBULA_HOST,
                port=NEBULA_PORT,
                username=NEBULA_USER,
                password=NEBULA_PASSWORD,
                space=NEBULA_SPACE,
                milvus_uri=MILVUS_URI,
                milvus_token=MILVUS_TOKEN,
            )

            await driver.connect()

            # Test basic query execution
            async with driver.session() as session:
                result = await session.run('RETURN 1 as test')
                assert result is not None

            await driver.close()

        except Exception as e:
            pytest.skip(f'Nebula not available for integration test: {e}')
