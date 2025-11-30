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

import asyncio
import logging
from typing import Any, Dict, Set

from .connection import AsyncNebulaPool, AsyncSession

logger = logging.getLogger(__name__)


class SchemaManager:
    def __init__(self, pool: AsyncNebulaPool, username: str, password: str, space: str):
        self.pool = pool
        self.username = username
        self.password = password
        self.space = space
        self._tag_cache: Dict[str, Set[str]] = {}
        self._edge_cache: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self):
        # Pre-load existing schema
        async with AsyncSession(self.pool, self.username, self.password, self.space) as session:
            # Load Tags
            resp = await session.execute('SHOW TAGS')
            if resp.is_succeeded():
                for row in resp.rows:
                    tag_name = row.values[0].s_val.decode('utf-8')
                    self._tag_cache[tag_name] = await self._fetch_properties(
                        session, 'TAG', tag_name
                    )

            # Load Edges
            resp = await session.execute('SHOW EDGES')
            if resp.is_succeeded():
                for row in resp.rows:
                    edge_name = row.values[0].s_val.decode('utf-8')
                    self._edge_cache[edge_name] = await self._fetch_properties(
                        session, 'EDGE', edge_name
                    )

    async def _fetch_properties(
        self, session: AsyncSession, type_kind: str, type_name: str
    ) -> Set[str]:
        props = set()
        resp = await session.execute(f'DESCRIBE {type_kind} `{type_name}`')
        if resp.is_succeeded():
            for row in resp.rows:
                prop_name = row.values[0].s_val.decode('utf-8')
                props.add(prop_name)
        return props

    async def ensure_schema(self, tag_name: str, properties: Dict[str, Any]):
        # Fast path: check without lock (common case - schema exists)
        if tag_name in self._tag_cache:
            existing_props = self._tag_cache[tag_name]
            if all(key in existing_props for key in properties.keys()):
                return  # All properties exist, no lock needed

        # Slow path: acquire lock and double-check
        async with self._lock:
            if tag_name not in self._tag_cache:
                await self._create_tag(tag_name, properties)
            else:
                await self._alter_tag_if_needed(tag_name, properties)

    async def ensure_edge_schema(self, edge_name: str, properties: Dict[str, Any]):
        # Fast path: check without lock (common case - schema exists)
        if edge_name in self._edge_cache:
            existing_props = self._edge_cache[edge_name]
            if all(key in existing_props for key in properties.keys()):
                return  # All properties exist, no lock needed

        # Slow path: acquire lock and double-check
        async with self._lock:
            if edge_name not in self._edge_cache:
                await self._create_edge(edge_name, properties)
            else:
                await self._alter_edge_if_needed(edge_name, properties)

    def _get_nebula_type(self, value: Any) -> str:
        if isinstance(value, bool):
            return 'BOOL'
        elif isinstance(value, int):
            return 'INT64'
        elif isinstance(value, float):
            return 'DOUBLE'
        else:
            return 'STRING'

    async def _create_tag(self, tag_name: str, properties: Dict[str, Any]):
        props_def = []
        for key, value in properties.items():
            type_str = self._get_nebula_type(value)
            props_def.append(f'`{key}` {type_str}')

        # Always ensure 'uuid' or similar if needed, but Graphiti usually manages IDs externally
        # Assuming properties contains all we need.

        stmt = f'CREATE TAG IF NOT EXISTS `{tag_name}` ({", ".join(props_def)})'
        await self._execute_ddl(stmt)
        self._tag_cache[tag_name] = set(properties.keys())

    async def _alter_tag_if_needed(self, tag_name: str, properties: Dict[str, Any]):
        existing_props = self._tag_cache[tag_name]
        new_props = []
        for key, value in properties.items():
            if key not in existing_props:
                type_str = self._get_nebula_type(value)
                new_props.append(f'ADD `{key}` {type_str}')

        if new_props:
            stmt = f'ALTER TAG `{tag_name}` {", ".join(new_props)}'
            await self._execute_ddl(stmt)
            self._tag_cache[tag_name].update(properties.keys())

    async def _create_edge(self, edge_name: str, properties: Dict[str, Any]):
        props_def = []
        for key, value in properties.items():
            type_str = self._get_nebula_type(value)
            props_def.append(f'`{key}` {type_str}')

        stmt = f'CREATE EDGE IF NOT EXISTS `{edge_name}` ({", ".join(props_def)})'
        await self._execute_ddl(stmt)
        self._edge_cache[edge_name] = set(properties.keys())

    async def _alter_edge_if_needed(self, edge_name: str, properties: Dict[str, Any]):
        existing_props = self._edge_cache[edge_name]
        new_props = []
        for key, value in properties.items():
            if key not in existing_props:
                type_str = self._get_nebula_type(value)
                new_props.append(f'ADD `{key}` {type_str}')

        if new_props:
            stmt = f'ALTER EDGE `{edge_name}` {", ".join(new_props)}'
            await self._execute_ddl(stmt)
            self._edge_cache[edge_name].update(properties.keys())

    async def _execute_ddl(self, stmt: str):
        logger.info(f'Executing DDL: {stmt}')
        async with AsyncSession(self.pool, self.username, self.password, self.space) as session:
            resp = await session.execute(stmt)
            if not resp.is_succeeded():
                raise RuntimeError(f'DDL failed: {resp.error_msg}')

            # Wait for schema propagation
            await asyncio.sleep(2)  # 2 heartbeats usually
