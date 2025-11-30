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
from typing import Any, Optional

try:
    from nebula3.Config import Config
    from nebula3.gclient.net import ConnectionPool
except ImportError:
    Config = None
    ConnectionPool = None

logger = logging.getLogger(__name__)


class AsyncNebulaPool:
    """Async wrapper around nebula3-python's ConnectionPool using asyncio.to_thread."""

    def __init__(
        self,
        host: str,
        port: int,
        min_size: int = 2,
        max_size: int = 20,
        timeout: int = 10,
    ):
        self._host = host
        self._port = port
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._pool: Optional[Any] = None
        self._closed = False
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the connection pool."""
        if self._pool is not None:
            return  # Already initialized

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._pool is not None:
                return

            if ConnectionPool is None or Config is None:
                raise ImportError(
                    'nebula3-python is not installed. Please install it with `pip install nebula3-python`.'
                )

            config = Config()
            config.max_connection_pool_size = self._max_size
            config.min_connection_pool_size = self._min_size
            config.timeout = self._timeout * 1000  # Convert to milliseconds

            pool = ConnectionPool()

            # Initialize pool in thread to avoid blocking
            ok = await asyncio.to_thread(
                pool.init, [(self._host, self._port)], config
            )
            if not ok:
                raise ConnectionError(
                    f'Failed to initialize connection pool to {self._host}:{self._port}'
                )
            self._pool = pool
            logger.info(f'Nebula connection pool initialized: {self._host}:{self._port}')

    async def get_session_async(self, username: str, password: str):
        """Get a session from the pool asynchronously (auto-initializes if needed)."""
        await self.initialize()
        assert self._pool is not None  # Guaranteed after initialize()
        return await asyncio.to_thread(self._pool.get_session, username, password)

    def get_session(self, username: str, password: str):
        """Get a session from the pool (synchronous, for use with asyncio.to_thread)."""
        if self._pool is None:
            raise RuntimeError('Connection pool not initialized')
        return self._pool.get_session(username, password)

    def session_context(self, username: str, password: str):
        """Get a session context manager from the pool (synchronous)."""
        if self._pool is None:
            raise RuntimeError('Connection pool not initialized')
        return self._pool.session_context(username, password)

    async def close(self):
        """Close the connection pool."""
        self._closed = True
        if self._pool is not None:
            await asyncio.to_thread(self._pool.close)
            self._pool = None
            logger.info('Nebula connection pool closed')


class AsyncSession:
    """Async wrapper for Nebula session using asyncio.to_thread."""

    def __init__(self, pool: AsyncNebulaPool, username: str, password: str, space: str):
        self._pool = pool
        self._username = username
        self._password = password
        self._space = space
        self._session = None

    async def __aenter__(self):
        # Get session from pool (auto-initializes pool if needed)
        self._session = await self._pool.get_session_async(self._username, self._password)
        # Switch to the specified space
        await self.execute(f'USE {self._space}')
        return self

    async def execute(self, stmt: str):
        """Execute a statement asynchronously."""
        if self._session is None:
            raise RuntimeError('Session not initialized')
        return await asyncio.to_thread(self._session.execute, stmt)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session is not None:
            try:
                # Release session back to pool
                await asyncio.to_thread(self._session.release)
            except Exception as e:
                logger.warning(f'Failed to release session: {e}')
            finally:
                self._session = None
