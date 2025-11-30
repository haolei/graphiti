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
from typing import Any

try:
    from nebula3.fbthrift.util.asyncio import create_client
    from nebula3.graph import GraphService
except ImportError:
    create_client = None
    GraphService = None

logger = logging.getLogger(__name__)


class AsyncNebulaPool:
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
        self._pool: asyncio.Queue = asyncio.Queue()
        self._sem = asyncio.Semaphore(max_size)
        self._lock = asyncio.Lock()
        self._current_size = 0
        self._closed = False

    async def initialize(self):
        """Initialize the connection pool with minimum connections."""
        if create_client is None:
            raise ImportError(
                'nebula3-python is not installed. Please install it with `pip install nebula3-python`.'
            )

        tasks = [self._create_connection() for _ in range(self._min_size)]
        await asyncio.gather(*tasks)

    async def _create_connection(self):
        """Create a new connection and add it to the pool."""
        if create_client is None or GraphService is None:
            raise ImportError('nebula3-python is not installed')

        async with self._lock:
            if self._current_size >= self._max_size:
                return
            self._current_size += 1

        try:
            # Create context manager
            ctx = create_client(GraphService.Client, host=self._host, port=self._port)
            # Manually enter context to get client
            client = await ctx.__aenter__()
            # Attach context manager to client for later cleanup
            client._ctx_manager = ctx

            self._pool.put_nowait(client)
        except Exception as e:
            async with self._lock:
                self._current_size -= 1
            logger.error(f'Failed to create async nebula client: {e}')
            raise ConnectionError(f'Failed to create async nebula client: {e}')

    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        if self._closed:
            raise RuntimeError('Connection pool is closed')

        await self._sem.acquire()

        if not self._pool.empty():
            return await self._pool.get()

        try:
            await self._create_connection()
            return await self._pool.get()
        except Exception:
            self._sem.release()
            raise

    async def release(self, client: Any):
        """Release a connection back to the pool."""
        if self._closed:
            await self._close_client(client)
            return

        try:
            # Optional: Check if connection is alive?
            # For now, just put it back
            self._pool.put_nowait(client)
        finally:
            self._sem.release()

    async def close(self):
        """Close all connections in the pool."""
        self._closed = True
        while not self._pool.empty():
            client = await self._pool.get()
            await self._close_client(client)

        # Wait for all semaphores to be released?
        # Or just let them be garbage collected since we set _closed=True

    async def _close_client(self, client: Any):
        if hasattr(client, '_ctx_manager'):
            await client._ctx_manager.__aexit__(None, None, None)


class AsyncSession:
    def __init__(self, pool: AsyncNebulaPool, username: str, password: str, space: str):
        self.pool = pool
        self.user = username
        self.pwd = password
        self.space = space
        self.client = None
        self.session_id = None

    async def __aenter__(self):
        self.client = await self.pool.acquire()
        try:
            resp = await self.client.authenticate(self.user, self.pwd)
            if resp.error_code != 0:
                raise RuntimeError(f'Auth failed: {resp.error_msg}')
            self.session_id = resp.session_id

            # Switch space
            # Note: execute requires session_id
            await self.execute(f'USE {self.space}')
            return self
        except Exception:
            if self.client:
                await self.pool.release(self.client)
                self.client = None
            raise

    async def execute(self, stmt: str):
        if not self.client or not self.session_id:
            raise RuntimeError('Session not initialized')
        return await self.client.execute(self.session_id, stmt)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client and self.session_id:
            try:
                await self.client.signout(self.session_id)
            except Exception as e:
                logger.warning(f'Failed to signout session: {e}')
            finally:
                await self.pool.release(self.client)
                self.client = None
