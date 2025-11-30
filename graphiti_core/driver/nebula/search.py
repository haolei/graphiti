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
from datetime import datetime
from typing import Any, TYPE_CHECKING

from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_filters import ComparisonOperator, SearchFilters

if TYPE_CHECKING:
    from .driver import NebulaDriver

logger = logging.getLogger(__name__)


class NebulaSearch(SearchInterface):
    def _build_group_filter(self, group_ids: list[str] | None) -> str:
        """Build Milvus filter expression for group_ids."""
        if not group_ids:
            return ''
        ids_str = ', '.join([f'"{gid}"' for gid in group_ids])
        return f'group_id in [{ids_str}]'

    def _build_node_search_filter(self, search_filter: SearchFilters | None) -> str:
        """
        Build Milvus filter expression from SearchFilters for node searches.
        
        Note: Milvus uses scalar field filtering with expressions like:
        - field in ["value1", "value2"]
        - field == "value"
        - field > 100
        - array_contains(array_field, "value")
        - array_contains_any(array_field, ["value1", "value2"])
        
        SearchFilters.node_labels is stored in Milvus as a JSON array field.
        """
        if not search_filter:
            return ''
        
        filters = []
        
        # Handle node_labels filter
        # The labels field is a list, so we need to check if any of the requested labels match
        if search_filter.node_labels:
            # Build OR condition: label1 in labels OR label2 in labels OR ...
            # Using array_contains for each label
            label_conditions = [
                f'array_contains(labels, "{label}")' for label in search_filter.node_labels
            ]
            if len(label_conditions) == 1:
                filters.append(label_conditions[0])
            else:
                # Group with OR
                filters.append(f"({' || '.join(label_conditions)})")
        
        return ' && '.join(filters) if filters else ''

    def _build_edge_search_filter(self, search_filter: SearchFilters | None) -> str:
        """
        Build Milvus filter expression from SearchFilters for edge searches.
        
        Handles edge-specific filters like edge_types, edge_uuids, and date filters.
        """
        if not search_filter:
            return ''
        
        filters = []
        
        # Handle edge_types filter
        if search_filter.edge_types:
            types_str = ', '.join([f'"{t}"' for t in search_filter.edge_types])
            filters.append(f'name in [{types_str}]')
        
        # Handle edge_uuids filter
        if search_filter.edge_uuids:
            uuids_str = ', '.join([f'"{u}"' for u in search_filter.edge_uuids])
            filters.append(f'uuid in [{uuids_str}]')
        
        # Handle date filters
        filters.extend(self._build_date_filters(search_filter))
        
        return ' && '.join(filters) if filters else ''

    def _build_date_filters(self, search_filter: SearchFilters) -> list[str]:
        """Build Milvus filter expressions for date fields."""
        filters = []
        
        # Helper to convert operator to Milvus syntax
        def op_to_milvus(op: ComparisonOperator) -> str:
            mapping = {
                ComparisonOperator.equals: '==',
                ComparisonOperator.not_equals: '!=',
                ComparisonOperator.greater_than: '>',
                ComparisonOperator.less_than: '<',
                ComparisonOperator.greater_than_equal: '>=',
                ComparisonOperator.less_than_equal: '<=',
            }
            return mapping.get(op, '==')
        
        # Helper to build date filter expression
        def build_date_filter(field: str, date_filters: list[list[Any]]) -> str:
            or_clauses = []
            for or_list in date_filters:
                and_clauses = []
                for date_filter in or_list:
                    if date_filter.comparison_operator == ComparisonOperator.is_null:
                        and_clauses.append(f'{field} == null')
                    elif date_filter.comparison_operator == ComparisonOperator.is_not_null:
                        and_clauses.append(f'{field} != null')
                    elif date_filter.date:
                        timestamp = int(date_filter.date.timestamp() * 1000)  # Convert to milliseconds
                        op = op_to_milvus(date_filter.comparison_operator)
                        and_clauses.append(f'{field} {op} {timestamp}')
                
                if and_clauses:
                    or_clauses.append(f"({' && '.join(and_clauses)})")
            
            return f"({' || '.join(or_clauses)})" if or_clauses else ''
        
        # Process each date field
        if search_filter.valid_at:
            filter_expr = build_date_filter('valid_at', search_filter.valid_at)
            if filter_expr:
                filters.append(filter_expr)
        
        if search_filter.invalid_at:
            filter_expr = build_date_filter('invalid_at', search_filter.invalid_at)
            if filter_expr:
                filters.append(filter_expr)
        
        if search_filter.created_at:
            filter_expr = build_date_filter('created_at', search_filter.created_at)
            if filter_expr:
                filters.append(filter_expr)
        
        if search_filter.expired_at:
            filter_expr = build_date_filter('expired_at', search_filter.expired_at)
            if filter_expr:
                filters.append(filter_expr)
        
        return filters

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: SearchFilters | None,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        # Build filter expressions for Milvus
        filters = []
        
        # Add group_id filter
        group_filter = self._build_group_filter(group_ids)
        if group_filter:
            filters.append(group_filter)
        
        # Add search filters
        search_filter_expr = self._build_node_search_filter(search_filter)
        if search_filter_expr:
            filters.append(search_filter_expr)
        
        expr = ' && '.join(filters) if filters else ''

        # Search Milvus
        results = await driver.milvus.search(
            search_vector, limit=limit, filter_expr=expr, collection_name=driver.nodes_collection
        )

        if not results:
            return []

        # Filter by score
        uuids = []
        for hit in results:
            if hit.score >= min_score:
                uuids.append(hit.id)

        if not uuids:
            return []

        # Fetch nodes from Nebula
        return await EntityNode.get_by_uuids(driver, uuids)

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: SearchFilters | None,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        # Build filter expressions for Milvus
        filters = []
        
        # Add group_id filter
        group_filter = self._build_group_filter(group_ids)
        if group_filter:
            filters.append(group_filter)
        
        # Add search filters
        search_filter_expr = self._build_edge_search_filter(search_filter)
        if search_filter_expr:
            filters.append(search_filter_expr)
        
        expr = ' && '.join(filters) if filters else ''

        results = await driver.milvus.search(
            search_vector, limit=limit, filter_expr=expr, collection_name=driver.edges_collection
        )

        if not results:
            return []

        uuids = []
        for hit in results:
            if hit.score >= min_score:
                uuids.append(hit.id)

        if not uuids:
            return []

        return await EntityEdge.get_by_uuids(driver, uuids)

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: SearchFilters | None,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform BM25 full-text search on nodes using Milvus sparse vector index.

        Args:
            driver: The Nebula driver with Milvus adapter
            query: The raw text query to search for
            search_filter: Search filters for node labels and other criteria
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            List of EntityNode objects matching the query
        """
        if not query or not query.strip():
            return []

        # Build filter expressions for Milvus
        filters = []
        
        # Add group_id filter
        group_filter = self._build_group_filter(group_ids)
        if group_filter:
            filters.append(group_filter)
        
        # Add search filters
        search_filter_expr = self._build_node_search_filter(search_filter)
        if search_filter_expr:
            filters.append(search_filter_expr)
        
        expr = ' && '.join(filters) if filters else ''

        # Perform BM25 search using Milvus
        results = await driver.milvus.bm25_search(
            query=query, limit=limit, filter_expr=expr, collection_name=driver.nodes_collection
        )

        if not results:
            return []

        # Extract UUIDs from search results
        uuids = [hit.id for hit in results]

        if not uuids:
            return []

        # Fetch nodes from Nebula
        return await EntityNode.get_by_uuids(driver, uuids)

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: SearchFilters | None,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform BM25 full-text search on edges using Milvus sparse vector index.

        Args:
            driver: The Nebula driver with Milvus adapter
            query: The raw text query to search for
            search_filter: Search filters for edge types, UUIDs, and date criteria
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            List of EntityEdge objects matching the query
        """
        if not query or not query.strip():
            return []

        # Build filter expressions for Milvus
        filters = []
        
        # Add group_id filter
        group_filter = self._build_group_filter(group_ids)
        if group_filter:
            filters.append(group_filter)
        
        # Add search filters
        search_filter_expr = self._build_edge_search_filter(search_filter)
        if search_filter_expr:
            filters.append(search_filter_expr)
        
        expr = ' && '.join(filters) if filters else ''

        # Perform BM25 search using Milvus
        results = await driver.milvus.bm25_search(
            query=query, limit=limit, filter_expr=expr, collection_name=driver.edges_collection
        )

        if not results:
            return []

        # Extract UUIDs from search results
        uuids = [hit.id for hit in results]

        if not uuids:
            return []

        # Fetch edges from Nebula
        return await EntityEdge.get_by_uuids(driver, uuids)

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: SearchFilters | None,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform BM25 full-text search on episodes using Milvus sparse vector index.

        Args:
            driver: The Nebula driver with Milvus adapter
            query: The raw text query to search for
            search_filter: Search filters (not typically used for episodes)
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            List of EpisodicNode objects matching the query
        """
        if not query or not query.strip():
            return []

        # Build filter expression for Milvus
        # Episodes primarily filter by group_id
        expr = self._build_group_filter(group_ids)

        # Perform BM25 search using Milvus
        results = await driver.milvus.bm25_search(
            query=query, limit=limit, filter_expr=expr, collection_name=driver.episodes_collection
        )

        if not results:
            return []

        # Extract UUIDs from search results
        uuids = [hit.id for hit in results]

        if not uuids:
            return []

        # Fetch episodes from Nebula
        return await EpisodicNode.get_by_uuids(driver, uuids)

    def build_node_search_filters(self, search_filters: SearchFilters | None) -> SearchFilters | None:
        """Pass through node search filters - Milvus filtering is handled in search methods."""
        return search_filters

    def build_edge_search_filters(self, search_filters: SearchFilters | None) -> SearchFilters | None:
        """Pass through edge search filters - Milvus filtering is handled in search methods."""
        return search_filters
