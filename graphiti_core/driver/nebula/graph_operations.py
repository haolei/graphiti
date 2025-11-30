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
import asyncio
from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.edges import Edge, EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

logger = logging.getLogger(__name__)


class NebulaGraphOperations(GraphOperationsInterface):
    async def node_save(self, node: Any, driver: Any) -> None:
        """Persist (create or update) a single node."""
        if isinstance(node, dict):
            if 'content' in node and 'source' in node:
                await self._save_episodic_node(node, driver)
            else:
                await self._save_entity_node(node, driver)
        elif isinstance(node, EntityNode):
            await self._save_entity_node(node, driver)
        elif isinstance(node, EpisodicNode):
            await self._save_episodic_node(node, driver)
        elif isinstance(node, CommunityNode):
            await self._save_community_node(node, driver)
        else:
            raise NotImplementedError(f'Saving node type {type(node)} not supported')

    async def _save_entity_node(self, node: Any, driver: Any):
        if isinstance(node, dict):
            uuid = node.get('uuid')
            name = node.get('name')
            group_id = node.get('group_id')
            summary = node.get('summary')
            created_at = node.get('created_at')
            name_embedding = node.get('name_embedding')
            labels = node.get('labels', ['Entity'])
            attributes = node.copy()
            for k in [
                'uuid',
                'name',
                'group_id',
                'summary',
                'created_at',
                'name_embedding',
                'labels',
            ]:
                attributes.pop(k, None)
        else:
            uuid = node.uuid
            name = node.name
            group_id = node.group_id
            summary = node.summary
            created_at = node.created_at
            name_embedding = node.name_embedding
            labels = node.labels if hasattr(node, 'labels') else ['Entity']
            attributes = node.attributes

        props = {
            'uuid': uuid,
            'name': name,
            'group_id': group_id,
            'summary': summary,
            'created_at': self._get_timestamp(created_at),
        }

        if attributes:
            props.update(attributes)

        # 2. Ensure Schema
        # We use "Entity" as the main tag.
        await driver.schema_manager.ensure_schema('Entity', props)

        # 3. Write to Nebula
        ngql = driver.query_builder.build_upsert_vertex('Entity', uuid, props)
        async with driver.session() as session:
            await session.execute(ngql)

        # 4. Write to Milvus
        if name_embedding:
            # Create text for BM25 search from name and summary
            text_content = f'{name or ""} {summary or ""}'
            milvus_data = {
                'id': uuid,
                'embedding': name_embedding,
                'text': text_content,  # For BM25 full-text search
                'created_at': props['created_at'],
                'group_id': group_id,
                'labels': labels,  # For filtering by node labels
            }
            await driver.milvus.upsert_vectors(
                [milvus_data], collection_name=driver.nodes_collection
            )

    async def _save_episodic_node(self, node: Any, driver: Any):
        if isinstance(node, dict):
            uuid = node.get('uuid')
            name = node.get('name')
            group_id = node.get('group_id')
            content = node.get('content')
            source = node.get('source')
            source_description = node.get('source_description')
            created_at = node.get('created_at')
            valid_at = node.get('valid_at')
        else:
            uuid = node.uuid
            name = node.name
            group_id = node.group_id
            content = node.content
            source = node.source.value if hasattr(node.source, 'value') else node.source
            source_description = node.source_description
            created_at = node.created_at
            valid_at = node.valid_at

        props = {
            'uuid': uuid,
            'name': name,
            'group_id': group_id,
            'content': content,
            'source': source,
            'source_description': source_description,
            'created_at': self._get_timestamp(created_at),
            'valid_at': self._get_timestamp(valid_at),
        }

        await driver.schema_manager.ensure_schema('Episodic', props)

        ngql = driver.query_builder.build_upsert_vertex('Episodic', uuid, props)
        async with driver.session() as session:
            await session.execute(ngql)

        # Write to Milvus for BM25 full-text search on episode content
        # Episodes don't need dense embeddings, but we use a placeholder for schema compatibility
        milvus_data = {
            'id': uuid,
            'embedding': [0.0] * driver.milvus.dim,  # Placeholder for required field
            'text': content or '',  # For BM25 full-text search
            'created_at': props['created_at'],
            'group_id': group_id,
        }
        await driver.milvus.upsert_vectors(
            [milvus_data], collection_name=driver.episodes_collection
        )

    async def _save_community_node(self, node: Any, driver: Any):
        if isinstance(node, dict):
            uuid = node.get('uuid')
            name = node.get('name')
            group_id = node.get('group_id')
            summary = node.get('summary')
            created_at = node.get('created_at')
            name_embedding = node.get('name_embedding')
        else:
            uuid = node.uuid
            name = node.name
            group_id = node.group_id
            summary = node.summary
            created_at = node.created_at
            name_embedding = node.name_embedding

        props = {
            'uuid': uuid,
            'name': name,
            'group_id': group_id,
            'summary': summary,
            'created_at': self._get_timestamp(created_at),
        }

        await driver.schema_manager.ensure_schema('Community', props)

        ngql = driver.query_builder.build_upsert_vertex('Community', uuid, props)
        async with driver.session() as session:
            await session.execute(ngql)

        if name_embedding:
            # Create text for BM25 search from name and summary
            text_content = f'{name or ""} {summary or ""}'
            milvus_data = {
                'id': uuid,
                'embedding': name_embedding,
                'text': text_content,  # For BM25 full-text search
                'created_at': props['created_at'],
                'group_id': group_id,
            }
            await driver.milvus.upsert_vectors(
                [milvus_data], collection_name=driver.nodes_collection
            )

    async def edge_save(self, edge: Any, driver: Any) -> None:
        """Persist (create or update) a single edge."""

        if isinstance(edge, dict):
            uuid = edge.get('uuid')
            source_uuid = edge.get('source_node_uuid') or edge.get('source_uuid')
            target_uuid = edge.get('target_node_uuid') or edge.get('target_uuid')
            fact = edge.get('fact')
            group_id = edge.get('group_id')
            created_at = edge.get('created_at')
            relation = edge.get('name', 'RELATED_TO')
            fact_embedding = edge.get('fact_embedding')
            episodes = edge.get('episodes')

            # Prepare props for Nebula
            props = edge.copy()
            # Remove fields that shouldn't be in Nebula props or need conversion
            props.pop('source_node_uuid', None)
            props.pop('source_uuid', None)
            props.pop('target_node_uuid', None)
            props.pop('target_uuid', None)
            props.pop('name', None)
            props.pop('fact_embedding', None)

            # Convert datetime to timestamp if needed
            props['created_at'] = self._get_timestamp(created_at)

            if episodes and isinstance(episodes, list):
                props['episodes'] = json.dumps(episodes)

        else:
            # Object case
            uuid = edge.uuid
            source_uuid = edge.source_node_uuid
            target_uuid = edge.target_node_uuid
            fact = edge.fact
            group_id = edge.group_id
            created_at = edge.created_at
            relation = (
                edge.relation if hasattr(edge, 'relation') and edge.relation else 'RELATED_TO'
            )
            fact_embedding = getattr(edge, 'fact_embedding', None)
            episodes = getattr(edge, 'episodes', None)

            props = {
                'uuid': uuid,
                'fact': fact,
                'group_id': group_id,
                'created_at': self._get_timestamp(created_at),
                'valid_at': self._get_timestamp(edge.valid_at) if hasattr(edge, 'valid_at') else 0.0,
            }
            if episodes:
                props['episodes'] = json.dumps(episodes)

            # Add other attributes if present
            if hasattr(edge, 'attributes') and edge.attributes:
                props.update(edge.attributes)

        # Ensure schema
        await driver.schema_manager.ensure_edge_schema(relation, props)

        # Rank
        rank = int(self._get_timestamp(created_at) * 1000)

        ngql = driver.query_builder.build_insert_edge(
            relation, source_uuid, target_uuid, rank, props
        )

        async with driver.session() as session:
            await session.execute(ngql)

        # Save embedding to Milvus
        if fact_embedding:
            # Create text for BM25 search from relation name and fact
            text_content = f'{relation or ""} {fact or ""}'
            
            # Extract date fields for filtering
            valid_at_ts = None
            invalid_at_ts = None
            expired_at_ts = None
            created_at_ts = rank / 1000.0
            
            if isinstance(edge, dict):
                valid_at = edge.get('valid_at')
                invalid_at = edge.get('invalid_at')
                expired_at = edge.get('expired_at')
            else:
                valid_at = getattr(edge, 'valid_at', None)
                invalid_at = getattr(edge, 'invalid_at', None)
                expired_at = getattr(edge, 'expired_at', None)
            
            # Convert datetimes to timestamps for Milvus
            if valid_at:
                valid_at_ts = int(self._get_timestamp(valid_at) * 1000)
            
            if invalid_at:
                invalid_at_ts = int(self._get_timestamp(invalid_at) * 1000)
            
            if expired_at:
                expired_at_ts = int(self._get_timestamp(expired_at) * 1000)
            
            milvus_data = {
                'id': uuid,
                'embedding': fact_embedding,
                'text': text_content,  # For BM25 full-text search
                'created_at': created_at_ts,
                'group_id': group_id,
                'name': relation,  # Edge type for filtering
                'uuid': uuid,  # For filtering by edge UUID
            }
            
            # Add date fields only if they exist (nullable fields)
            if valid_at_ts is not None:
                milvus_data['valid_at'] = valid_at_ts
            if invalid_at_ts is not None:
                milvus_data['invalid_at'] = invalid_at_ts
            if expired_at_ts is not None:
                milvus_data['expired_at'] = expired_at_ts
            
            await driver.milvus.upsert_vectors(
                [milvus_data], collection_name=driver.edges_collection
            )

    # Implement other methods as needed or raise NotImplementedError
    async def node_delete(self, node: Any, driver: Any) -> None:
        # DELETE VERTEX <vid>
        # Note: Nebula DELETE VERTEX deletes the vertex and all outgoing/incoming edges.
        ngql = f'DELETE VERTEX "{node.uuid}"'
        async with driver.session() as session:
            await session.execute(ngql)

        # Also delete from Milvus
        await driver.milvus.delete([node.uuid], collection_name=driver.nodes_collection)

    async def node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        """
        Bulk save nodes using batch operations for both NebulaGraph and Milvus.
        
        NebulaGraph: Uses batched INSERT/UPDATE statements
        Milvus: Uses single upsert_vectors call with all data
        """
        if not nodes:
            return
        
        # Separate nodes by type for efficient processing
        entity_nodes = []
        episodic_nodes = []
        community_nodes = []
        
        for node in nodes:
            if isinstance(node, dict):
                if 'content' in node and 'source' in node:
                    episodic_nodes.append(node)
                elif 'summary' in node and 'name' in node:
                    # Could be Entity or Community - check if it has member count
                    entity_nodes.append(node)
                else:
                    entity_nodes.append(node)
            elif isinstance(node, EpisodicNode):
                episodic_nodes.append(node)
            elif isinstance(node, CommunityNode):
                community_nodes.append(node)
            else:
                entity_nodes.append(node)
        
        # Process each type in batches
        if entity_nodes:
            await self._save_entity_nodes_batch(entity_nodes, driver, batch_size)
        if episodic_nodes:
            await self._save_episodic_nodes_batch(episodic_nodes, driver, batch_size)
        if community_nodes:
            await self._save_community_nodes_batch(community_nodes, driver, batch_size)

    async def _save_entity_nodes_batch(
        self, nodes: list[Any], driver: Any, batch_size: int = 100
    ):
        """Batch save entity nodes to both NebulaGraph and Milvus."""
        nebula_statements = []
        milvus_data = []
        
        # Collect all keys for schema check
        all_keys = {'uuid', 'name', 'group_id', 'summary', 'created_at'}
        for node in nodes:
            if isinstance(node, dict):
                attributes = node.copy()
                for k in ['uuid', 'name', 'group_id', 'summary', 'created_at', 'name_embedding', 'labels']:
                    attributes.pop(k, None)
                all_keys.update(attributes.keys())
            else:
                if hasattr(node, 'attributes') and node.attributes:
                    all_keys.update(node.attributes.keys())
        
        # Ensure schema once
        dummy_props = {k: '' for k in all_keys}
        await driver.schema_manager.ensure_schema('Entity', dummy_props)
        
        for node in nodes:
            # Extract node data
            if isinstance(node, dict):
                uuid = node.get('uuid')
                name = node.get('name')
                group_id = node.get('group_id')
                summary = node.get('summary')
                created_at = node.get('created_at')
                name_embedding = node.get('name_embedding')
                labels = node.get('labels', ['Entity'])
                attributes = node.copy()
                for k in ['uuid', 'name', 'group_id', 'summary', 'created_at', 'name_embedding', 'labels']:
                    attributes.pop(k, None)
            else:
                uuid = node.uuid
                name = node.name
                group_id = node.group_id
                summary = node.summary
                created_at = node.created_at
                name_embedding = node.name_embedding
                labels = node.labels if hasattr(node, 'labels') else ['Entity']
                attributes = node.attributes
            
            # Prepare Nebula props
            props = {
                'uuid': uuid,
                'name': name,
                'group_id': group_id,
                'summary': summary,
                'created_at': self._get_timestamp(created_at),
            }
            
            if attributes:
                props.update(attributes)
            
            # Build Nebula statement
            ngql = driver.query_builder.build_upsert_vertex('Entity', uuid, props)
            nebula_statements.append(ngql)
            
            # Prepare Milvus data
            if name_embedding:
                text_content = f'{name or ""} {summary or ""}'
                milvus_data.append({
                    'id': uuid,
                    'embedding': name_embedding,
                    'text': text_content,
                    'created_at': props['created_at'],
                    'group_id': group_id,
                    'labels': labels,
                })
        
        # Batch execute Nebula statements
        if nebula_statements:
            async with driver.session() as session:
                for i in range(0, len(nebula_statements), batch_size):
                    batch = nebula_statements[i:i + batch_size]
                    await self._execute_nebula_batch(session, batch)
        
        # Batch upsert to Milvus (Milvus handles batching internally)
        if milvus_data:
            await driver.milvus.upsert_vectors(milvus_data, collection_name=driver.nodes_collection)

    async def _save_episodic_nodes_batch(
        self, nodes: list[Any], driver: Any, batch_size: int = 100
    ):
        """Batch save episodic nodes to both NebulaGraph and Milvus."""
        nebula_statements = []
        milvus_data = []
        
        # Ensure schema once
        dummy_props = {
            'uuid': '', 'name': '', 'group_id': '', 'content': '', 
            'source': '', 'source_description': '', 'created_at': 0.0, 'valid_at': 0.0
        }
        await driver.schema_manager.ensure_schema('Episodic', dummy_props)
        
        for node in nodes:
            if isinstance(node, dict):
                uuid = node.get('uuid')
                name = node.get('name')
                group_id = node.get('group_id')
                content = node.get('content')
                source = node.get('source')
                source_description = node.get('source_description')
                created_at = node.get('created_at')
                valid_at = node.get('valid_at')
            else:
                uuid = node.uuid
                name = node.name
                group_id = node.group_id
                content = node.content
                source = node.source.value if hasattr(node.source, 'value') else node.source
                source_description = node.source_description
                created_at = node.created_at
                valid_at = node.valid_at
            
            props = {
                'uuid': uuid,
                'name': name,
                'group_id': group_id,
                'content': content,
                'source': source,
                'source_description': source_description,
                'created_at': self._get_timestamp(created_at),
                'valid_at': self._get_timestamp(valid_at),
            }
            
            ngql = driver.query_builder.build_upsert_vertex('Episodic', uuid, props)
            nebula_statements.append(ngql)
            
            milvus_data.append({
                'id': uuid,
                'embedding': [0.0] * driver.milvus.dim,
                'text': content or '',
                'created_at': props['created_at'],
                'group_id': group_id,
            })
        
        # Batch execute Nebula
        if nebula_statements:
            async with driver.session() as session:
                for i in range(0, len(nebula_statements), batch_size):
                    batch = nebula_statements[i:i + batch_size]
                    await self._execute_nebula_batch(session, batch)
        
        # Batch upsert to Milvus
        if milvus_data:
            await driver.milvus.upsert_vectors(milvus_data, collection_name=driver.episodes_collection)

    async def _save_community_nodes_batch(
        self, nodes: list[Any], driver: Any, batch_size: int = 100
    ):
        """Batch save community nodes to both NebulaGraph and Milvus."""
        nebula_statements = []
        milvus_data = []
        
        # Ensure schema once
        dummy_props = {
            'uuid': '', 'name': '', 'group_id': '', 'summary': '', 'created_at': 0.0
        }
        await driver.schema_manager.ensure_schema('Community', dummy_props)
        
        for node in nodes:
            if isinstance(node, dict):
                uuid = node.get('uuid')
                name = node.get('name')
                group_id = node.get('group_id')
                summary = node.get('summary')
                created_at = node.get('created_at')
                name_embedding = node.get('name_embedding')
            else:
                uuid = node.uuid
                name = node.name
                group_id = node.group_id
                summary = node.summary
                created_at = node.created_at
                name_embedding = node.name_embedding
            
            props = {
                'uuid': uuid,
                'name': name,
                'group_id': group_id,
                'summary': summary,
                'created_at': self._get_timestamp(created_at),
            }
            
            ngql = driver.query_builder.build_upsert_vertex('Community', uuid, props)
            nebula_statements.append(ngql)
            
            if name_embedding:
                text_content = f'{name or ""} {summary or ""}'
                milvus_data.append({
                    'id': uuid,
                    'embedding': name_embedding,
                    'text': text_content,
                    'created_at': props['created_at'],
                    'group_id': group_id,
                })
        
        # Batch execute Nebula
        if nebula_statements:
            async with driver.session() as session:
                for i in range(0, len(nebula_statements), batch_size):
                    batch = nebula_statements[i:i + batch_size]
                    await self._execute_nebula_batch(session, batch)
        
        # Batch upsert to Milvus
        if milvus_data:
            await driver.milvus.upsert_vectors(milvus_data, collection_name=driver.nodes_collection)

    async def edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100
    ) -> None:
        """
        Bulk save edges using batch operations for both NebulaGraph and Milvus.
        """
        if not edges:
            return
        
        nebula_statements = []
        milvus_data = []
        
        # Collect all keys for schema check
        edges_by_relation = {}
        for edge in edges:
            relation = edge.get('name', 'RELATED_TO') if isinstance(edge, dict) else (edge.relation if hasattr(edge, 'relation') and edge.relation else 'RELATED_TO')
            if relation not in edges_by_relation:
                edges_by_relation[relation] = []
            edges_by_relation[relation].append(edge)
            
        # Ensure schema for each relation
        for relation, rel_edges in edges_by_relation.items():
            all_keys = {'uuid', 'fact', 'group_id', 'created_at', 'valid_at', 'episodes'}
            for edge in rel_edges:
                if isinstance(edge, dict):
                    props = edge.copy()
                    for k in ['source_node_uuid', 'source_uuid', 'target_node_uuid', 'target_uuid', 'name', 'fact_embedding']:
                        props.pop(k, None)
                    all_keys.update(props.keys())
                else:
                    if hasattr(edge, 'attributes') and edge.attributes:
                        all_keys.update(edge.attributes.keys())
            
            dummy_props = {k: '' for k in all_keys}
            await driver.schema_manager.ensure_edge_schema(relation, dummy_props)
        
        for edge in edges:
            # Extract edge data
            if isinstance(edge, dict):
                uuid = edge.get('uuid')
                source_uuid = edge.get('source_node_uuid') or edge.get('source_uuid')
                target_uuid = edge.get('target_node_uuid') or edge.get('target_uuid')
                fact = edge.get('fact')
                group_id = edge.get('group_id')
                created_at = edge.get('created_at')
                relation = edge.get('name', 'RELATED_TO')
                fact_embedding = edge.get('fact_embedding')
                episodes = edge.get('episodes')
                
                props = edge.copy()
                props.pop('source_node_uuid', None)
                props.pop('source_uuid', None)
                props.pop('target_node_uuid', None)
                props.pop('target_uuid', None)
                props.pop('name', None)
                props.pop('fact_embedding', None)
                
                props['created_at'] = self._get_timestamp(created_at)
                
                if episodes and isinstance(episodes, list):
                    props['episodes'] = json.dumps(episodes)
                
                valid_at = edge.get('valid_at')
                invalid_at = edge.get('invalid_at')
                expired_at = edge.get('expired_at')
            else:
                uuid = edge.uuid
                source_uuid = edge.source_node_uuid
                target_uuid = edge.target_node_uuid
                fact = edge.fact
                group_id = edge.group_id
                created_at = edge.created_at
                relation = edge.relation if hasattr(edge, 'relation') and edge.relation else 'RELATED_TO'
                fact_embedding = getattr(edge, 'fact_embedding', None)
                episodes = getattr(edge, 'episodes', None)
                
                props = {
                    'uuid': uuid,
                    'fact': fact,
                    'group_id': group_id,
                    'created_at': self._get_timestamp(created_at),
                    'valid_at': self._get_timestamp(edge.valid_at) if hasattr(edge, 'valid_at') else 0.0,
                }
                if episodes:
                    props['episodes'] = json.dumps(episodes)
                
                if hasattr(edge, 'attributes') and edge.attributes:
                    props.update(edge.attributes)
                
                valid_at = getattr(edge, 'valid_at', None)
                invalid_at = getattr(edge, 'invalid_at', None)
                expired_at = getattr(edge, 'expired_at', None)
            
            # Calculate rank
            rank = int(props['created_at'] * 1000)
            
            # Build Nebula statement
            ngql = driver.query_builder.build_insert_edge(relation, source_uuid, target_uuid, rank, props)
            nebula_statements.append(ngql)
            
            # Prepare Milvus data
            if fact_embedding:
                text_content = f'{relation or ""} {fact or ""}'
                created_at_ts = rank / 1000.0
                
                edge_milvus_data = {
                    'id': uuid,
                    'embedding': fact_embedding,
                    'text': text_content,
                    'created_at': created_at_ts,
                    'group_id': group_id,
                    'name': relation,
                    'uuid': uuid,
                }
                
                if valid_at:
                    edge_milvus_data['valid_at'] = int(self._get_timestamp(valid_at) * 1000)
                if invalid_at:
                    edge_milvus_data['invalid_at'] = int(self._get_timestamp(invalid_at) * 1000)
                if expired_at:
                    edge_milvus_data['expired_at'] = int(self._get_timestamp(expired_at) * 1000)
                
                milvus_data.append(edge_milvus_data)
        
        # Batch execute Nebula statements
        if nebula_statements:
            async with driver.session() as session:
                for i in range(0, len(nebula_statements), batch_size):
                    batch = nebula_statements[i:i + batch_size]
                    await self._execute_nebula_batch(session, batch)
        
        # Batch upsert to Milvus
        if milvus_data:
            await driver.milvus.upsert_vectors(milvus_data, collection_name=driver.edges_collection)

    # ... other methods ...
    async def node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100
    ) -> None:
        # This is hard in Nebula without an index on group_id.
        # Assuming we have an index on group_id for Entity/Episodic/Community tags.
        # LOOKUP ON Entity WHERE Entity.group_id == "..." YIELD id(vertex) | DELETE VERTEX $-.VertexID
        # For now, raise NotImplementedError or implement if index exists.
        raise NotImplementedError('Bulk delete by group_id not implemented yet')

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        # DELETE VERTEX "u1", "u2", ...
        ids_str = ', '.join([f'"{uid}"' for uid in uuids])
        ngql = f'DELETE VERTEX {ids_str}'
        async with driver.session() as session:
            await session.execute(ngql)

        await driver.milvus.delete(uuids, collection_name=driver.nodes_collection)

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        await self._save_episodic_node(node, driver)

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        await self.node_delete(node, driver)

    async def episodic_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        await self.node_save_bulk(_cls, driver, transaction, nodes, batch_size)

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        # Episodic edges might be edges between Episode and Entity?
        # Assuming they are standard edges.
        await self.edge_save_bulk(_cls, driver, transaction, episodic_edges, batch_size)

    async def episodic_node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100
    ) -> None:
        await self.node_delete_by_group_id(_cls, driver, group_id, batch_size)

    async def episodic_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await self.node_delete_by_uuids(_cls, driver, uuids, group_id, batch_size)

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        # DELETE EDGE <edge_type> <src> -> <dst>@<rank>
        rank = int(self._get_timestamp(edge.created_at) * 1000) if hasattr(edge, 'created_at') else 0
        edge_type = edge.relation if edge.relation else 'RELATED_TO'
        ngql = f'DELETE EDGE `{edge_type}` "{edge.source_node_uuid}" -> "{edge.target_node_uuid}"@{rank}'
        async with driver.session() as session:
            await session.execute(ngql)

    async def edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> None:
        # We need src, dst, rank to delete edge in Nebula. UUID is not enough unless we index it.
        # Graphiti edges have UUIDs.
        # If we can't look up by UUID, we can't delete by UUID easily.
        # We might need to LOOKUP ON <edge_type> WHERE <edge_type>.uuid == "..."
        raise NotImplementedError('Delete edge by UUID not implemented yet')

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        # Load from Milvus
        results = await driver.milvus.get_vectors(
            [node.uuid], collection_name=driver.nodes_collection
        )
        if results and 'embedding' in results[0]:
            node.name_embedding = results[0]['embedding']

    async def node_load_embeddings_bulk(
        self, driver: Any, nodes: list[Any], batch_size: int = 100
    ) -> dict[str, list[float]]:
        uuids = [node.uuid for node in nodes]
        if not uuids:
            return {}

        results = await driver.milvus.get_vectors(uuids, collection_name=driver.nodes_collection)

        embedding_map = {}
        for res in results:
            if 'embedding' in res:
                embedding_map[res['id']] = res['embedding']

        return embedding_map

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        results = await driver.milvus.get_vectors(
            [edge.uuid], collection_name=driver.edges_collection
        )
        if results and 'embedding' in results[0]:
            edge.fact_embedding = results[0]['embedding']

    async def edge_load_embeddings_bulk(
        self, driver: Any, edges: list[Any], batch_size: int = 100
    ) -> dict[str, list[float]]:
        uuids = [edge.uuid for edge in edges]
        if not uuids:
            return {}

        results = await driver.milvus.get_vectors(uuids, collection_name=driver.edges_collection)

        embedding_map = {}
        for res in results:
            if 'embedding' in res:
                embedding_map[res['id']] = res['embedding']

        return embedding_map

    def _get_timestamp(self, value: Any) -> float:
        if hasattr(value, 'timestamp'):
            return float(value.timestamp())
        elif isinstance(value, (int, float)):
            return float(value)
        return 0.0

    async def _execute_nebula_batch(self, session: Any, statements: list[str]) -> None:
        if not statements:
            return
        # Execute statements concurrently
        await asyncio.gather(*(session.execute(stmt) for stmt in statements))
