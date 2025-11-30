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
from typing import Any, Dict, Tuple


class QueryBuilder:
    @staticmethod
    def build_upsert_vertex(tag: str, vid: str, props: Dict[str, Any]) -> str:
        """
        Builds an UPSERT VERTEX statement.
        Note: Nebula's UPSERT syntax is:
        UPSERT VERTEX ON <tag> <vid> SET <prop> = <value>, ...
        """
        set_clauses = []
        for key, value in props.items():
            val_str = QueryBuilder._format_value(value)
            set_clauses.append(f'`{key}` = {val_str}')

        set_str = ', '.join(set_clauses)
        # VID must be quoted string for STRING VID
        return f'UPSERT VERTEX ON `{tag}` "{vid}" SET {set_str}'

    @staticmethod
    def build_insert_edge(
        edge_type: str, src_vid: str, dst_vid: str, rank: int, props: Dict[str, Any]
    ) -> str:
        """
        Builds an INSERT EDGE statement.
        INSERT EDGE <edge_type> (prop1, prop2, ...) VALUES <src> -> <dst>@<rank>: (val1, val2, ...)
        """
        keys = list(props.keys())
        values = [QueryBuilder._format_value(props[k]) for k in keys]

        keys_str = ', '.join([f'`{k}`' for k in keys])
        values_str = ', '.join(values)

        return f'INSERT EDGE `{edge_type}` ({keys_str}) VALUES "{src_vid}" -> "{dst_vid}"@{rank}: ({values_str})'

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape quotes
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(value, (list, dict)):
            # Serialize to JSON string
            json_str = json.dumps(value).replace('"', '\\"')
            return f'"{json_str}"'
        elif value is None:
            return 'NULL'
        else:
            return f'"{str(value)}"'
