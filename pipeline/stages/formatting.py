"""
Output Formatter
================

Format compressed output for LLM consumption.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set, Optional

logger = logging.getLogger(__name__)


class OutputFormatter:
    """Format compressed output for LLM consumption"""
    
    def __init__(self, max_context_size: int = 128000):
        self.max_context_size = max_context_size
        self.format_templates = {
            'markdown': self._format_markdown,
            'json': self._format_json,
            'xml': self._format_xml,
            'custom': self._format_custom
        }
    
    def format_output(self, 
                     compressed_data: List[Dict[str, Any]], 
                     format_type: str = 'markdown',
                     chunk_strategy: str = 'semantic',
                     codebase_map: Optional[Dict[str, Any]] = None) -> List[str]:
        """Format compressed data into LLM-ready chunks"""
        
        # Apply formatting
        format_func = self.format_templates.get(format_type, self._format_markdown)
        formatted_items = []
        
        # Add codebase map as first item if provided
        if codebase_map and format_type == 'markdown':
            formatted_items.append(self._format_codebase_map(codebase_map))
        
        # Format each compressed item
        formatted_items.extend([format_func(item) for item in compressed_data])
        
        # Apply chunking strategy
        if chunk_strategy == 'semantic':
            return self._semantic_chunking(formatted_items, compressed_data)
        elif chunk_strategy == 'size':
            return self._size_based_chunking(formatted_items)
        else:
            return self._balanced_chunking(formatted_items, compressed_data)
    
    def _format_codebase_map(self, codebase_map: Dict[str, Any]) -> str:
        """Format codebase map information as markdown"""
        output = []
        
        # Add title
        output.append("# Codebase Analysis\n")
        
        # Add statistics if available
        if 'statistics' in codebase_map:
            stats = codebase_map['statistics']
            output.append("## 📊 Overview\n")
            output.append(f"- **Total Files**: {stats['total_files']}")
            output.append(f"- **Total Size**: {stats['total_size']:,} bytes")
            output.append(f"- **Total Lines**: {stats['total_lines']:,}")
            
            # Language distribution
            if stats['languages']:
                output.append(f"- **Languages**: {', '.join(f'{lang} ({count})' for lang, count in stats['languages'].items())}")
            
            # Complexity
            if stats['complexity']['average'] > 0:
                output.append(f"- **Average Complexity**: {stats['complexity']['average']:.1f}")
            
            output.append("")
        
        # Add directory tree
        if 'directory_tree' in codebase_map:
            output.append("## 🗺️ Project Structure\n")
            output.append("```")
            output.append(codebase_map['directory_tree'])
            output.append("```\n")
        
        # Add dependency graph summary
        if 'dependency_graph' in codebase_map:
            graph = codebase_map['dependency_graph']
            output.append("## 🔗 Dependency Graph\n")
            
            # Show import relationships
            if graph['edges']:
                output.append("### Import Relationships")
                # Group by source
                imports_by_source = {}
                for edge in graph['edges']:
                    source = edge['source']
                    target = edge['target']
                    if source not in imports_by_source:
                        imports_by_source[source] = []
                    imports_by_source[source].append(target)
                
                # Show top importers
                for source, targets in sorted(imports_by_source.items())[:10]:
                    output.append(f"- {Path(source).name} → {', '.join(Path(t).name for t in targets[:3])}")
                    if len(targets) > 3:
                        output.append(f"  (and {len(targets) - 3} more)")
            
            output.append("")
        
        # Add external dependencies
        if 'statistics' in codebase_map and codebase_map['statistics']['dependencies']['external'] > 0:
            output.append("## 📦 External Dependencies\n")
            top_imports = codebase_map['statistics']['top_imports']
            if top_imports:
                output.append("### Most Used Imports")
                for imp, count in list(top_imports.items())[:10]:
                    output.append(f"- {imp} ({count} files)")
            output.append("")
        
        # Add file organization
        output.append("## 📁 File Organization\n")
        
        return '\n'.join(output)
    
    def _format_markdown(self, item: Dict[str, Any]) -> str:
        """Format as markdown"""
        output = [f"## {item['path']}\n"]
        
        if item['type'] == 'full':
            output.append("```")
            output.append(item['content'])
            output.append("```\n")
        
        elif item['type'] == 'structural':
            structure = item['structure']
            
            if structure['imports']:
                output.append("### Imports")
                for imp in structure['imports']:
                    output.append(f"- {imp}")
                output.append("")
            
            if structure['functions']:
                output.append("### Functions")
                for func in structure['functions']:
                    output.append(f"- {func}")
                output.append("")
            
            if structure['classes']:
                output.append("### Classes")
                for cls in structure['classes']:
                    output.append(f"- {cls}")
                output.append("")
        
        elif item['type'] == 'signature':
            output.append("### Signatures")
            for sig in item['signatures'].get('functions', []):
                output.append(f"- {sig}")
            for sig in item['signatures'].get('classes', []):
                output.append(f"- {sig}")
            output.append("")
        
        elif item['type'] == 'summary':
            summary = item['summary']
            output.append(f"- Language: {summary['language']}")
            output.append(f"- Size: {summary['size']} bytes")
            output.append(f"- Complexity: {summary['complexity']:.2f}")
            output.append(f"- Functions: {summary['functions_count']}")
            output.append(f"- Classes: {summary['classes_count']}")
            output.append("")
        
        return '\n'.join(output)
    
    def _format_json(self, item: Dict[str, Any]) -> str:
        """Format as JSON"""
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(v) for v in obj]
            return obj
        
        return json.dumps(convert_sets(item), indent=2)
    
    def _format_xml(self, item: Dict[str, Any]) -> str:
        """Format as XML"""
        def dict_to_xml(tag: str, d: Any) -> str:
            if isinstance(d, dict):
                xml = f"<{tag}>"
                for key, value in d.items():
                    xml += dict_to_xml(key, value)
                xml += f"</{tag}>"
                return xml
            elif isinstance(d, list):
                xml = ""
                for item in d:
                    xml += dict_to_xml(tag, item)
                return xml
            else:
                return f"<{tag}>{d}</{tag}>"
        
        return dict_to_xml('file', item)
    
    def _format_custom(self, item: Dict[str, Any]) -> str:
        """Custom format optimized for LLM understanding"""
        output = []
        
        # Header with metadata
        output.append(f"[FILE: {item['path']}]")
        output.append(f"[TYPE: {item['type'].upper()}]")
        
        if 'metadata' in item:
            meta = item['metadata']
            output.append(f"[LANG: {meta['language']} | SIZE: {meta['size']} | COMPLEXITY: {meta['complexity']:.1f}]")
        
        output.append("")
        
        # Content based on type
        if item['type'] == 'structural':
            structure = item['structure']
            
            if structure['imports']:
                output.append("[IMPORTS]")
                output.extend(structure['imports'])
                output.append("")
            
            if structure['functions']:
                output.append("[FUNCTIONS]")
                output.extend(structure['functions'])
                output.append("")
            
            if structure['classes']:
                output.append("[CLASSES]")
                output.extend(structure['classes'])
                output.append("")
        
        return '\n'.join(output)
    
    def _semantic_chunking(self, 
                          formatted_items: List[str], 
                          compressed_data: List[Dict[str, Any]]) -> List[str]:
        """Chunk based on semantic relationships"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Group by import/export relationships
        dependency_graph = self._build_dependency_graph(compressed_data)
        clusters = self._find_clusters(dependency_graph)
        
        # If no clusters found, create individual clusters for each item
        if not clusters:
            clusters = [[i] for i in range(len(formatted_items))]
        
        for cluster in clusters:
            cluster_items = [formatted_items[i] for i in cluster if i < len(formatted_items)]
            cluster_size = sum(len(item) for item in cluster_items)
            
            if current_size + cluster_size > self.max_context_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.extend(cluster_items)
            current_size += cluster_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _size_based_chunking(self, formatted_items: List[str]) -> List[str]:
        """Simple size-based chunking"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in formatted_items:
            item_size = len(item)
            
            if current_size + item_size > self.max_context_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(item)
            current_size += item_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _balanced_chunking(self, 
                          formatted_items: List[str], 
                          compressed_data: List[Dict[str, Any]]) -> List[str]:
        """Balance chunk sizes while respecting relationships"""
        # Sort by complexity for better distribution
        items_with_complexity = [
            (i, item, compressed_data[i].get('metadata', {}).get('complexity', 0))
            for i, item in enumerate(formatted_items)
        ]
        items_with_complexity.sort(key=lambda x: x[2], reverse=True)
        
        # Distribute into balanced chunks
        num_chunks = max(1, sum(len(item[1]) for item in items_with_complexity) // self.max_context_size + 1)
        chunks = [[] for _ in range(num_chunks)]
        chunk_sizes = [0] * num_chunks
        
        for idx, item, _ in items_with_complexity:
            # Add to smallest chunk
            min_chunk_idx = chunk_sizes.index(min(chunk_sizes))
            chunks[min_chunk_idx].append(item)
            chunk_sizes[min_chunk_idx] += len(item)
        
        return ['\n'.join(chunk) for chunk in chunks if chunk]
    
    def _build_dependency_graph(self, compressed_data: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
        """Build dependency graph from import/export relationships"""
        graph = defaultdict(set)
        
        # Build export index
        export_to_file = {}
        for i, item in enumerate(compressed_data):
            if 'metadata' in item:
                for export in item['metadata'].get('exports', []):
                    export_to_file[export] = i
        
        # Build import relationships
        for i, item in enumerate(compressed_data):
            if 'metadata' in item:
                for import_name in item['metadata'].get('imports', []):
                    if import_name in export_to_file:
                        graph[i].add(export_to_file[import_name])
                        graph[export_to_file[import_name]].add(i)
        
        return graph
    
    def _find_clusters(self, graph: Dict[int, Set[int]]) -> List[List[int]]:
        """Find connected components in dependency graph"""
        visited = set()
        clusters = []
        
        def dfs(node: int, cluster: List[int]):
            if node in visited:
                return
            visited.add(node)
            cluster.append(node)
            for neighbor in graph.get(node, []):
                dfs(neighbor, cluster)
        
        # Process nodes that are in the graph
        for node in graph:
            if node not in visited:
                cluster = []
                dfs(node, cluster)
                if cluster:  # Only add non-empty clusters
                    clusters.append(cluster)
        
        return clusters