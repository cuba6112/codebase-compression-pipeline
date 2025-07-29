"""
Codebase Map Generator
======================

Generate comprehensive codebase structure and relationships.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

from base_classes import FileMetadata

logger = logging.getLogger(__name__)


class CodebaseMapGenerator:
    """Generate comprehensive codebase structure and relationships"""
    
    def __init__(self):
        self.file_tree = {}
        self.dependency_graph = {}
        self.statistics = {}
    
    def generate_directory_tree(self, file_paths: List[Path], base_path: Path) -> str:
        """Generate a visual directory tree representation"""
        tree_dict = {}
        
        # Build tree structure
        for file_path in sorted(file_paths):
            try:
                relative_path = file_path.relative_to(base_path)
                parts = relative_path.parts
                current = tree_dict
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Add file
                current[parts[-1]] = None
            except ValueError:
                logger.warning(f"File {file_path} is not relative to {base_path}")
                continue
        
        # Convert to string representation
        def build_tree_string(node, prefix="", is_last=True):
            lines = []
            items = sorted(node.items()) if node else []
            
            for i, (name, subtree) in enumerate(items):
                is_last_item = i == len(items) - 1
                
                # Add current item
                connector = "└── " if is_last_item else "├── "
                lines.append(prefix + connector + name)
                
                # Add children
                if subtree is not None:  # It's a directory
                    extension = "    " if is_last_item else "│   "
                    lines.extend(build_tree_string(subtree, prefix + extension, is_last_item))
            
            return lines
        
        tree_lines = [str(base_path.name) + "/"]
        tree_lines.extend(build_tree_string(tree_dict))
        return "\n".join(tree_lines)
    
    def generate_dependency_graph(self, metadata_list: List[FileMetadata]) -> Dict[str, Any]:
        """Generate import/export relationships between files"""
        graph = {
            "nodes": [],
            "edges": [],
            "clusters": {}
        }
        
        # Create nodes
        for metadata in metadata_list:
            node = {
                "id": metadata.path,
                "label": Path(metadata.path).name,
                "type": metadata.file_type,
                "size": metadata.size,
                "complexity": metadata.complexity_score
            }
            graph["nodes"].append(node)
        
        # Create edges based on imports
        path_to_metadata = {m.path: m for m in metadata_list}
        
        for metadata in metadata_list:
            for dep in metadata.internal_dependencies:
                # Find matching file
                for other_path, other_metadata in path_to_metadata.items():
                    if dep in other_path or Path(other_path).stem == dep:
                        edge = {
                            "source": metadata.path,
                            "target": other_path,
                            "type": "import"
                        }
                        graph["edges"].append(edge)
        
        # Identify clusters (modules)
        modules = {}
        for metadata in metadata_list:
            module_parts = metadata.module_path.split('.')[:-1]
            if module_parts:
                module = '.'.join(module_parts)
                if module not in modules:
                    modules[module] = []
                modules[module].append(metadata.path)
        
        graph["clusters"] = modules
        return graph
    
    def generate_codebase_statistics(self, metadata_list: List[FileMetadata]) -> Dict[str, Any]:
        """Generate comprehensive statistics about the codebase"""
        stats = {
            "total_files": len(metadata_list),
            "total_size": sum(m.size for m in metadata_list),
            "total_lines": sum(m.token_count for m in metadata_list),  # Approximate
            "languages": {},
            "file_types": {},
            "complexity": {
                "average": 0,
                "max": 0,
                "distribution": {}
            },
            "dependencies": {
                "internal": set(),
                "external": set()
            },
            "top_imports": {},
            "largest_files": [],
            "most_complex_files": []
        }
        
        # Language distribution
        for metadata in metadata_list:
            lang = metadata.language
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            file_type = metadata.file_type
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
        
        # Complexity analysis
        complexities = [m.complexity_score for m in metadata_list if m.complexity_score > 0]
        if complexities:
            stats["complexity"]["average"] = sum(complexities) / len(complexities)
            stats["complexity"]["max"] = max(complexities)
            
            # Distribution
            ranges = [(0, 5), (5, 10), (10, 20), (20, 50), (50, float('inf'))]
            for low, high in ranges:
                key = f"{low}-{high}" if high != float('inf') else f"{low}+"
                count = sum(1 for c in complexities if low <= c < high)
                stats["complexity"]["distribution"][key] = count
        
        # Dependencies
        for metadata in metadata_list:
            stats["dependencies"]["internal"].update(metadata.internal_dependencies)
            stats["dependencies"]["external"].update(metadata.external_dependencies)
        
        stats["dependencies"]["internal"] = len(stats["dependencies"]["internal"])
        stats["dependencies"]["external"] = len(stats["dependencies"]["external"])
        
        # Top imports
        import_counts = {}
        for metadata in metadata_list:
            for imp in metadata.imports:
                import_counts[imp] = import_counts.get(imp, 0) + 1
        
        stats["top_imports"] = dict(sorted(import_counts.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:10])
        
        # Largest and most complex files
        stats["largest_files"] = sorted(metadata_list, 
                                       key=lambda m: m.size, 
                                       reverse=True)[:5]
        stats["most_complex_files"] = sorted(metadata_list, 
                                           key=lambda m: m.complexity_score, 
                                           reverse=True)[:5]
        
        return stats
    
    def create_import_export_matrix(self, metadata_list: List[FileMetadata]) -> Dict[str, Any]:
        """Create a matrix showing import/export relationships"""
        files = [m.path for m in metadata_list]
        matrix = {f: {t: False for t in files} for f in files}
        
        # Build relationships
        for metadata in metadata_list:
            for dep in metadata.internal_dependencies:
                for other in metadata_list:
                    if dep in other.path or Path(other.path).stem == dep:
                        matrix[metadata.path][other.path] = True
        
        return {
            "files": files,
            "matrix": matrix,
            "summary": {
                "total_dependencies": sum(sum(row.values()) for row in matrix.values()),
                "most_imported": [],
                "most_dependent": []
            }
        }