"""
Columnar storage for efficient metadata queries and indexing.
"""

import pickle
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set, Optional

from base_classes import FileMetadata

logger = logging.getLogger(__name__)


class MetadataStore:
    """
    Columnar storage for efficient metadata queries.
    
    This store uses columnar storage and inverted indices for fast lookups
    of file metadata based on various criteria like imports, exports, 
    functions, and classes.
    """
    
    def __init__(self, store_path: Path):
        """
        Initialize metadata store with the given storage path.
        
        Args:
            store_path: Directory path for storing metadata
        """
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Columnar storage for different metadata types
        self.columns = {
            'paths': [],
            'sizes': [],
            'languages': [],
            'complexities': [],
            'token_counts': [],
            'imports': defaultdict(list),
            'exports': defaultdict(list),
            'functions': defaultdict(list),
            'classes': defaultdict(list)
        }
        
        # Inverted indices for fast lookups
        self.import_index: Dict[str, Set[int]] = defaultdict(set)
        self.export_index: Dict[str, Set[int]] = defaultdict(set)
        self.function_index: Dict[str, Set[int]] = defaultdict(set)
        self.class_index: Dict[str, Set[int]] = defaultdict(set)
        
        # Track total entries for statistics
        self._total_entries = 0
        
        logger.info(f"Initialized MetadataStore at {store_path}")
    
    def add_metadata(self, metadata: FileMetadata) -> int:
        """
        Add metadata to columnar store.
        
        Args:
            metadata: FileMetadata object to store
            
        Returns:
            Index of the stored metadata
        """
        idx = len(self.columns['paths'])
        
        # Add to columns
        self.columns['paths'].append(metadata.path)
        self.columns['sizes'].append(metadata.size)
        self.columns['languages'].append(metadata.language)
        self.columns['complexities'].append(metadata.complexity_score)
        self.columns['token_counts'].append(metadata.token_count)
        
        # Add to relational columns and update indices
        for imp in metadata.imports:
            self.columns['imports'][idx].append(imp)
            self.import_index[imp].add(idx)
        
        for exp in metadata.exports:
            self.columns['exports'][idx].append(exp)
            self.export_index[exp].add(idx)
        
        for func in metadata.functions:
            func_name = func['name'] if isinstance(func, dict) else str(func)
            self.columns['functions'][idx].append(func_name)
            self.function_index[func_name].add(idx)
        
        for cls in metadata.classes:
            cls_name = cls['name'] if isinstance(cls, dict) else str(cls)
            self.columns['classes'][idx].append(cls_name)
            self.class_index[cls_name].add(idx)
        
        self._total_entries += 1
        return idx
    
    def query(self, **criteria) -> List[int]:
        """
        Query metadata with multiple criteria.
        
        Supported criteria:
        - language: Exact language match
        - imports: Files importing specific module
        - exports: Files exporting specific symbol
        - function: Files containing specific function
        - class: Files containing specific class
        - min_complexity: Minimum complexity score
        - max_size: Maximum file size
        - min_tokens: Minimum token count
        
        Args:
            **criteria: Query criteria as keyword arguments
            
        Returns:
            List of indices matching all criteria
        """
        result_indices: Optional[Set[int]] = None
        
        # Language filter
        if 'language' in criteria:
            lang_indices = {i for i, lang in enumerate(self.columns['languages']) 
                           if lang == criteria['language']}
            result_indices = self._intersect_indices(result_indices, lang_indices)
        
        # Import filter
        if 'imports' in criteria:
            import_indices = self.import_index.get(criteria['imports'], set())
            result_indices = self._intersect_indices(result_indices, import_indices)
        
        # Export filter
        if 'exports' in criteria:
            export_indices = self.export_index.get(criteria['exports'], set())
            result_indices = self._intersect_indices(result_indices, export_indices)
        
        # Function filter
        if 'function' in criteria:
            function_indices = self.function_index.get(criteria['function'], set())
            result_indices = self._intersect_indices(result_indices, function_indices)
        
        # Class filter
        if 'class' in criteria:
            class_indices = self.class_index.get(criteria['class'], set())
            result_indices = self._intersect_indices(result_indices, class_indices)
        
        # Complexity filter
        if 'min_complexity' in criteria:
            complex_indices = {i for i, comp in enumerate(self.columns['complexities']) 
                              if comp >= criteria['min_complexity']}
            result_indices = self._intersect_indices(result_indices, complex_indices)
        
        # Size filter
        if 'max_size' in criteria:
            size_indices = {i for i, size in enumerate(self.columns['sizes']) 
                           if size <= criteria['max_size']}
            result_indices = self._intersect_indices(result_indices, size_indices)
        
        # Token count filter
        if 'min_tokens' in criteria:
            token_indices = {i for i, tokens in enumerate(self.columns['token_counts']) 
                            if tokens >= criteria['min_tokens']}
            result_indices = self._intersect_indices(result_indices, token_indices)
        
        return sorted(list(result_indices)) if result_indices else []
    
    def _intersect_indices(self, 
                          current: Optional[Set[int]], 
                          new: Set[int]) -> Set[int]:
        """Helper method to intersect index sets."""
        return new if current is None else current & new
    
    def get_metadata_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve metadata for given indices.
        
        Args:
            indices: List of indices to retrieve
            
        Returns:
            List of metadata dictionaries
        """
        results = []
        
        for idx in indices:
            if 0 <= idx < len(self.columns['paths']):
                results.append({
                    'path': self.columns['paths'][idx],
                    'size': self.columns['sizes'][idx],
                    'language': self.columns['languages'][idx],
                    'complexity': self.columns['complexities'][idx],
                    'token_count': self.columns['token_counts'][idx],
                    'imports': self.columns['imports'].get(idx, []),
                    'exports': self.columns['exports'].get(idx, []),
                    'functions': self.columns['functions'].get(idx, []),
                    'classes': self.columns['classes'].get(idx, [])
                })
            else:
                logger.warning(f"Invalid index {idx} (max: {len(self.columns['paths']) - 1})")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the metadata store.
        
        Returns:
            Dictionary with various statistics
        """
        if not self.columns['paths']:
            return {'total_files': 0}
            
        return {
            'total_files': len(self.columns['paths']),
            'total_size': sum(self.columns['sizes']),
            'languages': dict(defaultdict(int, {
                lang: self.columns['languages'].count(lang) 
                for lang in set(self.columns['languages'])
            })),
            'avg_complexity': sum(self.columns['complexities']) / len(self.columns['complexities']),
            'avg_tokens': sum(self.columns['token_counts']) / len(self.columns['token_counts']),
            'unique_imports': len(self.import_index),
            'unique_exports': len(self.export_index),
            'unique_functions': len(self.function_index),
            'unique_classes': len(self.class_index)
        }
    
    def save(self) -> bool:
        """
        Persist metadata store to disk safely.
        
        Returns:
            True if saved successfully, False otherwise
        """
        metadata_file = self.store_path / 'metadata.pkl'
        temp_file = metadata_file.with_suffix('.tmp')
        
        try:
            # Prepare data for serialization
            data = {
                'columns': dict(self.columns),
                'indices': {
                    'import': {k: list(v) for k, v in self.import_index.items()},
                    'export': {k: list(v) for k, v in self.export_index.items()},
                    'function': {k: list(v) for k, v in self.function_index.items()},
                    'class': {k: list(v) for k, v in self.class_index.items()}
                }
            }
            
            # Write to temporary file
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_file.replace(metadata_file)
            logger.info(f"Saved metadata store with {len(self.columns['paths'])} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save metadata store: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False
    
    def load(self) -> bool:
        """
        Load metadata store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        store_file = self.store_path / 'metadata.pkl'
        
        if not store_file.exists():
            logger.info("No existing metadata store found")
            return False
            
        try:
            with open(store_file, 'rb') as f:
                data = pickle.load(f)
                
            # Restore columns
            self.columns = defaultdict(list, data['columns'])
            
            # Restore indices (convert lists back to sets)
            self.import_index = defaultdict(set, {
                k: set(v) for k, v in data['indices']['import'].items()
            })
            self.export_index = defaultdict(set, {
                k: set(v) for k, v in data['indices']['export'].items()
            })
            self.function_index = defaultdict(set, {
                k: set(v) for k, v in data['indices']['function'].items()
            })
            self.class_index = defaultdict(set, {
                k: set(v) for k, v in data['indices']['class'].items()
            })
            
            self._total_entries = len(self.columns['paths'])
            logger.info(f"Loaded metadata store with {self._total_entries} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load metadata store: {e}")
            return False
    
    def clear(self):
        """Clear all metadata from the store."""
        self.columns.clear()
        self.columns = {
            'paths': [],
            'sizes': [],
            'languages': [],
            'complexities': [],
            'token_counts': [],
            'imports': defaultdict(list),
            'exports': defaultdict(list),
            'functions': defaultdict(list),
            'classes': defaultdict(list)
        }
        
        self.import_index.clear()
        self.export_index.clear()
        self.function_index.clear()
        self.class_index.clear()
        
        self._total_entries = 0
        logger.info("Cleared metadata store")