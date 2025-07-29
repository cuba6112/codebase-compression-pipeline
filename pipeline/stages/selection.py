"""
Selective Compression
=====================

Intelligent compression based on query patterns and importance.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any

from .metadata import MetadataStore

logger = logging.getLogger(__name__)


class SelectiveCompressor:
    """Intelligent compression based on query patterns and importance"""
    
    def __init__(self, metadata_store: MetadataStore):
        self.metadata_store = metadata_store
        self.compression_strategies = {
            'full': self._compress_full,
            'structural': self._compress_structural,
            'signature': self._compress_signature,
            'summary': self._compress_summary
        }
    
    def compress_by_query(self, 
                         query: Dict[str, Any], 
                         strategy: str = 'structural') -> List[Dict[str, Any]]:
        """Compress files matching query using specified strategy"""
        
        # Find matching files
        indices = self.metadata_store.query(**query)
        metadata_list = self.metadata_store.get_metadata_by_indices(indices)
        
        # Apply compression strategy
        compress_func = self.compression_strategies.get(strategy, self._compress_structural)
        compressed_results = []
        
        for metadata in metadata_list:
            compressed = compress_func(metadata)
            compressed_results.append(compressed)
        
        return compressed_results
    
    def _compress_full(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Full content with minimal compression"""
        return {
            'path': metadata['path'],
            'type': 'full',
            'content': self._load_and_minify(metadata['path']),
            'metadata': metadata
        }
    
    def _compress_structural(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve structure, compress implementation"""
        content = self._load_file(metadata['path'])
        
        # Extract and compress structure
        structure = {
            'imports': metadata['imports'],
            'exports': metadata['exports'],
            'functions': self._extract_function_signatures(content, metadata),
            'classes': self._extract_class_signatures(content, metadata),
            'constants': self._extract_constants(content)
        }
        
        return {
            'path': metadata['path'],
            'type': 'structural',
            'structure': structure,
            'metadata': metadata
        }
    
    def _compress_signature(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Only function/class signatures"""
        content = self._load_file(metadata['path'])
        
        signatures = {
            'functions': self._extract_function_signatures(content, metadata),
            'classes': self._extract_class_signatures(content, metadata)
        }
        
        return {
            'path': metadata['path'],
            'type': 'signature',
            'signatures': signatures,
            'metadata': metadata
        }
    
    def _compress_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """High-level summary only"""
        return {
            'path': metadata['path'],
            'type': 'summary',
            'summary': {
                'language': metadata['language'],
                'size': metadata['size'],
                'complexity': metadata['complexity'],
                'imports_count': len(metadata['imports']),
                'exports_count': len(metadata['exports']),
                'functions_count': len(metadata['functions']),
                'classes_count': len(metadata['classes'])
            }
        }
    
    def _load_file(self, path: str) -> str:
        """Load file content safely"""
        try:
            # Validate path
            file_path = Path(os.path.abspath(path))
            # Optional: Add path validation here if needed
            
            return file_path.read_text(encoding='utf-8', errors='ignore')
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return ""
        except PermissionError:
            logger.error(f"Permission denied reading file: {path}")
            return ""
        except Exception as e:
            logger.error(f"Error loading file {path}: {e}")
            return ""
    
    def _load_and_minify(self, path: str) -> str:
        """Load and minify file content"""
        content = self._load_file(path)
        
        # Remove comments and extra whitespace
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _extract_function_signatures(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract function signatures based on language"""
        signatures = []
        
        if metadata['language'] == 'python':
            # Extract Python function signatures
            pattern = r'def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^:]+)?:'
            for match in re.finditer(pattern, content):
                signatures.append(match.group(0).rstrip(':'))
        
        elif metadata['language'] == 'javascript':
            # Extract JavaScript function signatures
            patterns = [
                r'function\s+(\w+)\s*\([^)]*\)',
                r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+)\s*=>'
            ]
            for pattern in patterns:
                signatures.extend(re.findall(pattern, content))
        
        return signatures
    
    def _extract_class_signatures(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract class signatures based on language"""
        signatures = []
        
        if metadata['language'] == 'python':
            pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
            for match in re.finditer(pattern, content):
                signatures.append(match.group(0).rstrip(':'))
        
        elif metadata['language'] == 'javascript':
            pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
            signatures.extend(re.findall(pattern, content))
        
        return signatures
    
    def _extract_constants(self, content: str) -> List[str]:
        """Extract constant definitions"""
        constants = []
        
        # Common constant patterns
        patterns = [
            r'(?:const|final|static\s+final)\s+\w+\s*=',
            r'[A-Z_]+\s*=\s*[\'"\d]'  # UPPER_CASE constants
        ]
        
        for pattern in patterns:
            constants.extend(re.findall(pattern, content))
        
        return constants[:10]  # Limit to top 10