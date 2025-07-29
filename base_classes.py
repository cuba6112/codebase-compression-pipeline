"""
Base Classes for Codebase Compression Pipeline
=============================================

Contains core data structures and abstract base classes used throughout the pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional


@dataclass
class FileMetadata:
    """Metadata extracted from source files"""
    path: str
    size: int
    language: str
    last_modified: float
    content_hash: str
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    complexity_score: float = 0.0
    token_count: int = 0
    # New fields for enhanced mapping
    relative_path: str = ""  # Path relative to project root
    module_path: str = ""  # Python module notation (e.g., utils.helper)
    file_type: str = ""  # source, config, doc, test, etc.
    external_dependencies: List[str] = field(default_factory=list)  # Third-party imports
    internal_dependencies: List[str] = field(default_factory=list)  # Project imports
    
    # Optional fields
    encoding: str = 'utf-8'
    
    # TypeScript/JavaScript specific features (optional)
    typescript_features: Optional[Dict[str, Any]] = field(default_factory=dict)


class LanguageParser(ABC):
    """Abstract base class for language-specific parsers"""
    
    @abstractmethod
    def parse(self, content: str, filepath: str) -> FileMetadata:
        pass
    
    @abstractmethod
    def tokenize(self, content: str) -> List[str]:
        pass
    
    @abstractmethod
    def extract_structure(self, content: str) -> Dict[str, Any]:
        pass