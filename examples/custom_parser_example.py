"""
Example Custom Parser Plugin
============================

Demonstrates how to create a custom parser that can be registered
with the pluggable parser system either via entry points or at runtime.
"""

import re
import logging
from typing import List, Dict, Any
from pathlib import Path

from base_classes import LanguageParser, FileMetadata

logger = logging.getLogger(__name__)


class YamlParser(LanguageParser):
    """
    Example custom parser for YAML files.
    
    Parser Metadata (used by registry):
    """
    # These class attributes are used by the parser registry
    LANGUAGE = "yaml"
    EXTENSIONS = [".yml", ".yaml"]
    DESCRIPTION = "YAML configuration file parser"
    VERSION = "1.0.0"
    PRIORITY = 10  # Higher than built-in parsers
    
    def parse(self, content: str, file_path: str) -> FileMetadata:
        """
        Parse YAML content and extract metadata.
        
        Args:
            content: File content as string
            file_path: Path to the file being parsed
            
        Returns:
            FileMetadata object with extracted information
        """
        try:
            # Basic YAML structure extraction
            metadata = FileMetadata(
                path=file_path,
                size=len(content),
                language=self.LANGUAGE,
                last_modified=Path(file_path).stat().st_mtime if Path(file_path).exists() else 0,
                content_hash=self._calculate_content_hash(content)
            )
            
            # Extract YAML-specific information
            self._extract_yaml_structure(content, metadata)
            
            # Calculate complexity based on nesting depth and key count
            metadata.complexity_score = self._calculate_yaml_complexity(content)
            
            # Count tokens (approximate)
            metadata.token_count = len(re.findall(r'\S+', content))
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            # Return minimal metadata on error
            return FileMetadata(
                path=file_path,
                size=len(content),
                language=self.LANGUAGE,
                last_modified=0,
                content_hash=self._calculate_content_hash(content)
            )
    
    def _extract_yaml_structure(self, content: str, metadata: FileMetadata):
        """Extract YAML-specific structure information"""
        
        # Find top-level keys
        top_level_keys = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and ':' in line and not line.startswith(' '):
                key = line.split(':')[0].strip()
                if key and not key.startswith('-'):
                    top_level_keys.append(key)
        
        # Store as "functions" for consistency with other parsers
        metadata.functions = [{'name': key, 'type': 'yaml_key'} for key in top_level_keys]
        
        # Extract imports (if any YAML includes other files)
        imports = []
        for line in content.split('\n'):
            if 'include:' in line or 'import:' in line or '<<:' in line:
                # Simple extraction - could be more sophisticated
                match = re.search(r'[\'"]([^\'\"]+)[\'"]', line)
                if match:
                    imports.append(match.group(1))
        
        metadata.imports = imports
        
        # Look for environment variable references
        env_vars = re.findall(r'\$\{([^}]+)\}', content)
        metadata.external_dependencies = env_vars
        
    def _calculate_yaml_complexity(self, content: str) -> float:
        """Calculate complexity score based on YAML structure"""
        complexity = 1.0
        
        # Count nesting levels
        max_indent = 0
        for line in content.split('\n'):
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                indent = len(line) - len(stripped)
                max_indent = max(max_indent, indent)
        
        # Nesting depth contributes to complexity
        complexity += max_indent / 2.0
        
        # Count keys and lists
        key_count = len(re.findall(r'^\s*\w+:', content, re.MULTILINE))
        list_count = len(re.findall(r'^\s*-', content, re.MULTILINE))
        
        complexity += key_count * 0.1
        complexity += list_count * 0.05
        
        # Multi-line strings add complexity
        multiline_strings = len(re.findall(r'[>|][-+]?', content))
        complexity += multiline_strings * 0.5
        
        return complexity


def register_custom_parsers():
    """
    Example function showing how to register custom parsers at runtime.
    This can be called from your application code.
    """
    from parsers.registry import get_parser_registry
    
    registry = get_parser_registry()
    
    # Register YAML parser
    registry.register_custom_parser(
        name="yaml",
        parser_class=YamlParser,
        extensions=[".yml", ".yaml"],
        language="yaml",
        priority=15,  # Higher than built-in parsers
        description="Custom YAML configuration parser",
        version="1.0.0"
    )
    
    logger.info("Custom parsers registered successfully")


if __name__ == "__main__":
    # Example usage
    print("Testing custom parsers...")
    
    # Test YAML parser
    yaml_content = """
name: my-app
version: 1.0.0

database:
  host: ${DB_HOST}
  port: 5432
  
services:
  - web
  - api
  - worker

config:
  debug: true
  features:
    auth: enabled
    cache: redis
"""
    
    yaml_parser = YamlParser()
    yaml_metadata = yaml_parser.parse(yaml_content, "config.yml")
    print(f"YAML parser found {len(yaml_metadata.functions)} top-level keys")
    print(f"YAML complexity: {yaml_metadata.complexity_score:.2f}")