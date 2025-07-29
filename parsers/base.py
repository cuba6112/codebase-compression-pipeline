"""
Base parser mixins and utilities for language parsers.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ParserMixin:
    """Common parser utilities shared across all language parsers."""
    
    @staticmethod
    def remove_comments(content: str, single_line_pattern: str, multi_line_pattern: Optional[str] = None) -> str:
        """Remove comments from source code."""
        try:
            # Remove single-line comments
            content = re.sub(single_line_pattern, '', content, flags=re.MULTILINE)
            
            # Remove multi-line comments if pattern provided
            if multi_line_pattern:
                content = re.sub(multi_line_pattern, '', content, flags=re.DOTALL)
                
            return content
        except re.error as e:
            logger.error(f"Regex error removing comments: {e}")
            return content


class TokenizerMixin:
    """Common tokenization logic for all parsers."""
    
    @staticmethod
    def basic_tokenize(content: str, preserve_strings: bool = True) -> List[str]:
        """Basic tokenization that works for most languages."""
        if not isinstance(content, str):
            raise TypeError("Content must be a string")
        
        tokens = []
        
        # If preserving strings, use a more complex regex
        if preserve_strings:
            # Pattern to match strings, identifiers, numbers, and operators
            pattern = r'"[^"]*"|\'[^\']*\'|\w+|[^\w\s]'
        else:
            # Simple word and non-word pattern
            pattern = r'\w+|[^\w\s]'
        
        try:
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    tokens.extend(re.findall(pattern, line))
        except re.error as e:
            logger.error(f"Regex error in tokenization: {e}")
            
        return tokens
    
    @staticmethod
    def count_tokens(content: str) -> int:
        """Count approximate tokens in content."""
        # Simple approximation: split on whitespace and punctuation
        return len(re.findall(r'\w+|[^\w\s]', content))


class ImportExtractorMixin:
    """Common import extraction patterns."""
    
    @staticmethod
    def extract_python_style_imports(content: str) -> List[str]:
        """Extract Python-style imports (import X, from X import Y)."""
        imports = []
        
        # Match 'import module' and 'from module import ...'
        import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$'
        
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            from_module = match.group(1)
            import_names = match.group(2)
            
            if from_module:
                imports.append(from_module)
            else:
                # Handle comma-separated imports
                for name in import_names.split(','):
                    imports.append(name.strip().split()[0])
                    
        return imports
    
    @staticmethod
    def extract_javascript_style_imports(content: str) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript style imports."""
        imports = []
        
        # ES6 imports
        es6_pattern = r'import\s+(?:{[^}]+}|[\w*]+(?:\s+as\s+\w+)?)\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(es6_pattern, content):
            imports.append({
                'module': match.group(1),
                'type': 'es6'
            })
        
        # CommonJS require
        require_pattern = r'(?:const|let|var)\s+\w+\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        for match in re.finditer(require_pattern, content):
            imports.append({
                'module': match.group(1),
                'type': 'commonjs'
            })
            
        return imports
    
    @staticmethod
    def extract_go_style_imports(content: str) -> List[str]:
        """Extract Go-style imports."""
        imports = []
        
        # Single import
        single_pattern = r'import\s+"([^"]+)"'
        imports.extend(re.findall(single_pattern, content))
        
        # Multiple imports
        multi_pattern = r'import\s*\([^)]+\)'
        for block in re.finditer(multi_pattern, content, re.DOTALL):
            import_block = block.group(0)
            imports.extend(re.findall(r'"([^"]+)"', import_block))
            
        return imports


class StructureExtractorMixin:
    """Common structure extraction patterns."""
    
    @staticmethod
    def extract_function_signatures(content: str, patterns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract function signatures using language-specific patterns."""
        functions = []
        
        for lang, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.MULTILINE):
                func_info = {
                    'name': match.group('name') if 'name' in match.groupdict() else match.group(1),
                    'language_hint': lang
                }
                
                # Add optional groups if they exist
                if 'params' in match.groupdict():
                    func_info['params'] = match.group('params')
                if 'return_type' in match.groupdict():
                    func_info['return_type'] = match.group('return_type')
                    
                functions.append(func_info)
                
        return functions
    
    @staticmethod
    def extract_class_definitions(content: str, patterns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract class definitions using language-specific patterns."""
        classes = []
        
        for lang, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.MULTILINE):
                class_info = {
                    'name': match.group('name') if 'name' in match.groupdict() else match.group(1),
                    'language_hint': lang
                }
                
                # Add optional groups
                if 'extends' in match.groupdict():
                    class_info['extends'] = match.group('extends')
                if 'implements' in match.groupdict():
                    class_info['implements'] = match.group('implements')
                    
                classes.append(class_info)
                
        return classes
    
    @staticmethod
    def calculate_basic_complexity(content: str) -> float:
        """Calculate basic complexity based on control flow keywords."""
        complexity = 1
        
        # Control flow keywords that increase complexity
        control_flow = [
            r'\bif\b', r'\belse\b', r'\belif\b', r'\bwhile\b', 
            r'\bfor\b', r'\bcase\b', r'\bcatch\b', r'\bexcept\b',
            r'\bswitch\b', r'\btry\b', r'\bfinally\b'
        ]
        
        for pattern in control_flow:
            complexity += len(re.findall(pattern, content, re.IGNORECASE))
            
        # Logical operators also increase complexity
        complexity += len(re.findall(r'\b(?:and|or|&&|\|\|)\b', content))
        
        return float(complexity)