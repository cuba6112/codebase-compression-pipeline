"""
Python language parser using AST.
"""

import ast
import re
import time
import hashlib
import logging
from typing import Dict, List, Any

from base_classes import FileMetadata, LanguageParser
from .base import ParserMixin, TokenizerMixin

logger = logging.getLogger(__name__)


class PythonParser(LanguageParser, TokenizerMixin, ParserMixin):
    """Python-specific parser using AST for accurate parsing."""
    
    def parse(self, content: str, filepath: str) -> FileMetadata:
        """Parse Python file and extract metadata."""
        metadata = FileMetadata(
            path=filepath,
            size=len(content),
            language="python",
            last_modified=time.time(),
            content_hash=hashlib.sha256(content.encode()).hexdigest()
        )
        
        try:
            tree = ast.parse(content)
            self._extract_imports(tree, metadata)
            self._extract_definitions(tree, metadata)
            metadata.complexity_score = self._calculate_complexity(tree)
        except SyntaxError as e:
            # Handle parsing errors gracefully
            logger.warning(f"Syntax error parsing {filepath}: {e}")
            # Fall back to regex-based parsing for basic structure
            self._fallback_parse(content, metadata)
        except Exception as e:
            logger.error(f"Unexpected error parsing {filepath}: {e}")
            self._fallback_parse(content, metadata)
        
        metadata.token_count = len(self.tokenize(content))
        return metadata
    
    def tokenize(self, content: str) -> List[str]:
        """Intelligent tokenization preserving code structure."""
        # Remove comments while preserving strings
        content_no_comments = self.remove_comments(content, r'#.*$')
        
        # Use the mixin's tokenizer
        return self.basic_tokenize(content_no_comments, preserve_strings=True)
    
    def extract_structure(self, content: str) -> Dict[str, Any]:
        """Extract structural elements for semantic compression."""
        try:
            tree = ast.parse(content)
            return {
                'imports': [self._node_to_dict(node) for node in ast.walk(tree) 
                           if isinstance(node, (ast.Import, ast.ImportFrom))],
                'functions': [self._node_to_dict(node) for node in ast.walk(tree) 
                             if isinstance(node, ast.FunctionDef)],
                'classes': [self._node_to_dict(node) for node in ast.walk(tree) 
                           if isinstance(node, ast.ClassDef)],
                'async_functions': [self._node_to_dict(node) for node in ast.walk(tree) 
                                   if isinstance(node, ast.AsyncFunctionDef)],
                'decorators': self._extract_decorators(tree),
                'global_vars': self._extract_global_vars(tree)
            }
        except SyntaxError as e:
            logger.warning(f"Syntax error in extract_structure: {e}")
            return self._fallback_structure_extract(content)
        except Exception as e:
            logger.error(f"Unexpected error in extract_structure: {e}")
            return {}
    
    def _extract_imports(self, tree: ast.AST, metadata: FileMetadata):
        """Extract import statements from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    metadata.imports.append(import_name)
                    # Classify as external/internal
                    if import_name.startswith('.') or '.' not in import_name.split('/')[0]:
                        metadata.internal_dependencies.append(import_name)
                    else:
                        metadata.external_dependencies.append(import_name)
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    metadata.imports.append(node.module)
                    # Classify as external/internal
                    if node.level > 0 or node.module.startswith('.'):
                        metadata.internal_dependencies.append(node.module)
                    else:
                        metadata.external_dependencies.append(node.module)
    
    def _extract_definitions(self, tree: ast.AST, metadata: FileMetadata):
        """Extract function and class definitions from AST."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [self._decorator_to_string(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node),
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'line_number': node.lineno,
                    'complexity': self._calculate_function_complexity(node)
                }
                
                # Extract type hints if available
                if hasattr(node, 'returns') and node.returns:
                    func_info['return_type'] = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
                
                metadata.functions.append(func_info)
                
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'bases': [self._base_to_string(base) for base in node.bases],
                    'decorators': [self._decorator_to_string(d) for d in node.decorator_list],
                    'methods': [],
                    'attributes': [],
                    'docstring': ast.get_docstring(node),
                    'line_number': node.lineno
                }
                
                # Extract methods and attributes
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_info['methods'].append({
                            'name': item.name,
                            'is_async': isinstance(item, ast.AsyncFunctionDef),
                            'is_property': any(self._decorator_to_string(d) == 'property' 
                                             for d in item.decorator_list)
                        })
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        # Class attributes with type annotations
                        class_info['attributes'].append({
                            'name': item.target.id,
                            'type': ast.unparse(item.annotation) if hasattr(ast, 'unparse') else str(item.annotation)
                        })
                
                metadata.classes.append(class_info)
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity for the entire module."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                # Comprehensions add complexity
                complexity += 1
        return float(complexity)
    
    def _calculate_function_complexity(self, func_node: ast.AST) -> int:
        """Calculate complexity for a single function."""
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _node_to_dict(self, node: ast.AST) -> Dict[str, Any]:
        """Convert AST node to dictionary representation."""
        node_dict = {
            'type': node.__class__.__name__,
            'lineno': getattr(node, 'lineno', None),
            'col_offset': getattr(node, 'col_offset', None)
        }
        
        # Add node-specific information
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_dict['name'] = node.name
        elif isinstance(node, ast.Import):
            node_dict['names'] = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            node_dict['module'] = node.module
            node_dict['names'] = [alias.name for alias in node.names] if node.names else []
            
        return node_dict
    
    def _decorator_to_string(self, decorator: ast.AST) -> str:
        """Convert decorator AST node to string representation."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._decorator_to_string(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._decorator_to_string(decorator.func)
        else:
            return ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator)
    
    def _base_to_string(self, base: ast.AST) -> str:
        """Convert base class AST node to string representation."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self._base_to_string(base.value)}.{base.attr}"
        else:
            return ast.unparse(base) if hasattr(ast, 'unparse') else str(base)
    
    def _extract_decorators(self, tree: ast.AST) -> List[str]:
        """Extract all unique decorators used in the module."""
        decorators = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    decorators.add(self._decorator_to_string(decorator))
        return list(decorators)
    
    def _extract_global_vars(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract global variable definitions."""
        global_vars = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        global_vars.append({
                            'name': target.id,
                            'line': node.lineno
                        })
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                global_vars.append({
                    'name': node.target.id,
                    'type': ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation),
                    'line': node.lineno
                })
        return global_vars
    
    def _fallback_parse(self, content: str, metadata: FileMetadata):
        """Fallback parsing using regex when AST parsing fails."""
        # Extract imports using regex
        import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            if match.group(1):
                metadata.imports.append(match.group(1))
            else:
                imports = match.group(2).split(',')
                metadata.imports.extend([imp.strip().split()[0] for imp in imports])
        
        # Extract function definitions
        func_pattern = r'^(?:async\s+)?def\s+(\w+)\s*\([^)]*\):'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            metadata.functions.append({
                'name': match.group(1),
                'args': [],
                'decorators': [],
                'is_async': 'async' in match.group(0)
            })
        
        # Extract class definitions
        class_pattern = r'^class\s+(\w+)(?:\s*\([^)]*\))?:'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            metadata.classes.append({
                'name': match.group(1),
                'bases': [],
                'methods': []
            })
        
        # Basic complexity
        metadata.complexity_score = self.calculate_basic_complexity(content)
    
    def _fallback_structure_extract(self, content: str) -> Dict[str, Any]:
        """Fallback structure extraction using regex."""
        return {
            'imports': re.findall(r'^\s*(?:from\s+\S+\s+)?import\s+.+$', content, re.MULTILINE),
            'functions': re.findall(r'^(?:async\s+)?def\s+(\w+)', content, re.MULTILINE),
            'classes': re.findall(r'^class\s+(\w+)', content, re.MULTILINE),
            'decorators': re.findall(r'^@(\w+)', content, re.MULTILINE)
        }