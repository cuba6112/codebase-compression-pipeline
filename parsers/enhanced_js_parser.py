"""
Enhanced JavaScript/TypeScript Parser with AST Support
======================================================

Provides robust parsing of JavaScript and TypeScript code using
advanced pattern matching and optional Node.js integration.
"""

import re
import json
import subprocess
import tempfile
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field

from base_classes import LanguageParser, FileMetadata

logger = logging.getLogger(__name__)


@dataclass
class JSFunction:
    """Represents a JavaScript function"""
    name: str
    params: List[str] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    is_arrow: bool = False
    line_number: int = 0
    docstring: Optional[str] = None
    

@dataclass
class JSClass:
    """Represents a JavaScript class"""
    name: str
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    is_abstract: bool = False
    line_number: int = 0
    

@dataclass
class JSImport:
    """Represents a JavaScript import statement"""
    module: str
    imports: List[str] = field(default_factory=list)
    is_default: bool = False
    is_namespace: bool = False
    alias: Optional[str] = None
    line_number: int = 0


@dataclass 
class JSExport:
    """Represents a JavaScript export statement"""
    name: str
    is_default: bool = False
    is_type: bool = False
    line_number: int = 0


class EnhancedJavaScriptParser(LanguageParser):
    """Advanced JavaScript/TypeScript parser with fallback strategies"""
    
    def __init__(self, use_node_parser: bool = False):
        self.use_node_parser = use_node_parser
        self._check_node_availability()
        
    def _check_node_availability(self):
        """Check if Node.js is available for AST parsing"""
        if self.use_node_parser:
            try:
                result = subprocess.run(['node', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"Node.js available: {result.stdout.strip()}")
                else:
                    logger.warning("Node.js not available, falling back to regex parser")
                    self.use_node_parser = False
            except Exception:
                logger.warning("Node.js not available, falling back to regex parser")
                self.use_node_parser = False
                
    def parse(self, content: str, filepath: str) -> FileMetadata:
        """Parse JavaScript/TypeScript content"""
        # Create base metadata
        metadata = FileMetadata(
            path=filepath,
            size=len(content),
            language="typescript" if filepath.endswith(('.ts', '.tsx')) else "javascript",
            last_modified=time.time(),
            content_hash=hashlib.sha256(content.encode()).hexdigest()
        )
        
        if self.use_node_parser and filepath.endswith(('.ts', '.tsx')):
            # Use Node.js parser for TypeScript files
            parsed_data = self._parse_with_node(content, filepath)
        else:
            # Use enhanced regex parser
            parsed_data = self._parse_with_regex(content, filepath)
            
        # Populate metadata from parsed data
        self._populate_metadata(metadata, parsed_data)
        
        # Set token count
        metadata.token_count = len(self.tokenize(content))
        
        return metadata
            
    def _parse_with_regex(self, content: str, filepath: str) -> Dict[str, Any]:
        """Enhanced regex-based parsing"""
        # Remove comments first
        content_no_comments = self._remove_comments(content)
        
        return {
            'imports': self._extract_imports(content_no_comments),
            'exports': self._extract_exports(content_no_comments),
            'functions': self._extract_functions(content_no_comments),
            'classes': self._extract_classes(content_no_comments),
            'variables': self._extract_variables(content_no_comments),
            'types': self._extract_types(content_no_comments) if filepath.endswith('.ts') else [],
            'complexity': self._calculate_complexity(content_no_comments)
        }
        
    def _remove_comments(self, content: str) -> str:
        """Remove JavaScript comments while preserving strings"""
        # Remove single-line comments
        content = re.sub(r'(?<!:)//.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        
        return content
        
    def _extract_imports(self, content: str) -> List[Dict[str, Any]]:
        """Extract import statements with detailed information"""
        imports = []
        lines = content.split('\n')
        
        # Standard ES6 imports
        import_patterns = [
            # import { a, b } from 'module'
            r'import\s*\{([^}]+)\}\s*from\s*[\'"]([^\'"]+)[\'"]',
            # import * as name from 'module'
            r'import\s*\*\s*as\s+(\w+)\s*from\s*[\'"]([^\'"]+)[\'"]',
            # import name from 'module'
            r'import\s+(\w+)\s*from\s*[\'"]([^\'"]+)[\'"]',
            # import 'module'
            r'import\s*[\'"]([^\'"]+)[\'"]',
            # const { a, b } = require('module')
            r'const\s*\{([^}]+)\}\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)',
            # const name = require('module')
            r'const\s+(\w+)\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)'
        ]
        
        for i, line in enumerate(lines):
            for pattern in import_patterns:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) == 2:
                        imports.append({
                            'module': match.group(2),
                            'imports': [match.group(1).strip()],
                            'line': i + 1
                        })
                    elif len(match.groups()) == 1:
                        imports.append({
                            'module': match.group(1),
                            'imports': [],
                            'line': i + 1
                        })
                    break
                    
        return imports
        
    def _extract_exports(self, content: str) -> List[Dict[str, Any]]:
        """Extract export statements"""
        exports = []
        lines = content.split('\n')
        
        export_patterns = [
            # export default
            (r'export\s+default\s+(\w+)', True),
            # export { a, b }
            (r'export\s*\{([^}]+)\}', False),
            # export const/let/var/function/class
            (r'export\s+(?:const|let|var|function|class)\s+(\w+)', False),
            # export type (TypeScript)
            (r'export\s+type\s+(\w+)', False),
            # export interface (TypeScript)
            (r'export\s+interface\s+(\w+)', False)
        ]
        
        for i, line in enumerate(lines):
            for pattern, is_default in export_patterns:
                match = re.search(pattern, line)
                if match:
                    if '{' in match.group(0):
                        # Multiple exports
                        names = [n.strip() for n in match.group(1).split(',')]
                        for name in names:
                            exports.append({
                                'name': name,
                                'is_default': is_default,
                                'line': i + 1
                            })
                    else:
                        exports.append({
                            'name': match.group(1),
                            'is_default': is_default,
                            'line': i + 1
                        })
                    break
                    
        return exports
        
    def _extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """Extract function definitions with parameters"""
        functions = []
        lines = content.split('\n')
        
        # Function patterns
        patterns = [
            # Regular function: function name(params)
            r'(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
            # Arrow function: const name = (params) =>
            r'const\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>',
            # Arrow function: const name = async params =>
            r'const\s+(\w+)\s*=\s*async\s+([^=]+)\s*=>',
            # Method in object: name(params) {
            r'(\w+)\s*\(([^)]*)\)\s*\{',
            # Method with async: async name(params) {
            r'async\s+(\w+)\s*\(([^)]*)\)\s*\{'
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    params_str = match.group(2) if len(match.groups()) > 1 else ''
                    
                    # Parse parameters
                    params = []
                    if params_str:
                        params = [p.strip().split(':')[0].strip() 
                                for p in params_str.split(',') if p.strip()]
                    
                    functions.append({
                        'name': func_name,
                        'params': params,
                        'is_async': 'async' in line,
                        'is_arrow': '=>' in line,
                        'line': i + 1
                    })
                    break
                    
        return functions
        
    def _extract_classes(self, content: str) -> List[Dict[str, Any]]:
        """Extract class definitions"""
        classes = []
        lines = content.split('\n')
        
        class_pattern = r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?'
        
        i = 0
        while i < len(lines):
            match = re.search(class_pattern, lines[i])
            if match:
                class_name = match.group(1)
                extends = match.group(2)
                implements = []
                
                if match.group(3):
                    implements = [impl.strip() for impl in match.group(3).split(',')]
                
                # Extract class body to find methods
                methods = []
                properties = []
                
                # Find the opening brace
                brace_count = 0
                j = i
                class_started = False
                
                while j < len(lines):
                    if '{' in lines[j]:
                        class_started = True
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                        
                    if class_started and j > i:
                        # Look for methods
                        method_match = re.search(r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{', lines[j])
                        if method_match:
                            methods.append(method_match.group(1))
                        
                        # Look for properties
                        prop_match = re.search(r'(?:public|private|protected)?\s*(\w+)\s*[:=]', lines[j])
                        if prop_match and prop_match.group(1) not in methods:
                            properties.append(prop_match.group(1))
                    
                    if class_started and brace_count == 0:
                        break
                        
                    j += 1
                
                classes.append({
                    'name': class_name,
                    'extends': extends,
                    'implements': implements,
                    'methods': methods,
                    'properties': properties,
                    'is_abstract': 'abstract' in lines[i],
                    'line': i + 1
                })
                
            i += 1
            
        return classes
        
    def _extract_variables(self, content: str) -> List[Dict[str, Any]]:
        """Extract variable declarations"""
        variables = []
        lines = content.split('\n')
        
        var_patterns = [
            r'(?:const|let|var)\s+(\w+)\s*=',
            r'(?:const|let|var)\s+\{([^}]+)\}\s*=',
            r'(?:const|let|var)\s+\[([^\]]+)\]\s*='
        ]
        
        for i, line in enumerate(lines):
            for pattern in var_patterns:
                match = re.search(pattern, line)
                if match:
                    if '{' in match.group(0) or '[' in match.group(0):
                        # Destructuring
                        names = re.findall(r'\w+', match.group(1))
                        for name in names:
                            variables.append({
                                'name': name,
                                'type': 'const' if 'const' in line else 'let' if 'let' in line else 'var',
                                'line': i + 1
                            })
                    else:
                        variables.append({
                            'name': match.group(1),
                            'type': 'const' if 'const' in line else 'let' if 'let' in line else 'var',
                            'line': i + 1
                        })
                    break
                    
        return variables
        
    def _extract_types(self, content: str) -> List[Dict[str, Any]]:
        """Extract TypeScript type definitions"""
        types = []
        lines = content.split('\n')
        
        type_patterns = [
            r'type\s+(\w+)\s*=',
            r'interface\s+(\w+)\s*(?:extends\s+([^{]+))?\s*\{',
            r'enum\s+(\w+)\s*\{'
        ]
        
        for i, line in enumerate(lines):
            for pattern in type_patterns:
                match = re.search(pattern, line)
                if match:
                    type_info = {
                        'name': match.group(1),
                        'kind': 'type' if 'type' in line else 'interface' if 'interface' in line else 'enum',
                        'line': i + 1
                    }
                    
                    if 'interface' in line and len(match.groups()) > 1 and match.group(2):
                        type_info['extends'] = [e.strip() for e in match.group(2).split(',')]
                        
                    types.append(type_info)
                    break
                    
        return types
        
    def _calculate_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        # Count decision points
        complexity_patterns = [
            r'\bif\b',
            r'\belse\s+if\b',
            r'\bwhile\b',
            r'\bfor\b',
            r'\bcase\b',
            r'\bcatch\b',
            r'\?\s*[^:]+\s*:',  # Ternary operator
            r'&&',
            r'\|\|'
        ]
        
        for pattern in complexity_patterns:
            complexity += len(re.findall(pattern, content))
            
        return complexity
        
    def _populate_metadata(self, metadata: FileMetadata, parsed_data: Dict[str, Any]):
        """Populate FileMetadata object from parsed data"""
        # Extract imports
        for imp in parsed_data.get('imports', []):
            module = imp.get('module', '')
            metadata.imports.append(module)
            metadata.dependencies.add(module)
            
            # Classify as internal/external dependency
            if module.startswith('.') or module.startswith('/'):
                metadata.internal_dependencies.append(module)
            else:
                metadata.external_dependencies.append(module)
        
        # Extract exports
        for exp in parsed_data.get('exports', []):
            metadata.exports.append(exp.get('name', ''))
        
        # Extract functions
        for func in parsed_data.get('functions', []):
            func_info = {
                'name': func.get('name', ''),
                'args': func.get('params', []),
                'is_async': func.get('is_async', False),
                'is_arrow': func.get('is_arrow', False),
                'line_number': func.get('line', 0)
            }
            metadata.functions.append(func_info)
        
        # Extract classes
        for cls in parsed_data.get('classes', []):
            class_info = {
                'name': cls.get('name', ''),
                'extends': cls.get('extends'),
                'implements': cls.get('implements', []),
                'methods': cls.get('methods', []),
                'properties': cls.get('properties', []),
                'is_abstract': cls.get('is_abstract', False),
                'line_number': cls.get('line', 0)
            }
            metadata.classes.append(class_info)
        
        # Set complexity score
        metadata.complexity_score = float(parsed_data.get('complexity', 1))
        
        # Add TypeScript-specific features if available
        if parsed_data.get('types'):
            metadata.typescript_features = {
                'types_count': len(parsed_data.get('types', [])),
                'has_interfaces': any(t.get('kind') == 'interface' for t in parsed_data.get('types', [])),
                'has_enums': any(t.get('kind') == 'enum' for t in parsed_data.get('types', [])),
                'has_generics': self._detect_generics_in_data(parsed_data)
            }
    
    def _detect_generics_in_data(self, parsed_data: Dict[str, Any]) -> bool:
        """Detect if parsed data contains generics"""
        # Check if any functions or classes have generic type parameters
        for func in parsed_data.get('functions', []):
            if '<' in func.get('name', '') or any('<' in p for p in func.get('params', [])):
                return True
        for cls in parsed_data.get('classes', []):
            if '<' in cls.get('name', ''):
                return True
        return False
    
    def tokenize(self, content: str) -> List[str]:
        """Tokenize JavaScript/TypeScript content"""
        # Remove comments first
        content_no_comments = self._remove_comments(content)
        
        # JavaScript/TypeScript token patterns
        token_pattern = r'''
            \b(?:class|interface|type|enum|namespace|module|declare|abstract|
               readonly|private|protected|public|static|async|await|function|
               const|let|var|if|else|for|while|do|switch|case|default|
               try|catch|finally|throw|return|break|continue|new|this|
               extends|implements|export|import|from|as|typeof|instanceof|
               true|false|null|undefined)\b|
            \b\w+\b|                          # Regular words
            \d+\.?\d*|                       # Numbers
            =>|                               # Arrow functions
            \.\.\.|                           # Spread operator
            \?\.|                             # Optional chaining
            \?\?|                             # Nullish coalescing
            [=<>!+\-*/&|^~%]+|               # Operators
            [(){}[\];,.:]|                    # Delimiters
            \'[^\']*\'|"[^"]*"|`[^`]*`      # Strings (including template literals)
        '''
        
        tokens = re.findall(token_pattern, content_no_comments, re.VERBOSE)
        return [token for token in tokens if token.strip()]
    
    def extract_structure(self, content: str) -> Dict[str, Any]:
        """Extract structural elements from JavaScript/TypeScript code"""
        # Use the same parsing logic as parse method but return raw structure
        if self.use_node_parser:
            return self._parse_with_node(content, "temp.js")
        else:
            return self._parse_with_regex(content, "temp.js")
        
    def _parse_with_node(self, content: str, filepath: str) -> Dict[str, Any]:
        """Parse using Node.js AST parser (requires Node.js)"""
        # This would require a Node.js script to parse the file
        # For now, fall back to regex parser
        logger.info("Node.js parser not implemented, using regex parser")
        return self._parse_with_regex(content, filepath)