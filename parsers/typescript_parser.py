"""
TypeScript AST Parser Integration
=================================

Python wrapper for the TypeScript AST parser that uses @typescript-eslint/parser
to extract detailed metadata from TypeScript and JavaScript files.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import os
import hashlib
import time
import re
from dataclasses import dataclass, field

from base_classes import LanguageParser, FileMetadata

logger = logging.getLogger(__name__)


@dataclass
class TypeScriptFunction:
    """Represents a TypeScript function"""
    name: str
    params: List[Dict[str, Any]] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    is_arrow: bool = False
    return_type: Optional[str] = None
    line_number: int = 0


@dataclass
class TypeScriptClass:
    """Represents a TypeScript class"""
    name: str
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)
    methods: List[Dict[str, Any]] = field(default_factory=list)
    properties: List[Dict[str, Any]] = field(default_factory=list)
    is_abstract: bool = False
    line_number: int = 0


@dataclass
class TypeScriptInterface:
    """Represents a TypeScript interface"""
    name: str
    extends: List[str] = field(default_factory=list)
    members: List[Dict[str, Any]] = field(default_factory=list)
    line_number: int = 0


class TypeScriptParser(LanguageParser):
    """TypeScript/JavaScript parser using @typescript-eslint/parser"""
    
    def __init__(self):
        self.parser_script = Path(__file__).parent.parent / "typescript_ast_parser.cjs"
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check if Node.js and required packages are installed"""
        # Check Node.js
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Node.js available: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Node.js is not installed or not in PATH")
            raise RuntimeError("Node.js is required for TypeScript parsing")
            
        # Check if parser script exists
        if not self.parser_script.exists():
            logger.error(f"Parser script not found: {self.parser_script}")
            raise RuntimeError("TypeScript parser script not found")
            
    def parse(self, content: str, filepath: str) -> FileMetadata:
        """Parse TypeScript/JavaScript content using AST parser"""
        metadata = FileMetadata(
            path=filepath,
            size=len(content),
            language="typescript" if filepath.endswith(('.ts', '.tsx')) else "javascript",
            last_modified=time.time(),
            content_hash=hashlib.sha256(content.encode()).hexdigest()
        )
        
        # Determine file type
        path_obj = Path(filepath)
        if path_obj.suffix in ['.d.ts']:
            metadata.file_type = "declaration"
        elif path_obj.name.endswith(('.test.ts', '.test.tsx', '.spec.ts', '.spec.tsx')):
            metadata.file_type = "test"
        else:
            metadata.file_type = "source"
            
        try:
            # Create temporary file for parsing
            with tempfile.NamedTemporaryFile(mode='w', suffix=path_obj.suffix, delete=False) as tmp:
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name
                
            try:
                # Run the parser
                result = subprocess.run(
                    ["node", str(self.parser_script), tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True
                )
                
                # Parse the JSON output
                parsed_data = json.loads(result.stdout)
                
                if parsed_data.get('success'):
                    data = parsed_data['data']
                    
                    # Extract imports
                    for imp in data.get('imports', []):
                        source = imp['source']
                        metadata.imports.append(source)
                        
                        # Classify dependencies
                        if source.startswith('.'):
                            metadata.internal_dependencies.append(source)
                        else:
                            metadata.external_dependencies.append(source)
                            
                        # Add more detailed import info
                        metadata.dependencies.add(source)
                        
                    # Extract exports
                    for exp in data.get('exports', []):
                        if exp.get('default'):
                            metadata.exports.append('default')
                        if exp.get('names'):
                            for name_info in exp['names']:
                                metadata.exports.append(name_info['exported'])
                                
                    # Extract functions with full TypeScript details
                    for func in data.get('functions', []):
                        func_info = {
                            'name': func['name'],
                            'args': [p['name'] for p in func.get('params', [])],
                            'is_async': func.get('async', False),
                            'is_generator': func.get('generator', False),
                            'is_arrow': func['type'] == 'ArrowFunctionExpression',
                            'line': func.get('line', 0),
                            'params_details': func.get('params', []),
                            'return_type': func.get('returnType')
                        }
                        metadata.functions.append(func_info)
                        
                    # Extract classes with TypeScript features
                    for cls in data.get('classes', []):
                        class_info = {
                            'name': cls['name'],
                            'extends': cls.get('extends'),
                            'implements': cls.get('implements', []),
                            'methods': cls.get('methods', []),
                            'properties': cls.get('properties', []),
                            'is_abstract': cls.get('abstract', False),
                            'line': cls.get('line', 0)
                        }
                        metadata.classes.append(class_info)
                        
                    # Extract TypeScript-specific features
                    # Interfaces
                    for interface in data.get('interfaces', []):
                        # Store in classes for now but mark as interface
                        interface_info = {
                            'name': interface['name'],
                            'type': 'interface',
                            'extends': interface.get('extends', []),
                            'members': interface.get('members', []),
                            'line': interface.get('line', 0)
                        }
                        metadata.classes.append(interface_info)
                        
                    # Type aliases
                    types_count = len(data.get('types', []))
                    enums_count = len(data.get('enums', []))
                    
                    # Set complexity score
                    metadata.complexity_score = data.get('complexity', 1)
                    
                    # Add TypeScript-specific metadata
                    if not hasattr(metadata, 'typescript_features'):
                        metadata.typescript_features = {
                            'interfaces_count': len(data.get('interfaces', [])),
                            'types_count': types_count,
                            'enums_count': enums_count,
                            'has_generics': self._detect_generics(content),
                            'has_decorators': self._detect_decorators(content)
                        }
                        
                else:
                    # Parsing failed
                    error_msg = parsed_data.get('error', 'Unknown error')
                    logger.error(f"TypeScript parsing failed for {filepath}: {error_msg}")
                    # Fall back to basic metadata
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.debug(f"Could not delete temporary file {tmp_path}: {e}")
                    
        except subprocess.TimeoutExpired:
            logger.error(f"TypeScript parser timeout for {filepath}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse TypeScript parser output: {e}")
        except Exception as e:
            logger.error(f"TypeScript parsing error for {filepath}: {e}")
            
        # Always set token count
        metadata.token_count = len(self.tokenize(content))
        
        return metadata
        
    def tokenize(self, content: str) -> List[str]:
        """Enhanced tokenization for TypeScript"""
        # Remove comments
        content = self._remove_comments(content)
        
        # TypeScript-aware tokenization
        
        # Pattern for TypeScript tokens including type annotations
        token_pattern = r'''
            \b(?:class|interface|type|enum|namespace|module|declare|abstract|
               readonly|private|protected|public|static|async|await|
               extends|implements|export|import|from|as|is|keyof|typeof|
               infer|never|unknown|any|void|null|undefined)\b|
            \b\w+\b|                          # Regular words
            <[^>]+>|                          # Generics
            \?\.|                             # Optional chaining
            \?\?|                             # Nullish coalescing
            =>|                               # Arrow functions
            \.\.\.|                           # Spread operator
            [=<>!+\-*/&|^~%]+|               # Operators
            [(){}[\];,.:]|                    # Delimiters
            @\w+|                             # Decorators
            \'[^\']*\'|"[^"]*"|`[^`]*`      # Strings (including template literals)
        '''
        
        tokens = re.findall(token_pattern, content, re.VERBOSE)
        return tokens
        
    def extract_structure(self, content: str) -> Dict[str, Any]:
        """Extract detailed code structure"""
        # Use temporary file to parse
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
            
        try:
            result = subprocess.run(
                ["node", str(self.parser_script), tmp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                parsed = json.loads(result.stdout)
                if parsed.get('success'):
                    return parsed['data']
                    
        except Exception as e:
            logger.error(f"Structure extraction failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.debug(f"Could not delete temporary file {tmp_path}: {e}")
                
        return {}
        
    def _remove_comments(self, content: str) -> str:
        """Remove TypeScript comments while preserving strings"""
        
        # Remove single-line comments
        content = re.sub(r'(?<!:)//.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        
        # Remove JSDoc comments
        content = re.sub(r'/\*\*[\s\S]*?\*/', '', content)
        
        return content
        
    def _detect_generics(self, content: str) -> bool:
        """Detect if code uses TypeScript generics"""
        # Look for generic patterns like Array<T>, Promise<T>, <T extends ...>
        generic_patterns = [
            r'<[A-Z]\w*(?:\s+extends\s+\w+)?>', # Type parameters
            r'\w+<[^>]+>',                       # Generic usage
            r'Array<\w+>',                       # Array generics
            r'Promise<\w+>',                     # Promise generics
        ]
        
        for pattern in generic_patterns:
            if re.search(pattern, content):
                return True
        return False
        
    def _detect_decorators(self, content: str) -> bool:
        """Detect if code uses decorators"""
        # Look for decorator patterns @decorator
        return bool(re.search(r'@\w+\s*(?:\([^)]*\))?\s*(?:class|function|get|set|method)', content))


def test_typescript_parser():
    """Test the TypeScript parser"""
    parser = TypeScriptParser()
    
    # Test TypeScript code
    test_code = '''
import { Component, OnInit } from '@angular/core';
import * as lodash from 'lodash';

interface User {
    id: number;
    name: string;
    email?: string;
}

type Status = 'active' | 'inactive' | 'pending';

@Component({
    selector: 'app-user',
    templateUrl: './user.component.html'
})
export class UserComponent implements OnInit {
    private users: User[] = [];
    status: Status = 'active';
    
    constructor(private userService: UserService) {}
    
    async ngOnInit(): Promise<void> {
        this.users = await this.userService.getUsers();
    }
    
    public addUser(user: User): void {
        this.users.push(user);
    }
    
    private validateUser<T extends User>(user: T): boolean {
        return user.name.length > 0;
    }
}

export const helper = {
    formatName: (user: User): string => {
        return `${user.name} (${user.id})`;
    }
};

export default UserComponent;
'''
    
    # Parse the code
    metadata = parser.parse(test_code, "test.component.ts")
    
    # Print results
    print("=== TypeScript Parser Test Results ===")
    print(f"Language: {metadata.language}")
    print(f"Imports: {metadata.imports}")
    print(f"Exports: {metadata.exports}")
    print(f"Classes: {len(metadata.classes)}")
    for cls in metadata.classes:
        print(f"  - {cls['name']} {'(interface)' if cls.get('type') == 'interface' else ''}")
    print(f"Functions: {len(metadata.functions)}")
    for func in metadata.functions:
        print(f"  - {func['name']} {'(async)' if func['is_async'] else ''}")
    print(f"Complexity: {metadata.complexity_score}")
    
    if hasattr(metadata, 'typescript_features'):
        print(f"TypeScript features: {metadata.typescript_features}")


if __name__ == "__main__":
    test_typescript_parser()