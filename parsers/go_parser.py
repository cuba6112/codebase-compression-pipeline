"""
Go AST Parser Integration
========================

Python wrapper for the Go AST parser that uses go/ast package
to extract detailed metadata from Go source files.
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
class GoFunction:
    """Represents a Go function"""
    name: str
    receiver: Optional[str] = None
    params: List[Dict[str, str]] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    is_exported: bool = False
    line_number: int = 0


@dataclass 
class GoType:
    """Represents a Go type declaration"""
    name: str
    type_kind: str  # struct, interface, alias, etc.
    fields: List[Dict[str, Any]] = field(default_factory=list)
    methods: List[GoFunction] = field(default_factory=list)
    is_exported: bool = False
    line_number: int = 0


@dataclass
class GoInterface:
    """Represents a Go interface"""
    name: str
    methods: List[Dict[str, Any]] = field(default_factory=list)
    is_exported: bool = False
    line_number: int = 0


class GoParser(LanguageParser):
    """Go parser using go/ast package"""
    
    def __init__(self):
        self.parser_binary = Path(__file__).parent / "go_ast_parser"
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check if Go and the parser binary are available"""
        # Check Go
        try:
            result = subprocess.run(
                ["go", "version"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Go available: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Go is not installed or not in PATH")
            logger.info("Install Go from https://golang.org/dl/ or use: brew install go")
            
        # Check if parser binary exists
        if not self.parser_binary.exists():
            # Try to build it
            go_file = Path(__file__).parent / "go_ast_parser.go"
            if go_file.exists():
                try:
                    logger.info("Building Go parser binary...")
                    subprocess.run(
                        ["go", "build", "-o", str(self.parser_binary), str(go_file)],
                        check=True
                    )
                    logger.info("Go parser binary built successfully")
                except Exception as e:
                    logger.error(f"Failed to build Go parser: {e}")
                    
    def parse(self, content: str, filepath: str) -> FileMetadata:
        """Parse Go content using AST parser"""
        metadata = FileMetadata(
            path=filepath,
            size=len(content),
            language="go",
            last_modified=time.time(),
            content_hash=hashlib.sha256(content.encode()).hexdigest()
        )
        
        # Determine file type
        path_obj = Path(filepath)
        if path_obj.name.endswith('_test.go'):
            metadata.file_type = "test"
        elif path_obj.name == "main.go":
            metadata.file_type = "main"
        else:
            metadata.file_type = "source"
            
        # If parser binary doesn't exist, fall back to basic parsing
        if not self.parser_binary.exists():
            logger.warning("Go parser binary not available, using basic parsing")
            self._basic_parse(content, metadata)
        else:
            try:
                # Create temporary file for parsing
                with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as tmp:
                    tmp.write(content)
                    tmp.flush()
                    tmp_path = tmp.name
                    
                try:
                    # Run the parser
                    result = subprocess.run(
                        [str(self.parser_binary), tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=True
                    )
                    
                    # Parse the JSON output
                    parsed_data = json.loads(result.stdout)
                    
                    if parsed_data.get('success'):
                        data = parsed_data['data']
                        
                        # Extract package
                        if 'package' in data:
                            metadata.module_path = data['package']
                        
                        # Extract imports
                        for imp in data.get('imports', []):
                            path = imp['path']
                            metadata.imports.append(path)
                            
                            # Classify dependencies
                            if path.startswith('.'):
                                metadata.internal_dependencies.append(path)
                            else:
                                metadata.external_dependencies.append(path)
                                
                            metadata.dependencies.add(path)
                            
                        # Extract functions
                        for func in data.get('functions', []):
                            func_info = {
                                'name': func['name'],
                                'args': [p['name'] for p in func.get('params', [])],
                                'receiver': func.get('receiver'),
                                'is_exported': func.get('is_exported', False),
                                'line': func.get('line', 0),
                                'results': func.get('results', [])
                            }
                            metadata.functions.append(func_info)
                            
                        # Extract types (structs)
                        for typ in data.get('types', []):
                            type_info = {
                                'name': typ['name'],
                                'type': typ.get('type', 'type'),
                                'fields': typ.get('fields', []),
                                'methods': typ.get('methods', []),
                                'is_exported': typ.get('is_exported', False),
                                'line': typ.get('line', 0)
                            }
                            metadata.classes.append(type_info)
                            
                        # Extract interfaces
                        for iface in data.get('interfaces', []):
                            interface_info = {
                                'name': iface['name'],
                                'type': 'interface',
                                'methods': iface.get('methods', []),
                                'is_exported': iface.get('is_exported', False),
                                'line': iface.get('line', 0)
                            }
                            metadata.classes.append(interface_info)
                            
                        # Set complexity score
                        metadata.complexity_score = data.get('complexity', 1)
                        
                        # Add Go-specific metadata
                        if not hasattr(metadata, 'go_features'):
                            metadata.go_features = {
                                'types_count': len(data.get('types', [])),
                                'interfaces_count': len(data.get('interfaces', [])),
                                'constants_count': len(data.get('constants', [])),
                                'variables_count': len(data.get('variables', [])),
                                'has_goroutines': self._detect_goroutines(content),
                                'has_channels': self._detect_channels(content),
                                'has_generics': self._detect_generics(content)
                            }
                            
                    else:
                        # Parsing failed
                        error_msg = parsed_data.get('error', 'Unknown error')
                        logger.error(f"Go parsing failed for {filepath}: {error_msg}")
                        self._basic_parse(content, metadata)
                        
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.debug(f"Could not delete temporary file {tmp_path}: {e}")
                        
            except subprocess.TimeoutExpired:
                logger.error(f"Go parser timeout for {filepath}")
                self._basic_parse(content, metadata)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Go parser output: {e}")
                self._basic_parse(content, metadata)
            except Exception as e:
                logger.error(f"Go parsing error for {filepath}: {e}")
                self._basic_parse(content, metadata)
                
        # Always set token count
        metadata.token_count = len(self.tokenize(content))
        
        return metadata
        
    def _basic_parse(self, content: str, metadata: FileMetadata):
        """Basic Go parsing using regex when AST parser is not available"""
        
        # Extract package
        package_match = re.search(r'^package\s+(\w+)', content, re.MULTILINE)
        if package_match:
            metadata.module_path = package_match.group(1)
            
        # Extract imports
        imports = re.findall(r'import\s+(?:"([^"]+)"|(?:\(\s*((?:[^)]+\s*)+)\s*\)))', content, re.DOTALL)
        for imp in imports:
            if imp[0]:  # Single import
                metadata.imports.append(imp[0])
                metadata.dependencies.add(imp[0])
            else:  # Multiple imports
                for line in imp[1].strip().split('\n'):
                    line = line.strip()
                    if line:
                        # Handle aliased imports
                        import_match = re.match(r'(?:\w+\s+)?"([^"]+)"', line)
                        if import_match:
                            path = import_match.group(1)
                            metadata.imports.append(path)
                            metadata.dependencies.add(path)
                            
        # Extract functions
        func_pattern = r'func\s+(?:\(([^)]+)\)\s+)?(\w+)\s*\([^)]*\)(?:\s*(?:\([^)]*\)|[^{]*))?'
        for match in re.finditer(func_pattern, content):
            receiver = match.group(1)
            name = match.group(2)
            metadata.functions.append({
                'name': name,
                'receiver': receiver.split()[-1] if receiver else None,
                'args': [],
                'is_exported': name[0].isupper() if name else False
            })
            
        # Extract types
        type_pattern = r'type\s+(\w+)\s+(?:struct|interface)'
        for match in re.finditer(type_pattern, content):
            name = match.group(1)
            metadata.classes.append({
                'name': name,
                'is_exported': name[0].isupper() if name else False
            })
            
        # Basic complexity calculation
        metadata.complexity_score = 1
        metadata.complexity_score += len(re.findall(r'\bif\b', content))
        metadata.complexity_score += len(re.findall(r'\bfor\b', content))
        metadata.complexity_score += len(re.findall(r'\bswitch\b', content))
        metadata.complexity_score += len(re.findall(r'\bselect\b', content))
        
    def tokenize(self, content: str) -> List[str]:
        """Go-aware tokenization"""
        # Remove comments
        content = self._remove_comments(content)
        
        # Go-specific tokenization
        
        token_pattern = r'''
            \b(?:package|import|func|type|struct|interface|var|const|
               if|else|for|range|switch|case|default|select|
               go|chan|defer|return|break|continue|fallthrough|
               map|slice|make|new|append|len|cap|close|delete|panic|recover)\b|
            \b\w+\b|                          # Regular words
            <-|                               # Channel operator
            :=|                               # Short variable declaration
            \.\.\.|                           # Variadic
            [=<>!+\-*/&|^~%]+|               # Operators
            [(){}[\];,.:]|                    # Delimiters
            "[^"]*"|'[^']*'|`[^`]*`          # Strings (including raw strings)
        '''
        
        tokens = re.findall(token_pattern, content, re.VERBOSE)
        return tokens
        
    def extract_structure(self, content: str) -> Dict[str, Any]:
        """Extract detailed code structure"""
        # If parser binary exists, use it
        if self.parser_binary.exists():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as tmp:
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name
                
            try:
                result = subprocess.run(
                    [str(self.parser_binary), tmp_path],
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
        """Remove Go comments while preserving strings"""
        
        # Remove single-line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        
        return content
        
    def _detect_goroutines(self, content: str) -> bool:
        """Detect if code uses goroutines"""
        return bool(re.search(r'\bgo\s+\w+\s*\(', content))
        
    def _detect_channels(self, content: str) -> bool:
        """Detect if code uses channels"""
        return bool(re.search(r'\bchan\b|<-', content))
        
    def _detect_generics(self, content: str) -> bool:
        """Detect if code uses generics (Go 1.18+)"""
        # Look for type parameters in square brackets
        return bool(re.search(r'\[\s*\w+\s+\w+(?:\s*,\s*\w+\s+\w+)*\s*\]', content))


def test_go_parser():
    """Test the Go parser"""
    parser = GoParser()
    
    # Test Go code
    test_code = '''
package main

import (
    "fmt"
    "net/http"
    "github.com/gorilla/mux"
)

// User represents a user in the system
type User struct {
    ID       int    `json:"id"`
    Name     string `json:"name"`
    Email    string `json:"email"`
    IsActive bool   `json:"is_active"`
}

// UserService handles user operations
type UserService interface {
    GetUser(id int) (*User, error)
    CreateUser(user *User) error
    UpdateUser(user *User) error
    DeleteUser(id int) error
}

// userServiceImpl implements UserService
type userServiceImpl struct {
    db Database
}

// NewUserService creates a new user service
func NewUserService(db Database) UserService {
    return &userServiceImpl{db: db}
}

// GetUser retrieves a user by ID
func (s *userServiceImpl) GetUser(id int) (*User, error) {
    // Implementation here
    return nil, nil
}

// HandleUserRequest handles HTTP requests for users
func HandleUserRequest(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id := vars["id"]
    
    switch r.Method {
    case "GET":
        // Handle GET
    case "POST":
        // Handle POST
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

// Generic function example (Go 1.18+)
func Map[T, U any](slice []T, fn func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}
'''
    
    # Parse the code
    metadata = parser.parse(test_code, "main.go")
    
    # Print results
    print("=== Go Parser Test Results ===")
    print(f"Language: {metadata.language}")
    print(f"Package: {metadata.module_path}")
    print(f"Imports: {metadata.imports}")
    print(f"Functions: {len(metadata.functions)}")
    for func in metadata.functions:
        print(f"  - {func['name']} {'(method of ' + func['receiver'] + ')' if func.get('receiver') else ''}")
    print(f"Types: {len(metadata.classes)}")
    for cls in metadata.classes:
        print(f"  - {cls['name']} {'(interface)' if cls.get('type') == 'interface' else ''}")
    print(f"Complexity: {metadata.complexity_score}")
    
    if hasattr(metadata, 'go_features'):
        print(f"Go features: {metadata.go_features}")


if __name__ == "__main__":
    test_go_parser()