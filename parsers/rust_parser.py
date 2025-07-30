"""
Rust AST Parser Integration
===========================

Python wrapper for the Rust AST parser that uses syn crate
to extract detailed metadata from Rust source files.
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
class RustFunction:
    """Represents a Rust function"""
    name: str
    is_pub: bool = False
    is_async: bool = False
    is_unsafe: bool = False
    is_const: bool = False
    params: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    generics: List[str] = field(default_factory=list)
    line_number: int = 0


@dataclass
class RustStruct:
    """Represents a Rust struct"""
    name: str
    is_pub: bool = False
    fields: List[Dict[str, Any]] = field(default_factory=list)
    generics: List[str] = field(default_factory=list)
    derives: List[str] = field(default_factory=list)
    line_number: int = 0


@dataclass
class RustEnum:
    """Represents a Rust enum"""
    name: str
    is_pub: bool = False
    variants: List[Dict[str, Any]] = field(default_factory=list)
    generics: List[str] = field(default_factory=list)
    derives: List[str] = field(default_factory=list)
    line_number: int = 0


@dataclass
class RustTrait:
    """Represents a Rust trait"""
    name: str
    is_pub: bool = False
    methods: List[Dict[str, Any]] = field(default_factory=list)
    generics: List[str] = field(default_factory=list)
    line_number: int = 0


class RustParser(LanguageParser):
    """Rust parser using syn crate"""
    
    def __init__(self):
        self.parser_binary = Path(__file__).parent / "rust_ast_parser"
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check if Rust and the parser binary are available"""
        # Check Rust
        try:
            result = subprocess.run(
                ["rustc", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Rust available: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Rust is not installed or not in PATH")
            logger.info("Install Rust from https://rustup.rs/ or use: brew install rust")
            
        # Check if parser binary exists
        if not self.parser_binary.exists():
            # Try to build it
            cargo_file = Path(__file__).parent / "Cargo.toml"
            if cargo_file.exists():
                try:
                    logger.info("Building Rust parser binary...")
                    subprocess.run(
                        ["cargo", "build", "--release", "--bin", "rust_ast_parser"],
                        cwd=Path(__file__).parent,
                        check=True
                    )
                    # Move the binary to the expected location
                    built_binary = Path(__file__).parent / "target" / "release" / "rust_ast_parser"
                    if built_binary.exists():
                        built_binary.rename(self.parser_binary)
                        logger.info("Rust parser binary built successfully")
                except Exception as e:
                    logger.error(f"Failed to build Rust parser: {e}")
                    
    def parse(self, content: str, filepath: str) -> FileMetadata:
        """Parse Rust content using AST parser"""
        metadata = FileMetadata(
            path=filepath,
            size=len(content),
            language="rust",
            last_modified=time.time(),
            content_hash=hashlib.sha256(content.encode()).hexdigest()
        )
        
        # Determine file type
        path_obj = Path(filepath)
        if path_obj.name == "main.rs":
            metadata.file_type = "main"
        elif path_obj.name == "lib.rs":
            metadata.file_type = "library"
        elif path_obj.name == "mod.rs":
            metadata.file_type = "module"
        elif path_obj.name.endswith("_test.rs") or "test" in str(path_obj.parent):
            metadata.file_type = "test"
        else:
            metadata.file_type = "source"
            
        # If parser binary doesn't exist, fall back to basic parsing
        if not self.parser_binary.exists():
            logger.warning("Rust parser binary not available, using basic parsing")
            self._basic_parse(content, metadata)
        else:
            try:
                # Create temporary file for parsing
                with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as tmp:
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
                        
                        # Extract imports
                        for imp in data.get('imports', []):
                            path = imp['path']
                            # Clean up the path (remove 'use ' and ';')
                            if path.startswith('use '):
                                path = path[4:]
                            if path.endswith(';'):
                                path = path[:-1]
                            path = path.strip()
                            
                            metadata.imports.append(path)
                            
                            # Classify dependencies
                            if path.startswith('crate::') or path.startswith('super::') or path.startswith('self::'):
                                metadata.internal_dependencies.append(path)
                            else:
                                metadata.external_dependencies.append(path)
                                
                            metadata.dependencies.add(path)
                            
                        # Extract functions
                        for func in data.get('functions', []):
                            func_info = {
                                'name': func['name'],
                                'args': [p['name'] for p in func.get('params', [])],
                                'is_pub': func.get('is_pub', False),
                                'is_async': func.get('is_async', False),
                                'is_unsafe': func.get('is_unsafe', False),
                                'is_const': func.get('is_const', False),
                                'line': func.get('line', 0),
                                'return_type': func.get('return_type'),
                                'generics': func.get('generics', [])
                            }
                            metadata.functions.append(func_info)
                            
                        # Extract structs
                        for struct in data.get('structs', []):
                            struct_info = {
                                'name': struct['name'],
                                'type': 'struct',
                                'fields': struct.get('fields', []),
                                'is_pub': struct.get('is_pub', False),
                                'generics': struct.get('generics', []),
                                'derives': struct.get('derives', []),
                                'line': struct.get('line', 0)
                            }
                            metadata.classes.append(struct_info)
                            
                        # Extract enums
                        for enum in data.get('enums', []):
                            enum_info = {
                                'name': enum['name'],
                                'type': 'enum',
                                'variants': enum.get('variants', []),
                                'is_pub': enum.get('is_pub', False),
                                'generics': enum.get('generics', []),
                                'derives': enum.get('derives', []),
                                'line': enum.get('line', 0)
                            }
                            metadata.classes.append(enum_info)
                            
                        # Extract traits
                        for trait in data.get('traits', []):
                            trait_info = {
                                'name': trait['name'],
                                'type': 'trait',
                                'methods': trait.get('methods', []),
                                'is_pub': trait.get('is_pub', False),
                                'generics': trait.get('generics', []),
                                'line': trait.get('line', 0)
                            }
                            metadata.classes.append(trait_info)
                            
                        # Extract impl blocks
                        for impl in data.get('impls', []):
                            # Add methods from impl blocks to functions
                            for method in impl.get('methods', []):
                                method['impl_for'] = impl['for_type']
                                if impl.get('trait_name'):
                                    method['impl_trait'] = impl['trait_name']
                                metadata.functions.append(method)
                                
                        # Set complexity score
                        metadata.complexity_score = data.get('complexity', 1)
                        
                        # Add Rust-specific metadata
                        if not hasattr(metadata, 'rust_features'):
                            metadata.rust_features = {
                                'structs_count': len(data.get('structs', [])),
                                'enums_count': len(data.get('enums', [])),
                                'traits_count': len(data.get('traits', [])),
                                'impls_count': len(data.get('impls', [])),
                                'modules_count': len(data.get('modules', [])),
                                'constants_count': len(data.get('constants', [])),
                                'has_unsafe': self._detect_unsafe(content),
                                'has_async': self._detect_async(content),
                                'has_macros': self._detect_macros(content),
                                'has_generics': self._detect_generics(content)
                            }
                            
                    else:
                        # Parsing failed
                        error_msg = parsed_data.get('error', 'Unknown error')
                        logger.error(f"Rust parsing failed for {filepath}: {error_msg}")
                        self._basic_parse(content, metadata)
                        
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.debug(f"Could not delete temporary file {tmp_path}: {e}")
                        
            except subprocess.TimeoutExpired:
                logger.error(f"Rust parser timeout for {filepath}")
                self._basic_parse(content, metadata)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Rust parser output: {e}")
                self._basic_parse(content, metadata)
            except Exception as e:
                logger.error(f"Rust parsing error for {filepath}: {e}")
                self._basic_parse(content, metadata)
                
        # Always set token count
        metadata.token_count = len(self.tokenize(content))
        
        return metadata
        
    def _basic_parse(self, content: str, metadata: FileMetadata):
        """Basic Rust parsing using regex when AST parser is not available"""
        
        # Extract module declaration
        mod_match = re.search(r'^mod\s+(\w+)', content, re.MULTILINE)
        if mod_match:
            metadata.module_path = mod_match.group(1)
            
        # Extract uses (imports)
        uses = re.findall(r'use\s+([^;]+);', content)
        for use in uses:
            use = use.strip()
            metadata.imports.append(use)
            metadata.dependencies.add(use)
            
            if use.startswith('crate::') or use.startswith('super::') or use.startswith('self::'):
                metadata.internal_dependencies.append(use)
            else:
                metadata.external_dependencies.append(use)
                
        # Extract functions
        func_pattern = r'(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?fn\s+(\w+)'
        for match in re.finditer(func_pattern, content):
            name = match.group(1)
            is_pub = 'pub' in match.group(0)
            metadata.functions.append({
                'name': name,
                'is_pub': is_pub,
                'is_async': 'async' in match.group(0),
                'is_unsafe': 'unsafe' in match.group(0),
                'is_const': 'const' in match.group(0),
                'args': []
            })
            
        # Extract structs
        struct_pattern = r'(?:pub\s+)?struct\s+(\w+)'
        for match in re.finditer(struct_pattern, content):
            name = match.group(1)
            is_pub = 'pub' in match.group(0)
            metadata.classes.append({
                'name': name,
                'type': 'struct',
                'is_pub': is_pub
            })
            
        # Extract enums
        enum_pattern = r'(?:pub\s+)?enum\s+(\w+)'
        for match in re.finditer(enum_pattern, content):
            name = match.group(1)
            is_pub = 'pub' in match.group(0)
            metadata.classes.append({
                'name': name,
                'type': 'enum',
                'is_pub': is_pub
            })
            
        # Extract traits
        trait_pattern = r'(?:pub\s+)?trait\s+(\w+)'
        for match in re.finditer(trait_pattern, content):
            name = match.group(1)
            is_pub = 'pub' in match.group(0)
            metadata.classes.append({
                'name': name,
                'type': 'trait',
                'is_pub': is_pub
            })
            
        # Basic complexity calculation
        metadata.complexity_score = 1
        metadata.complexity_score += len(re.findall(r'\bif\b', content))
        metadata.complexity_score += len(re.findall(r'\bmatch\b', content))
        metadata.complexity_score += len(re.findall(r'\bfor\b', content))
        metadata.complexity_score += len(re.findall(r'\bwhile\b', content))
        metadata.complexity_score += len(re.findall(r'\bloop\b', content))
        
    def tokenize(self, content: str) -> List[str]:
        """Rust-aware tokenization"""
        # Remove comments
        content = self._remove_comments(content)
        
        # Rust-specific tokenization
        
        token_pattern = r'''
            \b(?:fn|struct|enum|trait|impl|mod|use|pub|priv|
               let|mut|const|static|ref|move|
               if|else|match|for|while|loop|return|break|continue|
               async|await|unsafe|extern|crate|self|super|
               where|type|as|in|box|dyn)\b|
            \b\w+\b|                          # Regular words
            ::|                               # Path separator
            ->|                               # Return type
            =>|                               # Match arm
            \?|                               # Try operator
            \.\.|\.\.=|                       # Range operators
            &mut|&|                           # References
            'static|'\w+|                     # Lifetimes
            #\[[\w,\s]*\]|                    # Attributes
            [=<>!+\-*/&|^~%]+|               # Operators
            [(){}[\];,.:]|                    # Delimiters
            r#"[^"]*"#|"[^"]*"|'[^']*'       # Strings (including raw strings)
        '''
        
        tokens = re.findall(token_pattern, content, re.VERBOSE)
        return tokens
        
    def extract_structure(self, content: str) -> Dict[str, Any]:
        """Extract detailed code structure"""
        # If parser binary exists, use it
        if self.parser_binary.exists():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as tmp:
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
        """Remove Rust comments while preserving strings"""
        
        # Remove single-line comments
        content = re.sub(r'//[^\n]*', '', content)
        
        # Remove multi-line comments
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        
        return content
        
    def _detect_unsafe(self, content: str) -> bool:
        """Detect if code uses unsafe blocks"""
        return bool(re.search(r'\bunsafe\s*\{', content))
        
    def _detect_async(self, content: str) -> bool:
        """Detect if code uses async/await"""
        return bool(re.search(r'\basync\b|\bawait\b', content))
        
    def _detect_macros(self, content: str) -> bool:
        """Detect if code uses macros"""
        return bool(re.search(r'\w+!', content))
        
    def _detect_generics(self, content: str) -> bool:
        """Detect if code uses generics"""
        # Look for generic parameters like <T>, <'a, T: Clone>
        return bool(re.search(r"<[^>]*(?:[A-Z]\w*|'\w+)[^>]*>", content))


def test_rust_parser():
    """Test the Rust parser"""
    parser = RustParser()
    
    # Test Rust code
    test_code = '''
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    id: u64,
    name: String,
    email: String,
    role: UserRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserRole {
    Admin,
    User,
    Guest,
}

pub trait UserService {
    fn get_user(&self, id: u64) -> Option<User>;
    fn create_user(&mut self, user: User) -> Result<u64, String>;
    fn update_user(&mut self, user: User) -> Result<(), String>;
    fn delete_user(&mut self, id: u64) -> Result<(), String>;
}

pub struct InMemoryUserService {
    users: Arc<Mutex<HashMap<u64, User>>>,
    next_id: Arc<Mutex<u64>>,
}

impl InMemoryUserService {
    pub fn new() -> Self {
        Self {
            users: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        }
    }
}

impl UserService for InMemoryUserService {
    fn get_user(&self, id: u64) -> Option<User> {
        let users = self.users.lock().unwrap();
        users.get(&id).cloned()
    }
    
    fn create_user(&mut self, mut user: User) -> Result<u64, String> {
        let mut next_id = self.next_id.lock().unwrap();
        let mut users = self.users.lock().unwrap();
        
        user.id = *next_id;
        users.insert(*next_id, user);
        *next_id += 1;
        
        Ok(*next_id - 1)
    }
    
    fn update_user(&mut self, user: User) -> Result<(), String> {
        let mut users = self.users.lock().unwrap();
        match users.get_mut(&user.id) {
            Some(existing) => {
                *existing = user;
                Ok(())
            }
            None => Err("User not found".to_string()),
        }
    }
    
    fn delete_user(&mut self, id: u64) -> Result<(), String> {
        let mut users = self.users.lock().unwrap();
        match users.remove(&id) {
            Some(_) => Ok(()),
            None => Err("User not found".to_string()),
        }
    }
}

pub async fn handle_user_request(id: u64) -> Result<User, String> {
    // Async handler
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(User {
        id,
        name: "Test User".to_string(),
        email: "test@example.com".to_string(),
        role: UserRole::User,
    })
}

// Generic function example
fn find_max<T: PartialOrd>(slice: &[T]) -> Option<&T> {
    slice.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
}

// Macro example
macro_rules! create_function {
    ($func_name:ident) => {
        fn $func_name() {
            println!("Function {:?} was called", stringify!($func_name));
        }
    };
}

create_function!(foo);
create_function!(bar);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_creation() {
        let mut service = InMemoryUserService::new();
        let user = User {
            id: 0,
            name: "Test".to_string(),
            email: "test@test.com".to_string(),
            role: UserRole::User,
        };
        
        let id = service.create_user(user).unwrap();
        assert_eq!(id, 1);
    }
}
'''
    
    # Parse the code
    metadata = parser.parse(test_code, "lib.rs")
    
    # Print results
    print("=== Rust Parser Test Results ===")
    print(f"Language: {metadata.language}")
    print(f"Imports: {metadata.imports}")
    print(f"Functions: {len(metadata.functions)}")
    for func in metadata.functions[:10]:
        name = func['name']
        if func.get('impl_for'):
            name = f"{func['impl_for']}::{name}"
        pub = " [pub]" if func.get('is_pub') else ""
        async_str = " [async]" if func.get('is_async') else ""
        unsafe_str = " [unsafe]" if func.get('is_unsafe') else ""
        print(f"  - {name}{pub}{async_str}{unsafe_str}")
    print(f"Types: {len(metadata.classes)}")
    for cls in metadata.classes:
        pub = " [pub]" if cls.get('is_pub') else ""
        print(f"  - {cls['name']} ({cls.get('type', 'unknown')}){pub}")
    print(f"Complexity: {metadata.complexity_score}")
    
    if hasattr(metadata, 'rust_features'):
        print(f"Rust features: {metadata.rust_features}")


if __name__ == "__main__":
    test_rust_parser()