"""
Security and Input Validation Module
====================================

Comprehensive security measures including input validation,
path traversal protection, and secure processing practices.
"""

import os
import re
import hashlib
import secrets
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Set
from dataclasses import dataclass
import logging
import json
import pickle
import ast
from urllib.parse import urlparse
try:
    import magic  # python-magic for file type detection
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not available. Install with: pip install python-magic")

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    # Path restrictions
    allowed_base_paths: List[Path] = None
    forbidden_paths: List[str] = None
    max_path_depth: int = 20
    
    # File restrictions
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: Set[str] = None
    forbidden_extensions: Set[str] = None
    check_file_magic: bool = True
    
    # Content restrictions
    max_line_length: int = 10000
    max_lines_per_file: int = 100000
    allow_binary_files: bool = False
    
    # Processing restrictions
    max_files_per_batch: int = 1000
    max_total_files: int = 100000
    timeout_per_file: float = 30.0
    
    # Security features
    enable_sandboxing: bool = True
    enable_content_scanning: bool = True
    enable_path_normalization: bool = True
    
    def __post_init__(self):
        if self.allowed_base_paths is None:
            self.allowed_base_paths = [Path.cwd()]
        
        if self.forbidden_paths is None:
            self.forbidden_paths = [
                '/etc', '/sys', '/proc', '/dev',
                '/root', '/boot', '/var/log',
                '/.git', '/.ssh', '/.aws',
                '/private', '/secrets'
            ]
            
        if self.allowed_extensions is None:
            self.allowed_extensions = {
                '.py', '.js', '.ts', '.jsx', '.tsx',
                '.java', '.cpp', '.c', '.h', '.hpp',
                '.go', '.rs', '.rb', '.php', '.swift',
                '.json', '.yaml', '.yml', '.xml',
                '.md', '.txt', '.rst', '.csv'
            }
            
        if self.forbidden_extensions is None:
            self.forbidden_extensions = {
                '.exe', '.dll', '.so', '.dylib',
                '.bin', '.dat', '.db', '.sqlite',
                '.key', '.pem', '.p12', '.pfx',
                '.env', '.secret', '.password'
            }


class PathValidator:
    """Validates and sanitizes file paths"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
    def validate_path(self, path: Union[str, Path], 
                     base_path: Optional[Path] = None) -> Optional[Path]:
        """Validate and sanitize a file path"""
        try:
            # Convert to Path object
            path = Path(path)
            
            # Normalize and resolve path
            if self.config.enable_path_normalization:
                path = path.resolve()
            
            # Check for null bytes (security risk)
            if '\x00' in str(path):
                logger.error(f"Path contains null bytes: {path}")
                return None
            
            # Check path depth
            if len(path.parts) > self.config.max_path_depth:
                logger.error(f"Path too deep: {path}")
                return None
            
            # Check forbidden paths
            # Check if any path component matches forbidden paths
            path_parts = [part.lower() for part in path.parts]
            for forbidden in self.config.forbidden_paths:
                forbidden_clean = forbidden.lower().strip('/')
                if forbidden_clean in path_parts:
                    logger.error(f"Path contains forbidden segment: {forbidden}")
                    return None
            
            # Check if path is within allowed base paths
            if base_path:
                try:
                    path.relative_to(base_path)
                except ValueError:
                    logger.error(f"Path outside base directory: {path}")
                    return None
            else:
                # Check against all allowed base paths
                is_allowed = False
                for allowed_base in self.config.allowed_base_paths:
                    try:
                        path.relative_to(allowed_base)
                        is_allowed = True
                        break
                    except ValueError:
                        continue
                        
                if not is_allowed:
                    logger.error(f"Path not in allowed directories: {path}")
                    return None
            
            # Check for directory traversal attempts
            if '..' in path.parts:
                logger.error(f"Directory traversal detected: {path}")
                return None
            
            # Additional checks for symbolic links
            if path.exists() and path.is_symlink():
                target = path.resolve()
                # Recursively validate the target
                validated_target = self.validate_path(target, base_path)
                if not validated_target:
                    logger.error(f"Symlink target validation failed: {path} -> {target}")
                    return None
            
            return path
            
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return None


class FileValidator:
    """Validates file properties and content"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._magic = None
        if config.check_file_magic and MAGIC_AVAILABLE:
            try:
                self._magic = magic.Magic(mime=True)
            except Exception as e:
                logger.warning(f"python-magic not available: {e}")
                self._magic = None
                
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Comprehensive file validation"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_type': None,
            'size': 0
        }
        
        try:
            # Check file existence
            if not file_path.exists():
                result['valid'] = False
                result['errors'].append("File does not exist")
                return result
                
            if not file_path.is_file():
                result['valid'] = False
                result['errors'].append("Path is not a file")
                return result
            
            # Check file size
            file_size = file_path.stat().st_size
            result['size'] = file_size
            
            if file_size > self.config.max_file_size:
                result['valid'] = False
                result['errors'].append(f"File too large: {file_size} bytes")
                return result
                
            if file_size == 0:
                result['warnings'].append("Empty file")
            
            # Check file extension
            extension = file_path.suffix.lower()
            
            if self.config.forbidden_extensions and extension in self.config.forbidden_extensions:
                result['valid'] = False
                result['errors'].append(f"Forbidden file extension: {extension}")
                return result
                
            if self.config.allowed_extensions and extension not in self.config.allowed_extensions:
                result['valid'] = False
                result['errors'].append(f"File extension not allowed: {extension}")
                return result
            
            # Check file type using magic numbers
            if self._magic and self.config.check_file_magic:
                try:
                    mime_type = self._magic.from_file(str(file_path))
                    result['file_type'] = mime_type
                    
                    # Check for executable or dangerous types
                    dangerous_types = [
                        'application/x-executable',
                        'application/x-dosexec',
                        'application/x-mach-binary'
                    ]
                    
                    if any(dangerous in mime_type for dangerous in dangerous_types):
                        result['valid'] = False
                        result['errors'].append(f"Dangerous file type: {mime_type}")
                        return result
                        
                except Exception as e:
                    result['warnings'].append(f"Could not determine file type: {e}")
            
            # Check file permissions (Unix-like systems)
            if hasattr(os, 'access'):
                if not os.access(file_path, os.R_OK):
                    result['valid'] = False
                    result['errors'].append("File not readable")
                    return result
            
            return result
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {e}")
            return result


class ContentScanner:
    """Scans file content for security issues"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Patterns for sensitive data
        self.sensitive_patterns = [
            # API Keys
            (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?[a-zA-Z0-9]{16,}["\']?', 'API Key'),
            # AWS Keys
            (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
            (r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9/+=]{40}["\']?', 'AWS Secret Key'),
            # Private Keys
            (r'-----BEGIN (RSA |EC )?PRIVATE KEY-----', 'Private Key'),
            # Passwords
            (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?[^\s"\']{8,}["\']?', 'Password'),
            # Database URLs
            (r'(?i)(mysql|postgresql|mongodb)://[^:]+:[^@]+@[^/]+', 'Database URL with credentials'),
            # JWT Tokens
            (r'eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+', 'JWT Token'),
        ]
        
        # Patterns for malicious code
        self.malicious_patterns = [
            # Command injection attempts
            (r'(?i)(eval|exec|system|passthru|shell_exec)\s*\(', 'Code execution'),
            # SQL injection patterns
            (r'(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)', 'SQL injection pattern'),
            # Path traversal
            (r'\.\./', 'Path traversal'),
            # Script injection
            (r'<script[^>]*>', 'Script tag'),
        ]
        
    def scan_content(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Scan content for security issues"""
        result = {
            'safe': True,
            'issues': [],
            'sensitive_data': [],
            'suspicious_patterns': []
        }
        
        if not self.config.enable_content_scanning:
            return result
            
        try:
            # Check line length and count
            lines = content.split('\n')
            
            if len(lines) > self.config.max_lines_per_file:
                result['issues'].append(f"Too many lines: {len(lines)}")
                result['safe'] = False
                
            for i, line in enumerate(lines[:1000]):  # Check first 1000 lines
                if len(line) > self.config.max_line_length:
                    result['issues'].append(f"Line {i+1} too long: {len(line)} chars")
                    
                # Check for sensitive data
                for pattern, description in self.sensitive_patterns:
                    if re.search(pattern, line):
                        result['sensitive_data'].append({
                            'line': i + 1,
                            'type': description,
                            'file': str(file_path)
                        })
                        
                # Check for malicious patterns
                for pattern, description in self.malicious_patterns:
                    if re.search(pattern, line):
                        result['suspicious_patterns'].append({
                            'line': i + 1,
                            'type': description,
                            'file': str(file_path)
                        })
            
            # Mark as unsafe if sensitive data found
            if result['sensitive_data']:
                result['safe'] = False
                result['issues'].append(f"Found {len(result['sensitive_data'])} sensitive data patterns")
                
            return result
            
        except Exception as e:
            result['safe'] = False
            result['issues'].append(f"Content scanning error: {e}")
            return result


class SecureProcessor:
    """Secure processing utilities"""
    
    @staticmethod
    def safe_json_loads(data: str) -> Any:
        """Safely parse JSON with size limits"""
        if len(data) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("JSON data too large")
            
        try:
            # Use object_hook to prevent deep nesting attacks
            def check_depth(obj, depth=0):
                if depth > 100:
                    raise ValueError("JSON nesting too deep")
                if isinstance(obj, dict):
                    for v in obj.values():
                        check_depth(v, depth + 1)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, depth + 1)
                return obj
                
            return json.loads(data, object_hook=lambda obj: check_depth(obj))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
            
    @staticmethod
    def safe_pickle_load(file_path: Path) -> Any:
        """Safely unpickle with restricted imports"""
        
        class RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Only allow safe modules
                ALLOWED_MODULES = {
                    'builtins',
                    'collections',
                    'datetime',
                    'numpy',
                    'pandas'
                }
                
                if module not in ALLOWED_MODULES:
                    raise pickle.UnpicklingError(f"Forbidden module: {module}")
                    
                return super().find_class(module, name)
                
        try:
            with open(file_path, 'rb') as f:
                return RestrictedUnpickler(f).load()
        except Exception as e:
            raise ValueError(f"Unsafe pickle: {e}")
            
    @staticmethod
    def safe_ast_parse(code: str) -> ast.AST:
        """Safely parse Python code with restrictions"""
        try:
            tree = ast.parse(code)
            
            # Check for dangerous constructs
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Check for dangerous imports
                    dangerous_modules = {'os', 'subprocess', 'sys', '__builtin__'}
                    
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in dangerous_modules:
                                raise ValueError(f"Dangerous import: {alias.name}")
                    elif node.module and node.module in dangerous_modules:
                        raise ValueError(f"Dangerous import: {node.module}")
                        
            return tree
            
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")


class SecurityValidator:
    """Main security validation orchestrator"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.path_validator = PathValidator(self.config)
        self.file_validator = FileValidator(self.config)
        self.content_scanner = ContentScanner(self.config)
        
    def validate_input(self, 
                      file_path: Union[str, Path],
                      base_path: Optional[Path] = None,
                      scan_content: bool = True) -> Dict[str, Any]:
        """Comprehensive input validation"""
        result = {
            'valid': True,
            'path': None,
            'errors': [],
            'warnings': [],
            'security_issues': []
        }
        
        # Validate path
        validated_path = self.path_validator.validate_path(file_path, base_path)
        if not validated_path:
            result['valid'] = False
            result['errors'].append("Path validation failed")
            return result
            
        result['path'] = validated_path
        
        # Validate file
        file_validation = self.file_validator.validate_file(validated_path)
        if not file_validation['valid']:
            result['valid'] = False
            result['errors'].extend(file_validation['errors'])
            return result
            
        result['warnings'].extend(file_validation['warnings'])
        
        # Scan content if requested
        if scan_content and validated_path.suffix in self.config.allowed_extensions:
            try:
                with open(validated_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                scan_result = self.content_scanner.scan_content(content, validated_path)
                
                if not scan_result['safe']:
                    result['warnings'].append("Content security issues detected")
                    result['security_issues'] = scan_result['issues']
                    
                if scan_result['sensitive_data']:
                    result['warnings'].append(f"Found {len(scan_result['sensitive_data'])} sensitive data patterns")
                    
            except Exception as e:
                result['warnings'].append(f"Could not scan content: {e}")
                
        return result
        
    def create_sandbox_config(self) -> Dict[str, Any]:
        """Create configuration for sandboxed execution"""
        return {
            'allowed_paths': [str(p) for p in self.config.allowed_base_paths],
            'forbidden_paths': self.config.forbidden_paths,
            'resource_limits': {
                'max_memory': 512 * 1024 * 1024,  # 512MB
                'max_cpu_time': 30,  # seconds
                'max_file_size': self.config.max_file_size,
                'max_open_files': 100
            },
            'network_access': False,
            'allow_exec': False
        }