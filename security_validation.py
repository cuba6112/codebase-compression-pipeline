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
from typing import Optional, List, Dict, Any, Union, Set, Tuple
from dataclasses import dataclass
import logging
import json
import pickle
import ast
from urllib.parse import urlparse
import unicodedata
import sys
import tempfile
import time
import resource
import hmac
from collections import deque
import uuid
import psutil
import io
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
    max_memory_usage: int = 512 * 1024 * 1024  # 512MB
    max_cpu_time: int = 30  # seconds
    max_open_files: int = 1024
    
    # Rate limiting
    rate_limit_calls: int = 100
    rate_limit_window: float = 60.0  # seconds
    
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
                     base_path: Optional[Path] = None,
                     must_exist: bool = True,
                     allow_creation: bool = False) -> Optional[Path]:
        """Validate and sanitize a file path
        
        Args:
            path: Path to validate
            base_path: Optional base path for relative validation
            must_exist: Whether the path must already exist
            allow_creation: Whether to allow creation of non-existing directories
            
        Returns:
            Validated Path object or None if validation fails
        """
        try:
            # Convert to Path object
            path = Path(path)
            
            # Unicode normalization to prevent homograph attacks
            path_str = unicodedata.normalize('NFKC', str(path))
            if path_str != str(path):
                logger.warning(f"Path required Unicode normalization: {path} -> {path_str}")
            path = Path(path_str)
            
            # Check for non-ASCII characters
            if not path_str.isascii():
                logger.warning(f"Non-ASCII characters in path: {path}")
            
            # Normalize and resolve path
            if self.config.enable_path_normalization:
                # Reject symlinks by default
                if path.exists() and path.is_symlink():
                    logger.error(f"Symbolic links not allowed by default: {path}")
                    return None

                # Use strict resolution in Python 3.10+
                if sys.version_info >= (3, 10):
                    try:
                        path = path.resolve(strict=must_exist and not allow_creation)
                    except (FileNotFoundError, RuntimeError) as e:
                        if must_exist and not allow_creation:
                            logger.error(f"Path resolution failed: {e}")
                            return None
                        else:
                            # Resolve without strict mode when path creation is allowed
                            logger.info(f"Path does not exist but creation allowed: {path}")
                            path = path.resolve()
                else:
                    path = path.resolve()
                    if must_exist and not allow_creation and not path.exists():
                        logger.error(f"Path does not exist: {path}")
                        return None
            
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
            
            # Double-check after resolution - no symlinks allowed
            if path.is_symlink():
                logger.error(f"Path resolved to symlink: {path}")
                return None
                
            # Check for case sensitivity issues
            if not self._check_case_sensitivity(path):
                logger.error(f"Case sensitivity validation failed: {path}")
                return None
            
            return path
            
        except ValueError as e:
            logger.error(f"Invalid path value: {e}")
            return None
        except OSError as e:
            logger.error(f"OS error validating path: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected path validation error: {e}", exc_info=True)
            return None
    
    def validate_directory_path(self, path: Union[str, Path], 
                               create_if_missing: bool = False,
                               base_path: Optional[Path] = None) -> Optional[Path]:
        """Validate a directory path with creation support
        
        Args:
            path: Directory path to validate
            create_if_missing: Whether to create the directory if it doesn't exist
            base_path: Optional base path for relative validation
            
        Returns:
            Validated Path object or None if validation fails
        """
        # Validate the path allowing creation
        validated_path = self.validate_path(
            path, 
            base_path=base_path, 
            must_exist=False, 
            allow_creation=True
        )
        
        if not validated_path:
            return None
            
        try:
            # Check if it exists and is a directory
            if validated_path.exists():
                if not validated_path.is_dir():
                    logger.error(f"Path exists but is not a directory: {validated_path}")
                    return None
            elif create_if_missing:
                # Create the directory with secure permissions
                validated_path.mkdir(parents=True, exist_ok=True, mode=0o750)
                logger.info(f"Created directory: {validated_path}")
            else:
                logger.warning(f"Directory does not exist: {validated_path}")
                
            return validated_path
            
        except PermissionError as e:
            logger.error(f"Permission denied creating directory {validated_path}: {e}")
            return None
        except OSError as e:
            logger.error(f"OS error with directory {validated_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with directory {validated_path}: {e}", exc_info=True)
            return None
    
    def _check_case_sensitivity(self, path: Path) -> bool:
        """Check for case sensitivity issues"""
        try:
            # Create a temporary file to test filesystem case sensitivity
            test_file = Path(tempfile.gettempdir()) / f'TeSt_CaSe_{uuid.uuid4()}.tmp'
            test_file.touch()
            case_sensitive = not (test_file.parent / test_file.name.lower()).exists()
            test_file.unlink()
            
            if not case_sensitive:
                # On case-insensitive systems, check for confusing names
                path_lower = str(path).lower()
                for forbidden in self.config.forbidden_paths:
                    if forbidden.lower() in path_lower:
                        return False
                        
            return True
        except Exception as e:
            logger.error(f"Case sensitivity check failed: {e}")
            return False


class FileValidator:
    """Validates file properties and content"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._magic = None
        if config.check_file_magic and MAGIC_AVAILABLE:
            try:
                self._magic = magic.Magic(mime=True)
            except ImportError as e:
                logger.warning(f"python-magic module not available: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error initializing magic: {e}", exc_info=True)
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
                        
                except OSError as e:
                    result['warnings'].append(f"OS error determining file type: {e}")
                except Exception as e:
                    result['warnings'].append(f"Unexpected error determining file type: {e}")
            
            # Check file permissions (Unix-like systems)
            if hasattr(os, 'access'):
                if not os.access(file_path, os.R_OK):
                    result['valid'] = False
                    result['errors'].append("File not readable")
                    return result
            
            return result
            
        except PermissionError as e:
            result['valid'] = False
            result['errors'].append(f"Permission error during validation: {e}")
            return result
        except OSError as e:
            result['valid'] = False
            result['errors'].append(f"OS error during validation: {e}")
            return result
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Unexpected validation error: {e}")
            logger.error(f"Unexpected error validating {file_path}: {e}", exc_info=True)
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
            
        except UnicodeDecodeError as e:
            result['safe'] = False
            result['issues'].append(f"Encoding error during content scan: {e}")
            return result
        except re.error as e:
            result['safe'] = False
            result['issues'].append(f"Regex error during content scan: {e}")
            return result
        except Exception as e:
            result['safe'] = False
            result['issues'].append(f"Unexpected content scanning error: {e}")
            logger.error(f"Unexpected error scanning content: {e}", exc_info=True)
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
            ALLOWED_CLASSES = {
                ('builtins', 'dict'),
                ('builtins', 'list'),
                ('builtins', 'tuple'),
                ('builtins', 'set'),
                ('builtins', 'str'),
                ('builtins', 'int'),
                ('builtins', 'float'),
                ('builtins', 'bool'),
                ('builtins', 'bytes'),
                ('builtins', 'bytearray'),
                ('collections', 'OrderedDict'),
                ('collections', 'defaultdict'),
                ('datetime', 'datetime'),
                ('datetime', 'date'),
                ('datetime', 'time'),
                ('datetime', 'timedelta'),
            }
            
            def find_class(self, module, name):
                if (module, name) not in self.ALLOWED_CLASSES:
                    raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")
                return super().find_class(module, name)
                
            def persistent_load(self, pid):
                raise pickle.UnpicklingError("Persistent objects are forbidden")
                
        try:
            # Verify file size before loading
            if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB limit
                raise ValueError("Pickle file too large")
                
            # Load with integrity check if available
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Check for HMAC if present (first 32 bytes)
            if len(data) > 32:
                # This is just a placeholder - in production, use a proper key
                expected_hmac = data[:32]
                actual_data = data[32:]
                # Verify HMAC here if implemented
            else:
                actual_data = data
                
            return RestrictedUnpickler(io.BytesIO(actual_data)).load()
        except pickle.UnpicklingError as e:
            raise ValueError(f"Unsafe pickle - unpickling error: {e}")
        except (OSError, IOError) as e:
            raise ValueError(f"Unsafe pickle - I/O error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading pickle: {e}", exc_info=True)
            raise ValueError(f"Unsafe pickle - unexpected error: {e}")
            
    @staticmethod
    def safe_ast_parse(code: str) -> ast.AST:
        """Safely parse Python code with restrictions"""
        try:
            # First check for null bytes
            if '\x00' in code:
                raise ValueError("Code contains null bytes")
                
            tree = ast.parse(code)
            
            # Use comprehensive visitor
            visitor = SecurityASTVisitor()
            visitor.visit(tree)
            
            if visitor.violations:
                raise ValueError(f"Security violations found: {'; '.join(visitor.violations)}")
                
            return tree
            
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")


class SecurityASTVisitor(ast.NodeVisitor):
    """Comprehensive AST visitor for security checks"""
    
    def __init__(self):
        self.violations = []
        
    def visit_Call(self, node):
        # Check for dangerous function calls
        dangerous_funcs = {
            'eval', 'exec', 'compile', '__import__', 'open',
            'input', 'globals', 'locals', 'vars', 'dir',
            'getattr', 'setattr', 'delattr', 'hasattr'
        }
        
        if isinstance(node.func, ast.Name) and node.func.id in dangerous_funcs:
            self.violations.append(f"Dangerous function call: {node.func.id}")
            
        # Check for subprocess calls
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in {'subprocess', 'os', 'sys'}:
                    self.violations.append(f"Dangerous module method: {node.func.value.id}.{node.func.attr}")
                    
        self.generic_visit(node)
        
    def visit_Import(self, node):
        dangerous_modules = {
            'os', 'subprocess', 'sys', 'importlib', '__builtin__',
            'builtins', 'imp', 'importlib', 'runpy', 'pkgutil'
        }
        
        for alias in node.names:
            if alias.name in dangerous_modules:
                self.violations.append(f"Dangerous import: {alias.name}")
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        dangerous_modules = {
            'os', 'subprocess', 'sys', 'importlib', '__builtin__',
            'builtins', 'imp', 'importlib', 'runpy', 'pkgutil'
        }
        
        if node.module and node.module in dangerous_modules:
            self.violations.append(f"Dangerous import from: {node.module}")
        self.generic_visit(node)


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        
    def allow_request(self) -> bool:
        now = time.time()
        
        # Remove old calls outside the time window
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()
            
        if len(self.calls) >= self.max_calls:
            return False
            
        self.calls.append(now)
        return True


class SecurityValidator:
    """Main security validation orchestrator"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.path_validator = PathValidator(self.config)
        self.file_validator = FileValidator(self.config)
        self.content_scanner = ContentScanner(self.config)
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_calls,
            self.config.rate_limit_window
        )
        self._set_resource_limits()
        
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
                    
            except (OSError, IOError) as e:
                result['warnings'].append(f"I/O error scanning content: {e}")
            except UnicodeDecodeError as e:
                result['warnings'].append(f"Encoding error scanning content: {e}")
            except Exception as e:
                result['warnings'].append(f"Unexpected error scanning content: {e}")
                logger.warning(f"Unexpected error scanning content: {e}", exc_info=True)
                
        return result
        
    def _set_resource_limits(self):
        """Set resource limits for the process"""
        try:
            # Limit file descriptors
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (min(self.config.max_open_files, soft), hard)
            )
            
            # Limit memory usage (if available)
            if hasattr(resource, 'RLIMIT_AS'):
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.config.max_memory_usage, -1)
                )
                
            # Limit CPU time
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.config.max_cpu_time, self.config.max_cpu_time * 2)
            )
            
            logger.info("Resource limits set successfully")
        except Exception as e:
            logger.warning(f"Could not set resource limits: {e}")
    
    def create_sandbox_config(self) -> Dict[str, Any]:
        """Create configuration for sandboxed execution"""
        return {
            'allowed_paths': [str(p) for p in self.config.allowed_base_paths],
            'forbidden_paths': self.config.forbidden_paths,
            'resource_limits': {
                'max_memory': self.config.max_memory_usage,
                'max_cpu_time': self.config.max_cpu_time,
                'max_file_size': self.config.max_file_size,
                'max_open_files': self.config.max_open_files
            },
            'network_access': False,
            'allow_exec': False,
            'rate_limits': {
                'max_calls': self.config.rate_limit_calls,
                'time_window': self.config.rate_limit_window
            }
        }
    
    def sanitize_error_message(self, error: Exception, internal: bool = False) -> str:
        """Sanitize error messages to prevent information leakage"""
        if internal:
            return str(error)
            
        # Map to generic messages for external consumption
        error_map = {
            FileNotFoundError: "Requested resource not found",
            PermissionError: "Access denied",
            ValueError: "Invalid input provided",
            OSError: "System error occurred",
            IOError: "I/O operation failed",
            MemoryError: "Insufficient resources",
            TimeoutError: "Operation timed out"
        }
        
        return error_map.get(type(error), "An error occurred")
    
    def check_resource_usage(self) -> Dict[str, Any]:
        """Check current resource usage"""
        try:
            process = psutil.Process()
            return {
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(interval=0.1),
                'open_files': len(process.open_files()),
                'num_threads': process.num_threads(),
                'memory_info': process.memory_info()._asdict()
            }
        except Exception as e:
            logger.error(f"Error checking resource usage: {e}")
            return {}