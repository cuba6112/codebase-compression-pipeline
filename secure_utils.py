"""
Secure Utilities Module
=======================

Additional security utilities for safe file operations,
temporary file handling, and cryptographic operations.
"""

import os
import tempfile
import secrets
import hashlib
import hmac
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
from contextlib import contextmanager
import logging
import json
import shutil

logger = logging.getLogger(__name__)


class SecureFileHandler:
    """Secure file operations with atomic writes and integrity checks"""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.gettempdir())
        # Ensure temp directory has proper permissions
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(os, 'chmod'):
            os.chmod(self.temp_dir, 0o700)
    
    @contextmanager
    def secure_temp_file(self, prefix: str = 'sec_', suffix: str = '.tmp') -> Tuple[int, Path]:
        """Create a secure temporary file with restricted permissions"""
        fd = None
        temp_path = None
        
        try:
            # Create secure temp file
            fd, temp_path_str = tempfile.mkstemp(
                dir=str(self.temp_dir),
                prefix=prefix,
                suffix=suffix
            )
            temp_path = Path(temp_path_str)
            
            # Set restrictive permissions (owner read/write only)
            if hasattr(os, 'chmod'):
                os.chmod(temp_path, 0o600)
            
            yield fd, temp_path
            
        finally:
            # Cleanup
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                    
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError as e:
                    logger.warning(f"Failed to remove temp file {temp_path}: {e}")
    
    def atomic_write(self, target_path: Path, content: bytes, 
                    use_integrity_check: bool = True) -> bool:
        """Write file atomically with optional integrity check"""
        target_path = Path(target_path)
        temp_path = target_path.with_suffix(f'.tmp.{secrets.token_hex(8)}')
        
        try:
            # Write to temporary file
            with open(temp_path, 'wb') as f:
                if use_integrity_check:
                    # Add integrity check (HMAC with a key - in production use proper key management)
                    integrity_key = b'temporary_key'  # TODO: Use proper key management
                    mac = hmac.new(integrity_key, content, hashlib.sha256).digest()
                    f.write(mac)
                f.write(content)
            
            # Set proper permissions
            if hasattr(os, 'chmod'):
                os.chmod(temp_path, 0o644)
            
            # Atomic rename
            temp_path.replace(target_path)
            return True
            
        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
            # Cleanup on failure
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return False
    
    def verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity using stored HMAC"""
        try:
            with open(file_path, 'rb') as f:
                stored_mac = f.read(32)  # SHA-256 HMAC is 32 bytes
                content = f.read()
                
            # Verify HMAC
            integrity_key = b'temporary_key'  # TODO: Use proper key management
            expected_mac = hmac.new(integrity_key, content, hashlib.sha256).digest()
            
            return hmac.compare_digest(stored_mac, expected_mac)
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False


class SecureSerializer:
    """Secure serialization with integrity and type checking"""
    
    @staticmethod
    def serialize_json(data: Any, include_checksum: bool = True) -> bytes:
        """Serialize data to JSON with optional checksum"""
        # Ensure data is JSON-serializable
        json_str = json.dumps(data, ensure_ascii=True, sort_keys=True)
        
        if include_checksum:
            # Add checksum for integrity
            checksum = hashlib.sha256(json_str.encode()).hexdigest()
            wrapped_data = {
                'data': data,
                'checksum': checksum,
                'version': '1.0'
            }
            return json.dumps(wrapped_data, ensure_ascii=True).encode()
        
        return json_str.encode()
    
    @staticmethod
    def deserialize_json(data: bytes, verify_checksum: bool = True) -> Any:
        """Deserialize JSON with optional checksum verification"""
        try:
            decoded = json.loads(data.decode('utf-8'))
            
            if verify_checksum and isinstance(decoded, dict) and 'checksum' in decoded:
                # Verify checksum
                data_str = json.dumps(decoded['data'], ensure_ascii=True, sort_keys=True)
                expected_checksum = hashlib.sha256(data_str.encode()).hexdigest()
                
                if not hmac.compare_digest(decoded['checksum'], expected_checksum):
                    raise ValueError("Checksum verification failed")
                
                return decoded['data']
            
            return decoded
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Deserialization failed: {e}")


class SecurePath:
    """Secure path operations with validation"""
    
    @staticmethod
    def join_safe(*parts: str) -> Path:
        """Safely join path parts preventing traversal"""
        # Remove any parent directory references
        safe_parts = []
        for part in parts:
            # Remove leading/trailing slashes and parent refs
            cleaned = str(part).strip('/\\').replace('..', '')
            if cleaned and cleaned != '.':
                safe_parts.append(cleaned)
        
        if not safe_parts:
            raise ValueError("No valid path parts provided")
        
        return Path(*safe_parts)
    
    @staticmethod
    def is_safe_filename(filename: str) -> bool:
        """Check if filename is safe (no special characters, reasonable length)"""
        # Allow only alphanumeric, dash, underscore, and dot
        import re
        
        if not filename or len(filename) > 255:
            return False
        
        # Check for safe characters only
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            return False
        
        # Prevent special filenames
        if filename.lower() in {'con', 'prn', 'aux', 'nul', 'com1', 'lpt1'}:
            return False
        
        # Prevent hidden files
        if filename.startswith('.'):
            return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to make it safe"""
        import re
        
        # Replace unsafe characters with underscore
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limit length
        if len(safe_name) > 200:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:200-len(ext)] + ext
        
        # Ensure not empty
        if not safe_name:
            safe_name = f"file_{secrets.token_hex(8)}"
        
        return safe_name


def secure_hash_file(file_path: Path, algorithm: str = 'sha256') -> str:
    """Compute secure hash of file contents"""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(65536), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token"""
    return secrets.token_urlsafe(length)


def constant_time_compare(a: str, b: str) -> bool:
    """Compare strings in constant time to prevent timing attacks"""
    return hmac.compare_digest(a, b)


@contextmanager
def secure_working_directory(target_dir: Path):
    """Context manager for safely changing working directory"""
    original_dir = Path.cwd()
    
    try:
        os.chdir(target_dir)
        yield target_dir
    finally:
        os.chdir(original_dir)


class AuditLogger:
    """Security audit logging"""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("security_audit.log")
        
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = 'INFO'):
        """Log security-relevant events"""
        import datetime
        
        event = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'process_id': os.getpid()
        }
        
        # Log to file (append mode)
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def log_access_attempt(self, path: str, allowed: bool, reason: str = ""):
        """Log file access attempts"""
        self.log_security_event(
            'file_access',
            {
                'path': str(path),
                'allowed': allowed,
                'reason': reason
            },
            severity='WARNING' if not allowed else 'INFO'
        )
    
    def log_validation_failure(self, validation_type: str, details: str):
        """Log validation failures"""
        self.log_security_event(
            'validation_failure',
            {
                'type': validation_type,
                'details': details
            },
            severity='WARNING'
        )