#!/usr/bin/env python3
"""
Unit tests for security validation components.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from security_validation import (
    SecurityConfig, SecurityValidator, PathValidator, 
    FileValidator, ContentScanner, SecureProcessor
)


class TestSecurityConfig:
    """Tests for SecurityConfig"""
    
    def test_default_configuration(self):
        """Test default security configuration"""
        config = SecurityConfig()
        
        # Check defaults
        assert config.max_file_size == 100 * 1024 * 1024  # 100MB
        assert config.max_path_depth == 20
        assert config.enable_content_scanning is True
        assert config.enable_path_normalization is True
        
        # Should have default forbidden paths
        assert '/etc' in config.forbidden_paths
        assert '/sys' in config.forbidden_paths
        assert '/.git' in config.forbidden_paths
        
        # Should have default allowed extensions
        assert '.py' in config.allowed_extensions
        assert '.js' in config.allowed_extensions
        assert '.exe' in config.forbidden_extensions


class TestPathValidator:
    """Tests for PathValidator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = SecurityConfig()
        self.validator = PathValidator(self.config)
    
    def test_basic_path_validation(self):
        """Test basic path validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Update config to allow temp directory
            self.config.allowed_base_paths = [temp_path]
            
            # Valid path should pass
            test_file = temp_path / "test.txt"
            test_file.touch()
            
            validated = self.validator.validate_path(test_file)
            assert validated is not None
            assert validated == test_file.resolve()
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.config.allowed_base_paths = [temp_path]
            
            # Path with .. should be rejected
            malicious_path = temp_path / ".." / "etc" / "passwd"
            validated = self.validator.validate_path(malicious_path)
            assert validated is None
    
    def test_null_byte_injection(self):
        """Test rejection of paths with null bytes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.config.allowed_base_paths = [temp_path]
            
            # Path with null byte should be rejected
            malicious_path = str(temp_path / "test.txt") + "\x00.exe"
            validated = self.validator.validate_path(malicious_path)
            assert validated is None
    
    def test_path_depth_limits(self):
        """Test path depth limitations"""
        self.config.max_path_depth = 3
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.config.allowed_base_paths = [temp_path]
            
            # Create a deep path
            deep_path = temp_path / "a" / "b" / "c" / "d" / "e" / "f" / "test.txt"
            deep_path.parent.mkdir(parents=True, exist_ok=True)
            deep_path.touch()
            
            # Should be rejected due to depth
            validated = self.validator.validate_path(deep_path)
            assert validated is None
    
    def test_forbidden_path_segments(self):
        """Test rejection of forbidden path segments"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.config.allowed_base_paths = [temp_path]
            
            # Create path with forbidden segment
            forbidden_path = temp_path / "etc" / "test.txt"
            forbidden_path.parent.mkdir(parents=True, exist_ok=True)
            forbidden_path.touch()
            
            validated = self.validator.validate_path(forbidden_path)
            assert validated is None
    
    def test_validate_directory_path_creation(self):
        """Test directory validation with creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.config.allowed_base_paths = [temp_path]
            
            # Test creating new directory
            new_dir = temp_path / "new_directory"
            assert not new_dir.exists()
            
            validated = self.validator.validate_directory_path(
                new_dir, create_if_missing=True
            )
            
            assert validated is not None
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_validate_directory_path_existing_file(self):
        """Test directory validation when path is existing file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.config.allowed_base_paths = [temp_path]
            
            # Create a file
            test_file = temp_path / "test.txt"
            test_file.touch()
            
            # Should fail directory validation
            validated = self.validator.validate_directory_path(test_file)
            assert validated is None
    
    def test_allow_creation_parameter(self):
        """Test the allow_creation parameter for non-existing paths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.config.allowed_base_paths = [temp_path]
            
            non_existing = temp_path / "does_not_exist.txt"
            
            # Without allow_creation, should fail for non-existing
            validated = self.validator.validate_path(
                non_existing, must_exist=True, allow_creation=False
            )
            assert validated is None
            
            # With allow_creation, should succeed
            validated = self.validator.validate_path(
                non_existing, must_exist=False, allow_creation=True
            )
            assert validated is not None


class TestFileValidator:
    """Tests for FileValidator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = SecurityConfig()
        self.validator = FileValidator(self.config)
    
    def test_valid_file_validation(self):
        """Test validation of a valid file"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(b"print('hello world')")
            temp_file.flush()
            
            try:
                result = self.validator.validate_file(Path(temp_file.name))
                
                assert result['valid'] is True
                assert result['size'] > 0
                assert len(result['errors']) == 0
            finally:
                os.unlink(temp_file.name)
    
    def test_file_size_limits(self):
        """Test file size validation"""
        self.config.max_file_size = 100  # Very small limit
        
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            # Write more than the limit
            temp_file.write(b"x" * 200)
            temp_file.flush()
            
            try:
                result = self.validator.validate_file(Path(temp_file.name))
                
                assert result['valid'] is False
                assert any("too large" in error.lower() for error in result['errors'])
            finally:
                os.unlink(temp_file.name)
    
    def test_forbidden_extensions(self):
        """Test rejection of forbidden file extensions"""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as temp_file:
            temp_file.write(b"fake executable")
            temp_file.flush()
            
            try:
                result = self.validator.validate_file(Path(temp_file.name))
                
                assert result['valid'] is False
                assert any("forbidden" in error.lower() for error in result['errors'])
            finally:
                os.unlink(temp_file.name)
    
    def test_non_existent_file(self):
        """Test validation of non-existent file"""
        non_existent = Path("/tmp/does_not_exist_12345.txt")
        result = self.validator.validate_file(non_existent)
        
        assert result['valid'] is False
        assert any("does not exist" in error.lower() for error in result['errors'])
    
    def test_empty_file_warning(self):
        """Test warning for empty files"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            # Don't write anything - file will be empty
            temp_file.flush()
            
            try:
                result = self.validator.validate_file(Path(temp_file.name))
                
                assert result['valid'] is True  # Empty files are valid
                assert any("empty" in warning.lower() for warning in result['warnings'])
            finally:
                os.unlink(temp_file.name)


class TestContentScanner:
    """Tests for ContentScanner"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = SecurityConfig()
        self.scanner = ContentScanner(self.config)
    
    def test_safe_content(self):
        """Test scanning of safe content"""
        safe_content = """
def hello_world():
    print("Hello, World!")
    return True
"""
        result = self.scanner.scan_content(safe_content, Path("test.py"))
        
        assert result['safe'] is True
        assert len(result['issues']) == 0
        assert len(result['sensitive_data']) == 0
    
    def test_sensitive_data_detection(self):
        """Test detection of sensitive data patterns"""
        sensitive_content = """
API_KEY = "sk-1234567890abcdef"
DATABASE_URL = "postgresql://user:password123@localhost/db"
"""
        result = self.scanner.scan_content(sensitive_content, Path("config.py"))
        
        assert result['safe'] is False
        assert len(result['sensitive_data']) > 0
        
        # Should detect API key and database URL
        sensitive_types = [item['type'] for item in result['sensitive_data']]
        assert any("API Key" in stype for stype in sensitive_types)
    
    def test_malicious_pattern_detection(self):
        """Test detection of potentially malicious patterns"""
        malicious_content = """
import subprocess
result = subprocess.system("rm -rf /")
eval(user_input)
"""
        result = self.scanner.scan_content(malicious_content, Path("malicious.py"))
        
        assert len(result['suspicious_patterns']) > 0
        
        # Should detect code execution patterns
        pattern_types = [item['type'] for item in result['suspicious_patterns']]
        assert any("Code execution" in ptype for ptype in pattern_types)
    
    def test_line_length_limits(self):
        """Test line length validation"""
        self.config.max_line_length = 50
        
        long_line_content = "x" * 100  # Single long line
        result = self.scanner.scan_content(long_line_content, Path("test.py"))
        
        assert any("too long" in issue.lower() for issue in result['issues'])
    
    def test_file_line_count_limits(self):
        """Test file line count validation"""
        self.config.max_lines_per_file = 5
        
        many_lines_content = "\n".join([f"line {i}" for i in range(10)])
        result = self.scanner.scan_content(many_lines_content, Path("test.py"))
        
        assert result['safe'] is False
        assert any("too many lines" in issue.lower() for issue in result['issues'])
    
    def test_content_scanning_disabled(self):
        """Test behavior when content scanning is disabled"""
        self.config.enable_content_scanning = False
        
        sensitive_content = 'API_KEY = "secret123"'
        result = self.scanner.scan_content(sensitive_content, Path("test.py"))
        
        # Should return safe result without scanning
        assert result['safe'] is True
        assert len(result['sensitive_data']) == 0


class TestSecureProcessor:
    """Tests for SecureProcessor utilities"""
    
    def test_safe_json_loads_valid(self):
        """Test safe JSON loading with valid data"""
        valid_json = '{"key": "value", "number": 42}'
        result = SecureProcessor.safe_json_loads(valid_json)
        
        assert result == {"key": "value", "number": 42}
    
    def test_safe_json_loads_invalid(self):
        """Test safe JSON loading with invalid data"""
        invalid_json = '{"key": "value"'  # Missing closing brace
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            SecureProcessor.safe_json_loads(invalid_json)
    
    def test_safe_json_loads_too_large(self):
        """Test rejection of overly large JSON"""
        large_json = '{"key": "' + "x" * (11 * 1024 * 1024) + '"}'  # > 10MB
        
        with pytest.raises(ValueError, match="too large"):
            SecureProcessor.safe_json_loads(large_json)
    
    def test_safe_ast_parse_valid(self):
        """Test safe AST parsing with valid Python code"""
        valid_code = """
def hello():
    return "world"
"""
        tree = SecureProcessor.safe_ast_parse(valid_code)
        
        assert tree is not None
        assert hasattr(tree, 'body')
    
    def test_safe_ast_parse_dangerous(self):
        """Test rejection of dangerous Python code"""
        dangerous_code = """
import os
os.system("rm -rf /")
"""
        with pytest.raises(ValueError, match="Security violations"):
            SecureProcessor.safe_ast_parse(dangerous_code)
    
    def test_safe_ast_parse_syntax_error(self):
        """Test handling of syntax errors"""
        invalid_code = "def invalid_syntax(:"
        
        with pytest.raises(ValueError, match="Invalid Python syntax"):
            SecureProcessor.safe_ast_parse(invalid_code)
    
    def test_safe_ast_parse_null_bytes(self):
        """Test rejection of code with null bytes"""
        null_code = "print('hello')\x00print('world')"
        
        with pytest.raises(ValueError, match="null bytes"):
            SecureProcessor.safe_ast_parse(null_code)


class TestSecurityValidator:
    """Integration tests for SecurityValidator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = SecurityValidator()
    
    def test_comprehensive_validation_safe_file(self):
        """Test comprehensive validation of a safe file"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
            temp_file.write("""
def calculate_sum(a, b):
    '''Calculate the sum of two numbers.'''
    return a + b

if __name__ == "__main__":
    result = calculate_sum(5, 3)
    print(f"Result: {result}")
""")
            temp_file.flush()
            
            try:
                # Update config to allow temp directory
                temp_path = Path(temp_file.name).parent
                self.validator.config.allowed_base_paths = [temp_path]
                
                result = self.validator.validate_input(temp_file.name)
                
                assert result['valid'] is True
                assert result['path'] is not None
                assert len(result['errors']) == 0
            finally:
                os.unlink(temp_file.name)
    
    def test_comprehensive_validation_unsafe_file(self):
        """Test comprehensive validation of an unsafe file"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
            temp_file.write("""
import os
API_KEY = "sk-1234567890abcdef"
os.system("dangerous command")
""")
            temp_file.flush()
            
            try:
                # Update config to allow temp directory
                temp_path = Path(temp_file.name).parent  
                self.validator.config.allowed_base_paths = [temp_path]
                
                result = self.validator.validate_input(temp_file.name)
                
                # File should be valid but have security warnings
                assert result['valid'] is True
                assert len(result['warnings']) > 0
                assert len(result['security_issues']) > 0
            finally:
                os.unlink(temp_file.name)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Set very low rate limit for testing
        self.validator.config.rate_limit_calls = 2
        self.validator.config.rate_limit_window = 60.0
        
        # Make requests up to the limit
        assert self.validator.rate_limiter.allow_request() is True
        assert self.validator.rate_limiter.allow_request() is True
        
        # Next request should be denied
        assert self.validator.rate_limiter.allow_request() is False
    
    def test_error_message_sanitization(self):
        """Test sanitization of error messages"""
        # Test internal error message (should not be sanitized)
        error = FileNotFoundError("File /secret/path not found")
        internal_msg = self.validator.sanitize_error_message(error, internal=True)
        assert "/secret/path" in internal_msg
        
        # Test external error message (should be sanitized)
        external_msg = self.validator.sanitize_error_message(error, internal=False)
        assert "/secret/path" not in external_msg
        assert "not found" in external_msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])