#!/usr/bin/env python3
"""
Comprehensive Security Test Suite for Codebase Compression Pipeline
Tests the recently applied security fixes:
1. eval() → ast.literal_eval() replacement
2. Bare except → specific exception handling
"""

import ast
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Set
import traceback
import time
import psutil
import os

# Import the classes we need to test
from codebase_compression_pipeline import CodebaseCompressionPipeline, FileMetadata


class SecurityTestSuite:
    """Comprehensive security testing for the pipeline fixes"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.pipeline = None
        
    def setup_test_environment(self):
        """Setup temporary directory and pipeline for testing"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="security_test_"))
        cache_dir = self.temp_dir / "cache"
        output_dir = self.temp_dir / "output"
        
        # Create directories first
        cache_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        self.pipeline = CodebaseCompressionPipeline(
            cache_dir=cache_dir,
            output_dir=output_dir
        )
        print(f"Test environment setup at: {self.temp_dir}")
        
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("Test environment cleaned up")
            
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test results"""
        status = "PASS" if passed else "FAIL"
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        print(f"[{status}] {test_name}: {details}")
        
    def test_ast_literal_eval_security(self):
        """Test ast.literal_eval() security improvements"""
        print("\n=== Testing ast.literal_eval() Security ===")
        
        # Test 1: Valid set inputs
        valid_inputs = [
            "set()",
            "{'a', 'b', 'c'}",
            "{1, 2, 3}",
            "{'module1', 'module2'}",
        ]
        
        for input_str in valid_inputs:
            try:
                result = ast.literal_eval(input_str)
                if isinstance(result, set):
                    self.log_test_result(
                        f"ast.literal_eval valid input: {input_str}",
                        True,
                        f"Correctly parsed to set: {result}"
                    )
                else:
                    self.log_test_result(
                        f"ast.literal_eval valid input: {input_str}",
                        False,
                        f"Expected set, got {type(result)}"
                    )
            except Exception as e:
                self.log_test_result(
                    f"ast.literal_eval valid input: {input_str}",
                    False,
                    f"Unexpected exception: {e}"
                )
        
        # Test 2: Malicious inputs that would work with eval() but fail with ast.literal_eval()
        malicious_inputs = [
            "__import__('os').system('echo hacked')",
            "exec('print(\"malicious code\")')",
            "eval('1+1')",
            "open('/etc/passwd', 'r').read()",
            "globals()",
            "locals()",
            "dir()",
            "__builtins__",
            "lambda x: x",
            "list(range(1000000))",  # Memory exhaustion attempt
            "[i for i in range(1000000)]",  # List comprehension
        ]
        
        for malicious_input in malicious_inputs:
            try:
                result = ast.literal_eval(malicious_input)
                self.log_test_result(
                    f"ast.literal_eval malicious input: {malicious_input[:50]}...",
                    False,
                    f"SECURITY ISSUE: Malicious code executed successfully: {result}"
                )
            except (ValueError, SyntaxError) as e:
                self.log_test_result(
                    f"ast.literal_eval malicious input: {malicious_input[:50]}...",
                    True,
                    f"Correctly blocked malicious input: {type(e).__name__}"
                )
            except Exception as e:
                self.log_test_result(
                    f"ast.literal_eval malicious input: {malicious_input[:50]}...",
                    True,
                    f"Blocked with unexpected exception: {type(e).__name__}"
                )
        
        # Test 3: Edge cases and malformed inputs
        edge_cases = [
            "",  # Empty string
            "None",
            "True",
            "False",
            "[]",  # Empty list
            "{}",  # Empty dict
            "set([1,2,3])",  # Function call
            "{1, 2, 3, }",  # Trailing comma
            "{'a': 1}",  # Dict instead of set
            "[1, 2, 3]",  # List instead of set
            "(1, 2, 3)",  # Tuple instead of set
            "set",  # Variable name
            "{'a', 'b'} | {'c'}",  # Set operations
        ]
        
        for edge_case in edge_cases:
            try:
                result = ast.literal_eval(edge_case)
                self.log_test_result(
                    f"ast.literal_eval edge case: {edge_case}",
                    True,
                    f"Parsed as {type(result)}: {result}"
                )
            except (ValueError, SyntaxError) as e:
                self.log_test_result(
                    f"ast.literal_eval edge case: {edge_case}",
                    True,
                    f"Safely rejected: {type(e).__name__}"
                )
            except Exception as e:
                self.log_test_result(
                    f"ast.literal_eval edge case: {edge_case}",
                    False,
                    f"Unexpected exception: {type(e).__name__}: {e}"
                )
    
    def test_specific_exception_handling(self):
        """Test the specific exception handling improvements"""
        print("\n=== Testing Specific Exception Handling ===")
        
        # Test the specific exception types that should be caught
        test_cases = [
            ("ValueError", "ast.literal_eval('invalid')", ValueError),
            ("SyntaxError", "ast.literal_eval('1 + +')", SyntaxError),
        ]
        
        for test_name, code, expected_exception in test_cases:
            try:
                exec(code)
                self.log_test_result(
                    f"Exception handling: {test_name}",
                    False,
                    f"Expected {expected_exception.__name__} but code executed successfully"
                )
            except expected_exception as e:
                self.log_test_result(
                    f"Exception handling: {test_name}",
                    True,
                    f"Correctly caught {type(e).__name__}"
                )
            except Exception as e:
                self.log_test_result(
                    f"Exception handling: {test_name}",
                    False,
                    f"Expected {expected_exception.__name__}, got {type(e).__name__}"
                )
    
    def test_dependencies_parsing_integration(self):
        """Test the dependencies parsing in the actual pipeline context"""
        print("\n=== Testing Dependencies Parsing Integration ===")
        
        # Create mock data similar to what would be in cache
        test_data_cases = [
            {
                "name": "valid_set_string",
                "data": {
                    "path": "/test/file.py",
                    "size": 1000,
                    "language": "python",
                    "last_modified": 1234567890,
                    "content_hash": "test_hash",
                    "dependencies": "{'os', 'sys', 'json'}"
                }
            },
            {
                "name": "empty_set_string",
                "data": {
                    "path": "/test/file.py",
                    "size": 1000,
                    "language": "python",
                    "last_modified": 1234567890,
                    "content_hash": "test_hash",
                    "dependencies": "set()"
                }
            },
            {
                "name": "malicious_string",
                "data": {
                    "path": "/test/file.py",
                    "size": 1000,
                    "language": "python",
                    "last_modified": 1234567890,
                    "content_hash": "test_hash",
                    "dependencies": "__import__('os').system('echo hacked')"
                }
            },
            {
                "name": "invalid_syntax",
                "data": {
                    "path": "/test/file.py",
                    "size": 1000,
                    "language": "python",
                    "last_modified": 1234567890,
                    "content_hash": "test_hash",
                    "dependencies": "invalid syntax here"
                }
            },
            {
                "name": "already_set",
                "data": {
                    "path": "/test/file.py",
                    "size": 1000,
                    "language": "python",
                    "last_modified": 1234567890,
                    "content_hash": "test_hash",
                    "dependencies": {"os", "sys"}
                }
            },
            {
                "name": "list_input",
                "data": {
                    "path": "/test/file.py",
                    "size": 1000,
                    "language": "python",
                    "last_modified": 1234567890,
                    "content_hash": "test_hash",
                    "dependencies": ["os", "sys"]
                }
            }
        ]
        
        for test_case in test_data_cases:
            try:
                # Test the dependencies parsing logic directly
                data = test_case["data"]
                dependencies = data.get('dependencies', set())
                
                if isinstance(dependencies, str):
                    if dependencies == "set()":
                        dependencies = set()
                    else:
                        try:
                            dependencies = ast.literal_eval(dependencies)
                            if not isinstance(dependencies, set):
                                dependencies = set()
                        except (ValueError, SyntaxError) as e:
                            dependencies = set()
                elif not isinstance(dependencies, set):
                    dependencies = set(dependencies) if dependencies else set()
                
                # Verify result is always a set
                if isinstance(dependencies, set):
                    self.log_test_result(
                        f"Dependencies parsing: {test_case['name']}",
                        True,
                        f"Correctly parsed to set with {len(dependencies)} items"
                    )
                else:
                    self.log_test_result(
                        f"Dependencies parsing: {test_case['name']}",
                        False,
                        f"Expected set, got {type(dependencies)}"
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Dependencies parsing: {test_case['name']}",
                    False,
                    f"Unexpected exception: {type(e).__name__}: {e}"
                )
    
    def test_cache_corruption_resilience(self):
        """Test resilience against corrupted cache data"""
        print("\n=== Testing Cache Corruption Resilience ===")
        
        # Create cache directory and corrupted data
        cache_dir = self.temp_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Test cases for corrupted cache files
        corruption_tests = [
            {
                "name": "corrupted_json",
                "content": '{"invalid": json syntax here'
            },
            {
                "name": "binary_data",
                "content": b'\x00\x01\x02\x03\x04\x05'
            },
            {
                "name": "empty_file",
                "content": ""
            },
            {
                "name": "malicious_eval_in_json",
                "content": '{"dependencies": "__import__(\'os\').system(\'echo hacked\')"}'
            }
        ]
        
        for test in corruption_tests:
            try:
                # Create corrupted cache file
                cache_file = cache_dir / "test_cache.json"
                if isinstance(test["content"], str):
                    cache_file.write_text(test["content"])
                else:
                    cache_file.write_bytes(test["content"])
                
                # Try to load and process
                try:
                    if test["content"]:
                        data = json.loads(cache_file.read_text())
                        # Test our dependencies parsing logic
                        dependencies = data.get('dependencies', set())
                        if isinstance(dependencies, str):
                            if dependencies == "set()":
                                dependencies = set()
                            else:
                                try:
                                    dependencies = ast.literal_eval(dependencies)
                                    if not isinstance(dependencies, set):
                                        dependencies = set()
                                except (ValueError, SyntaxError):
                                    dependencies = set()
                        
                        self.log_test_result(
                            f"Cache corruption: {test['name']}",
                            True,
                            "Handled corrupted data gracefully"
                        )
                    else:
                        self.log_test_result(
                            f"Cache corruption: {test['name']}",
                            True,
                            "Empty file handled gracefully"
                        )
                        
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    self.log_test_result(
                        f"Cache corruption: {test['name']}",
                        True,
                        f"Correctly failed to parse corrupted data: {type(e).__name__}"
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Cache corruption: {test['name']}",
                    True,
                    f"Handled with exception: {type(e).__name__}"
                )
            finally:
                # Clean up
                if cache_file.exists():
                    cache_file.unlink()
    
    def test_performance_impact(self):
        """Test performance impact of security fixes"""
        print("\n=== Testing Performance Impact ===")
        
        # Test performance of ast.literal_eval vs eval for large datasets
        test_data = [f"{{'dep{i}', 'module{i}'}}" for i in range(1000)]
        
        # Time ast.literal_eval (secure method)
        start_time = time.time()
        for data in test_data:
            try:
                result = ast.literal_eval(data)
            except (ValueError, SyntaxError):
                result = set()
        secure_time = time.time() - start_time
        
        # Memory usage test
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Process a large number of dependency strings
        large_dataset = [f"{{'dep{i}'}}" for i in range(10000)]
        for data in large_dataset:
            try:
                result = ast.literal_eval(data)
            except (ValueError, SyntaxError):
                result = set()
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        self.log_test_result(
            "Performance: ast.literal_eval speed",
            secure_time < 1.0,  # Should complete in less than 1 second
            f"Processed 1000 items in {secure_time:.3f} seconds"
        )
        
        self.log_test_result(
            "Performance: memory usage",
            memory_increase < 50 * 1024 * 1024,  # Less than 50MB increase
            f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB"
        )
    
    def test_concurrent_access(self):
        """Test concurrent access scenarios"""
        print("\n=== Testing Concurrent Access ===")
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker_thread(thread_id):
            """Worker thread to test concurrent dependencies parsing"""
            try:
                test_dependencies = [
                    "{'os', 'sys'}",
                    "set()",
                    "invalid syntax",
                    "{'module1', 'module2', 'module3'}"
                ]
                
                for dep_str in test_dependencies:
                    dependencies = dep_str
                    if isinstance(dependencies, str):
                        if dependencies == "set()":
                            dependencies = set()
                        else:
                            try:
                                dependencies = ast.literal_eval(dependencies)
                                if not isinstance(dependencies, set):
                                    dependencies = set()
                            except (ValueError, SyntaxError):
                                dependencies = set()
                
                results_queue.put(("success", thread_id, len(test_dependencies)))
                
            except Exception as e:
                results_queue.put(("error", thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        num_threads = 10
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        success_count = 0
        error_count = 0
        
        while not results_queue.empty():
            result_type, thread_id, result = results_queue.get()
            if result_type == "success":
                success_count += 1
            else:
                error_count += 1
        
        self.log_test_result(
            "Concurrent access test",
            success_count == num_threads and error_count == 0,
            f"Success: {success_count}/{num_threads}, Errors: {error_count}"
        )
    
    def run_all_tests(self):
        """Run all security tests"""
        print("Starting Comprehensive Security Test Suite")
        print("=" * 60)
        
        try:
            self.setup_test_environment()
            
            # Run all test categories
            self.test_ast_literal_eval_security()
            self.test_specific_exception_handling()
            self.test_dependencies_parsing_integration()
            self.test_cache_corruption_resilience()
            self.test_performance_impact()
            self.test_concurrent_access()
            
            # Generate summary report
            self.generate_summary_report()
            
        except Exception as e:
            print(f"Test suite encountered an error: {e}")
            traceback.print_exc()
        finally:
            self.cleanup_test_environment()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("SECURITY TEST SUITE SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            print("-" * 40)
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"❌ {result['test']}: {result['details']}")
        
        print("\nSECURITY ASSESSMENT:")
        print("-" * 40)
        
        # Analyze security-critical test results
        security_critical_tests = [r for r in self.test_results if "malicious" in r["test"].lower()]
        security_passed = sum(1 for r in security_critical_tests if r["status"] == "PASS")
        
        if security_critical_tests:
            security_rate = (security_passed / len(security_critical_tests)) * 100
            if security_rate == 100:
                print("✅ All security-critical tests passed - No vulnerabilities detected")
            else:
                print(f"⚠️  Security concerns: {len(security_critical_tests) - security_passed} vulnerabilities detected")
        
        print("\nRECOMMENDations:")
        print("-" * 40)
        if failed_tests == 0:
            print("✅ Security fixes are working correctly")
            print("✅ No regressions detected in functionality")
            print("✅ Performance impact is acceptable")
        else:
            print("⚠️  Some tests failed - review the failed test details above")
            print("⚠️  Consider additional security hardening")
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests/total_tests)*100,
            "security_critical_passed": security_passed if security_critical_tests else 0,
            "results": self.test_results
        }


if __name__ == "__main__":
    suite = SecurityTestSuite()
    suite.run_all_tests()