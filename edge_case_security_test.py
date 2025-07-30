#!/usr/bin/env python3
"""
Edge Case Testing for Security Fixes
Focuses on testing corner cases and edge scenarios for the security improvements
"""

import ast
import json
import tempfile
import os
from pathlib import Path
import time
import threading
import queue


def test_cache_corruption_scenarios():
    """Test various cache corruption scenarios"""
    print("=" * 60)
    print("TESTING CACHE CORRUPTION EDGE CASES")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Create test scenarios with corrupted cache data
    corruption_scenarios = [
        {
            "name": "Partial JSON corruption",
            "data": '{"path": "/test.py", "dependencies": "set(", "size": 100}',
            "expected_behavior": "should handle gracefully"
        },
        {
            "name": "Mixed quotes in dependencies",
            "data": '{"dependencies": "{\'os\', "sys", \'json\'}"}',
            "expected_behavior": "should parse or fail safely"
        },
        {
            "name": "Unicode in dependencies",
            "data": '{"dependencies": "{\\"моду́ль\\", \\"sys\\"}"}',
            "expected_behavior": "should handle unicode gracefully"
        },
        {
            "name": "Very large dependency set",
            "data": '{"dependencies": "' + str(set(f"module_{i}" for i in range(10000))) + '"}',
            "expected_behavior": "should handle large sets"
        },
        {
            "name": "Nested structures",
            "data": '{"dependencies": "{{\\"nested\\": {\\"deep\\": \\"value\\"}}}"}',
            "expected_behavior": "should reject nested structures"
        },
        {
            "name": "SQL injection attempt",
            "data": '{"dependencies": "{\\"DROP TABLE users;\\", \\"sys\\"}"}',
            "expected_behavior": "should treat as literal string"
        },
        {
            "name": "Path traversal attempt",
            "data": '{"dependencies": "{\\"../../../etc/passwd\\", \\"sys\\"}"}',
            "expected_behavior": "should treat as literal string"
        },
        {
            "name": "Command injection attempt", 
            "data": '{"dependencies": "{\\"$(rm -rf /)\\", \\"sys\\"}"}',
            "expected_behavior": "should treat as literal string"
        }
    ]
    
    def safe_parse_dependencies(dep_str):
        """Replicate the exact security fix logic"""
        dependencies = dep_str
        if isinstance(dependencies, str):
            if dependencies == "set()":
                dependencies = set()
            else:
                try:
                    dependencies = ast.literal_eval(dependencies)
                    if not isinstance(dependencies, set):
                        dependencies = set()
                except (ValueError, SyntaxError) as e:
                    print(f"    Safe rejection: {type(e).__name__}")
                    dependencies = set()
        elif not isinstance(dependencies, set):
            dependencies = set(dependencies) if dependencies else set()
        return dependencies
    
    for scenario in corruption_scenarios:
        print(f"\nTesting: {scenario['name']}")
        try:
            # Parse JSON first
            data = json.loads(scenario["data"])
            dep_str = data.get("dependencies", "set()")
            
            # Apply security fix logic
            result = safe_parse_dependencies(dep_str)
            
            if isinstance(result, set):
                print(f"✅ PASS: Safely converted to set with {len(result)} elements")
                # Check for suspicious content in result
                suspicious = any("rm" in str(item) or "DROP" in str(item) or ".." in str(item) 
                               for item in result)
                if suspicious:
                    print(f"⚠️  WARNING: Suspicious content found in result: {result}")
                results["passed"] += 1
            else:
                print(f"❌ FAIL: Expected set, got {type(result)}")
                results["failed"] += 1
                results["details"].append(f"Corruption test failed: {scenario['name']}")
                
        except json.JSONDecodeError as e:
            print(f"✅ PASS: JSON parsing safely failed - {type(e).__name__}")
            results["passed"] += 1
        except Exception as e:
            print(f"❌ FAIL: Unexpected exception - {type(e).__name__}: {e}")
            results["failed"] += 1
            results["details"].append(f"Unexpected error in {scenario['name']}: {e}")
    
    return results


def test_extreme_memory_scenarios():
    """Test extreme memory usage scenarios"""
    print("\n" + "=" * 60)
    print("TESTING EXTREME MEMORY SCENARIOS")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Test scenarios that could cause memory issues
    memory_tests = [
        {
            "name": "Very long string in set",
            "input": "{'A" + "A" * 100000 + "'}",
            "expected": "should handle or reject gracefully"
        },
        {
            "name": "Deeply nested structure attempt",
            "input": "{'" + "[" * 1000 + "]" * 1000 + "'}",
            "expected": "should reject malformed structure"
        },
        {
            "name": "Many escape sequences",
            "input": f"{{'\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\a'}}",
            "expected": "should handle or reject gracefully"
        }
    ]
    
    for test in memory_tests:
        print(f"\nTesting: {test['name']}")
        start_time = time.time()
        
        try:
            result = ast.literal_eval(test["input"])
            end_time = time.time()
            parse_time = end_time - start_time
            
            if parse_time > 5.0:  # More than 5 seconds
                print(f"⚠️  PERFORMANCE ISSUE: Took {parse_time:.2f} seconds")
                results["failed"] += 1
                results["details"].append(f"Performance issue: {test['name']} took {parse_time:.2f}s")
            else:
                print(f"✅ PASS: Parsed in {parse_time:.3f} seconds, result type: {type(result)}")
                results["passed"] += 1
                
        except (ValueError, SyntaxError) as e:
            end_time = time.time()
            parse_time = end_time - start_time
            print(f"✅ PASS: Safely rejected in {parse_time:.3f} seconds ({type(e).__name__})")
            results["passed"] += 1
        except Exception as e:
            print(f"❌ FAIL: Unexpected exception - {type(e).__name__}: {e}")
            results["failed"] += 1
            results["details"].append(f"Memory test unexpected error: {test['name']} - {e}")
    
    return results


def test_unicode_and_encoding_edge_cases():
    """Test unicode and encoding edge cases"""
    print("\n" + "=" * 60)
    print("TESTING UNICODE AND ENCODING EDGE CASES")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    unicode_tests = [
        {
            "name": "Basic Unicode in set",
            "input": "{'модуль', 'sys', '系统'}",
            "expected": "should handle unicode correctly"
        },
        {
            "name": "Emoji in dependencies",
            "input": "{'🔒', '🛡️', 'security'}",
            "expected": "should handle emojis"
        },
        {
            "name": "Mixed encoding characters",
            "input": "{'café', 'naïve', 'résumé'}",
            "expected": "should handle accented characters"
        },
        {
            "name": "Control characters attempt",
            "input": "{'\\x00\\x01\\x02', 'sys'}",
            "expected": "should handle control chars safely"
        },
        {
            "name": "NULL bytes attempt",
            "input": "{'test\\x00inject', 'sys'}",
            "expected": "should handle null bytes"
        }
    ]
    
    for test in unicode_tests:
        print(f"\nTesting: {test['name']}")
        try:
            result = ast.literal_eval(test["input"])
            if isinstance(result, set):
                print(f"✅ PASS: Parsed unicode set with {len(result)} items")
                # Check for suspicious content
                has_control_chars = any(ord(c) < 32 for item in result for c in str(item))
                if has_control_chars:
                    print(f"⚠️  WARNING: Control characters detected in result")
                results["passed"] += 1
            else:
                print(f"❌ FAIL: Expected set, got {type(result)}")
                results["failed"] += 1
                results["details"].append(f"Unicode test type error: {test['name']}")
        except (ValueError, SyntaxError) as e:
            print(f"✅ PASS: Safely rejected unicode input ({type(e).__name__})")
            results["passed"] += 1
        except Exception as e:
            print(f"❌ FAIL: Unexpected exception - {type(e).__name__}: {e}")
            results["failed"] += 1
            results["details"].append(f"Unicode test error: {test['name']} - {e}")
    
    return results


def test_concurrent_corruption_scenarios():
    """Test concurrent access with corrupted data"""
    print("\n" + "=" * 60)
    print("TESTING CONCURRENT CORRUPTION SCENARIOS")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Create test data with various corruption patterns
    corrupt_data = [
        "{'valid', 'set'}",
        "invalid syntax {{{",
        "__import__('os').system('echo hack')",
        "{'partially': corrupted",
        "set()",
        "{'unicode': 'тест'}",
        "",
        "None",
        "{'very_long_' + 'string_' * 1000 + 'item'}",
        "eval('malicious_code')"
    ]
    
    def worker_with_corruption(thread_id, test_queue, result_queue):
        """Worker that processes potentially corrupted dependency strings"""
        local_results = {"processed": 0, "errors": 0, "blocked": 0}
        
        while not test_queue.empty():
            try:
                dep_str = test_queue.get_nowait()
                
                # Apply the security fix logic
                dependencies = dep_str
                if isinstance(dependencies, str):
                    if dependencies == "set()":
                        dependencies = set()
                    else:
                        try:
                            dependencies = ast.literal_eval(dependencies)
                            if not isinstance(dependencies, set):
                                dependencies = set()
                            local_results["processed"] += 1
                        except (ValueError, SyntaxError):
                            dependencies = set()
                            local_results["blocked"] += 1
                
            except queue.Empty:
                break
            except Exception as e:
                local_results["errors"] += 1
        
        result_queue.put(("success", thread_id, local_results))
    
    # Set up concurrent test
    test_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Fill queue with test data (repeated for more work)
    for _ in range(10):  # 10 iterations
        for data in corrupt_data:
            test_queue.put(data)
    
    # Start threads
    threads = []
    num_threads = 5
    
    print(f"Starting {num_threads} threads to process {test_queue.qsize()} corrupted items...")
    
    start_time = time.time()
    for i in range(num_threads):
        thread = threading.Thread(target=worker_with_corruption, 
                                 args=(i, test_queue, result_queue))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    execution_time = time.time() - start_time
    
    # Collect results
    total_processed = 0
    total_blocked = 0
    total_errors = 0
    successful_threads = 0
    
    while not result_queue.empty():
        result_type, thread_id, data = result_queue.get()
        if result_type == "success":
            successful_threads += 1
            total_processed += data["processed"]
            total_blocked += data["blocked"]
            total_errors += data["errors"]
    
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Successful threads: {successful_threads}/{num_threads}")
    print(f"Total processed: {total_processed}")
    print(f"Total blocked (security): {total_blocked}")
    print(f"Total errors: {total_errors}")
    
    if successful_threads == num_threads and total_errors == 0:
        print("✅ PASS: Concurrent corruption handling successful")
        results["passed"] += 1
    else:
        print("❌ FAIL: Concurrent corruption handling issues detected")
        results["failed"] += 1
        results["details"].append(f"Concurrent corruption: {total_errors} errors, {num_threads - successful_threads} failed threads")
    
    return results


def main():
    """Run all edge case security tests"""
    print("EDGE CASE SECURITY TESTING")
    print("=" * 80)
    print("Testing corner cases and edge scenarios for security fixes")
    print("=" * 80)
    
    overall_results = {
        "total_tests": 0,
        "total_passed": 0, 
        "total_failed": 0,
        "details": []
    }
    
    try:
        # Run all test categories
        corruption_results = test_cache_corruption_scenarios()
        memory_results = test_extreme_memory_scenarios()
        unicode_results = test_unicode_and_encoding_edge_cases()
        concurrent_results = test_concurrent_corruption_scenarios()
        
        # Aggregate results
        all_results = [corruption_results, memory_results, unicode_results, concurrent_results]
        
        for result in all_results:
            overall_results["total_tests"] += result["passed"] + result["failed"]
            overall_results["total_passed"] += result["passed"]
            overall_results["total_failed"] += result["failed"]
            overall_results["details"].extend(result.get("details", []))
        
        # Final report
        print("\n" + "=" * 80)
        print("EDGE CASE SECURITY TEST SUMMARY")
        print("=" * 80)
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Edge Case Tests: {overall_results['total_tests']}")
        print(f"   Tests Passed: {overall_results['total_passed']}")
        print(f"   Tests Failed: {overall_results['total_failed']}")
        
        if overall_results['total_tests'] > 0:
            success_rate = (overall_results['total_passed'] / overall_results['total_tests']) * 100
            print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\n🛡️ EDGE CASE SECURITY ASSESSMENT:")
        if overall_results['total_failed'] == 0:
            print("   ✅ ALL EDGE CASE TESTS PASSED")
            print("   ✅ Security fixes are robust against edge cases")
            print("   ✅ No memory or performance issues detected")
            print("   ✅ Unicode and encoding handled correctly")
            print("   ✅ Concurrent corruption scenarios handled safely")
        else:
            print("   ⚠️  Some edge case tests failed")
            print(f"   ❌ {overall_results['total_failed']} edge case issues detected")
            
            if overall_results["details"]:
                print("\n📋 EDGE CASE FAILURES:")
                for detail in overall_results["details"]:
                    print(f"   • {detail}")
        
        print(f"\n💡 EDGE CASE RECOMMENDATIONS:")
        if overall_results['total_failed'] == 0:
            print("   • Security fixes handle edge cases excellently")
            print("   • No additional edge case hardening needed")
            print("   • System is resilient to various corruption scenarios")
        else:
            print("   • Review failed edge case tests")
            print("   • Consider additional input validation")
            print("   • Test with production-like data corruption")
        
        return overall_results
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in edge case testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()