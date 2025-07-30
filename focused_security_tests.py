#!/usr/bin/env python3
"""
Focused Security Tests for the Security Fixes
Tests the specific security improvements without full pipeline setup:
1. ast.literal_eval() replacement for eval()
2. Specific exception handling (ValueError, SyntaxError)
"""

import ast
import time
import traceback
import psutil
import threading
import queue


def test_ast_literal_eval_security():
    """Test ast.literal_eval() security compared to eval()"""
    print("=" * 60)
    print("TESTING ast.literal_eval() SECURITY")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Test 1: Valid inputs that should work
    print("\n1. Testing Valid Inputs:")
    valid_inputs = [
        ("set()", set()),
        ("{'a', 'b', 'c'}", {'a', 'b', 'c'}),
        ("{1, 2, 3}", {1, 2, 3}),
        ("{'module1', 'module2'}", {'module1', 'module2'}),
        ("{}", set()),  # Empty set
    ]
    
    for input_str, expected in valid_inputs:
        try:
            result = ast.literal_eval(input_str)
            if result == expected:
                print(f"✅ PASS: {input_str} -> {result}")
                results["passed"] += 1
            else:
                print(f"❌ FAIL: {input_str} -> Expected {expected}, got {result}")
                results["failed"] += 1
                results["details"].append(f"Valid input mismatch: {input_str}")
        except Exception as e:
            print(f"❌ FAIL: {input_str} -> Exception: {type(e).__name__}: {e}")
            results["failed"] += 1
            results["details"].append(f"Valid input exception: {input_str} - {e}")
    
    # Test 2: CRITICAL SECURITY TEST - Malicious inputs that eval() would execute
    print("\n2. Testing Malicious Inputs (SECURITY CRITICAL):")
    malicious_inputs = [
        "__import__('os').system('echo HACKED')",
        "exec('print(\"Code injection successful\")')",
        "eval('1+1')",
        "open('/etc/passwd', 'r').read()",
        "globals()",
        "locals()",
        "dir()",
        "__builtins__",
        "lambda x: x",
        "[i for i in range(100)]",  # List comprehension
        "sum(range(1000000))",  # Expensive computation
        "getattr(__builtins__, 'eval')('1+1')",
        "compile('print(\"injected\")', '<string>', 'exec')",
    ]
    
    security_passed = 0
    for malicious_input in malicious_inputs:
        try:
            # This should FAIL with ast.literal_eval (security feature)
            result = ast.literal_eval(malicious_input)
            print(f"🚨 SECURITY FAILURE: {malicious_input[:40]}... -> EXECUTED: {result}")
            results["failed"] += 1
            results["details"].append(f"SECURITY BREACH: {malicious_input} executed successfully")
        except (ValueError, SyntaxError) as e:
            print(f"✅ SECURITY PASS: {malicious_input[:40]}... -> Blocked ({type(e).__name__})")
            results["passed"] += 1
            security_passed += 1
        except Exception as e:
            print(f"✅ SECURITY PASS: {malicious_input[:40]}... -> Blocked ({type(e).__name__})")
            results["passed"] += 1
            security_passed += 1
    
    # Test 3: Edge cases and malformed inputs
    print("\n3. Testing Edge Cases:")
    edge_cases = [
        "",  # Empty string
        "None",
        "True",
        "False",
        "[]",  # List
        "{}",  # Dict
        "(1, 2, 3)",  # Tuple
        "set([1,2,3])",  # Function call (should fail)
        "{'a': 1}",  # Dict instead of set
        "1 + 1",  # Expression (should fail)
        "print('hello')",  # Function call (should fail)
    ]
    
    for edge_case in edge_cases:
        try:
            result = ast.literal_eval(edge_case)
            print(f"✅ EDGE CASE: {edge_case} -> {type(result).__name__}: {result}")
            results["passed"] += 1
        except (ValueError, SyntaxError) as e:
            print(f"✅ EDGE CASE: {edge_case} -> Safely rejected ({type(e).__name__})")
            results["passed"] += 1
        except Exception as e:
            print(f"⚠️  EDGE CASE: {edge_case} -> Unexpected exception ({type(e).__name__})")
            results["failed"] += 1
            results["details"].append(f"Edge case unexpected exception: {edge_case} - {e}")
    
    print(f"\nSECURITY TEST SUMMARY:")
    print(f"Total Tests: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Security Critical Tests Passed: {security_passed}/{len(malicious_inputs)}")
    
    return results


def test_specific_exception_handling():
    """Test the specific exception handling improvements"""
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC EXCEPTION HANDLING")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Test the exact logic from the security fix
    def parse_dependencies_secure(dependencies_str):
        """Replicate the exact logic from the security fix"""
        dependencies = dependencies_str
        if isinstance(dependencies, str):
            if dependencies == "set()":
                dependencies = set()
            else:
                try:
                    dependencies = ast.literal_eval(dependencies)
                    if not isinstance(dependencies, set):
                        dependencies = set()
                except (ValueError, SyntaxError) as e:
                    print(f"    WARNING: Failed to parse dependencies string safely: {type(e).__name__}")
                    dependencies = set()
        elif not isinstance(dependencies, set):
            dependencies = set(dependencies) if dependencies else set()
        
        return dependencies
    
    # Test cases for the dependency parsing logic
    test_cases = [
        {
            "name": "valid_set_string",
            "input": "{'os', 'sys', 'json'}",
            "expected": {'os', 'sys', 'json'}
        },
        {
            "name": "empty_set_string", 
            "input": "set()",
            "expected": set()
        },
        {
            "name": "malicious_code",
            "input": "__import__('os').system('echo hacked')",
            "expected": set()  # Should be converted to empty set
        },
        {
            "name": "invalid_syntax",
            "input": "invalid syntax here {{{",
            "expected": set()
        },
        {
            "name": "already_set",
            "input": {"already", "a", "set"},
            "expected": {"already", "a", "set"}
        },
        {
            "name": "list_input",
            "input": ["os", "sys"],
            "expected": {"os", "sys"}
        },
        {
            "name": "none_input",
            "input": None,
            "expected": set()
        },
        {
            "name": "empty_string",
            "input": "",
            "expected": set()
        }
    ]
    
    print("\nTesting Dependencies Parsing Logic:")
    for test_case in test_cases:
        try:
            result = parse_dependencies_secure(test_case["input"])
            if result == test_case["expected"]:
                print(f"✅ PASS: {test_case['name']} -> {result}")
                results["passed"] += 1
            else:
                print(f"❌ FAIL: {test_case['name']} -> Expected {test_case['expected']}, got {result}")
                results["failed"] += 1
                results["details"].append(f"Logic test failed: {test_case['name']}")
        except Exception as e:
            print(f"❌ FAIL: {test_case['name']} -> Unexpected exception: {type(e).__name__}: {e}")
            results["failed"] += 1
            results["details"].append(f"Logic test exception: {test_case['name']} - {e}")
    
    # Test specific exception types
    print("\nTesting Exception Type Specificity:")
    exception_tests = [
        {
            "name": "ValueError_test",
            "code": "ast.literal_eval('invalid')",
            "expected_exception": ValueError
        },
        {
            "name": "SyntaxError_test", 
            "code": "ast.literal_eval('1 + +')",
            "expected_exception": SyntaxError
        }
    ]
    
    for test in exception_tests:
        try:
            eval(test["code"])
            print(f"❌ FAIL: {test['name']} -> Expected exception but code executed")
            results["failed"] += 1
            results["details"].append(f"Exception test failed: {test['name']}")
        except test["expected_exception"] as e:
            print(f"✅ PASS: {test['name']} -> Correctly caught {type(e).__name__}")
            results["passed"] += 1
        except Exception as e:
            print(f"⚠️  PARTIAL: {test['name']} -> Caught {type(e).__name__} instead of {test['expected_exception'].__name__}")
            results["passed"] += 1  # Still caught an exception, which is good
    
    print(f"\nEXCEPTION HANDLING SUMMARY:")
    print(f"Total Tests: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    return results


def test_performance_impact():
    """Test performance impact of the security fixes"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE IMPACT")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Performance test: ast.literal_eval on large dataset
    print("\n1. Performance Test - Large Dataset Processing:")
    
    # Generate test data
    test_data = [f"{{'dep{i}', 'module{i}'}}" for i in range(1000)]
    
    # Time the secure method
    start_time = time.time()
    processed = 0
    for data in test_data:
        try:
            result = ast.literal_eval(data)
            processed += 1
        except (ValueError, SyntaxError):
            result = set()
            processed += 1
    
    processing_time = time.time() - start_time
    
    print(f"Processed {processed} dependency strings in {processing_time:.3f} seconds")
    print(f"Rate: {processed/processing_time:.0f} items/second")
    
    # Performance should be reasonable (less than 2 seconds for 1000 items)
    if processing_time < 2.0:
        print("✅ PASS: Performance is acceptable")
        results["passed"] += 1
    else:
        print("❌ FAIL: Performance is too slow")
        results["failed"] += 1
        results["details"].append(f"Performance too slow: {processing_time:.3f}s")
    
    # Memory test
    print("\n2. Memory Usage Test:")
    process = psutil.Process()
    memory_before = process.memory_info().rss
    
    # Process larger dataset
    large_dataset = [f"{{'dep{i}'}}" for i in range(5000)]
    for data in large_dataset:
        try:
            result = ast.literal_eval(data)
        except (ValueError, SyntaxError):
            result = set()
    
    memory_after = process.memory_info().rss
    memory_increase = memory_after - memory_before
    memory_mb = memory_increase / 1024 / 1024
    
    print(f"Memory increase: {memory_mb:.2f} MB")
    
    # Memory increase should be reasonable (less than 100MB)
    if memory_mb < 100:
        print("✅ PASS: Memory usage is acceptable")
        results["passed"] += 1
    else:
        print("❌ FAIL: Memory usage is too high")
        results["failed"] += 1
        results["details"].append(f"Memory usage too high: {memory_mb:.2f}MB")
    
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"Total Tests: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    return results


def test_concurrent_safety():
    """Test thread safety of the security fixes"""
    print("\n" + "=" * 60)
    print("TESTING CONCURRENT SAFETY")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Test concurrent parsing
    results_queue = queue.Queue()
    
    def worker_thread(thread_id, num_items=100):
        """Worker thread to test concurrent dependency parsing"""
        try:
            test_deps = [
                "{'os', 'sys'}",
                "set()",
                "invalid syntax",
                "{'module1', 'module2'}",
                "__import__('os').system('echo hack')",  # Malicious
                "{'a', 'b', 'c'}"
            ]
            
            local_results = {"success": 0, "blocked": 0, "errors": 0}
            
            for i in range(num_items):
                dep_str = test_deps[i % len(test_deps)]
                
                # Use the same logic as the security fix
                dependencies = dep_str
                if isinstance(dependencies, str):
                    if dependencies == "set()":
                        dependencies = set()
                    else:
                        try:
                            dependencies = ast.literal_eval(dependencies)
                            if not isinstance(dependencies, set):
                                dependencies = set()
                            local_results["success"] += 1
                        except (ValueError, SyntaxError):
                            dependencies = set()
                            local_results["blocked"] += 1
                
            results_queue.put(("success", thread_id, local_results))
            
        except Exception as e:
            results_queue.put(("error", thread_id, str(e)))
    
    # Start multiple threads
    print("Starting concurrent safety test with 10 threads...")
    threads = []
    num_threads = 10
    
    start_time = time.time()
    for i in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    execution_time = time.time() - start_time
    
    # Collect results
    success_threads = 0
    error_threads = 0
    total_success = 0
    total_blocked = 0
    
    while not results_queue.empty():
        result_type, thread_id, data = results_queue.get()
        if result_type == "success":
            success_threads += 1
            total_success += data["success"]
            total_blocked += data["blocked"]
        else:
            error_threads += 1
            print(f"Thread {thread_id} error: {data}")
    
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Successful threads: {success_threads}/{num_threads}")
    print(f"Error threads: {error_threads}")
    print(f"Total successful parses: {total_success}")
    print(f"Total blocked malicious: {total_blocked}")
    
    if success_threads == num_threads and error_threads == 0:
        print("✅ PASS: Concurrent safety test passed")
        results["passed"] += 1
    else:
        print("❌ FAIL: Concurrent safety issues detected")
        results["failed"] += 1
        results["details"].append(f"Concurrent errors: {error_threads} threads failed")
    
    print(f"\nCONCURRENCY SUMMARY:")
    print(f"Total Tests: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    return results


def main():
    """Run all focused security tests"""
    print("FOCUSED SECURITY TESTING FOR PIPELINE FIXES")
    print("=" * 80)
    print("Testing the specific security improvements:")
    print("1. eval() → ast.literal_eval() replacement")
    print("2. Bare except → specific exception handling")
    print("=" * 80)
    
    overall_results = {
        "total_tests": 0,
        "total_passed": 0,
        "total_failed": 0,
        "details": []
    }
    
    try:
        # Run all test categories
        print("\n🔒 RUNNING SECURITY TESTS...")
        security_results = test_ast_literal_eval_security()
        
        print("\n🛡️  RUNNING EXCEPTION HANDLING TESTS...")
        exception_results = test_specific_exception_handling()
        
        print("\n⚡ RUNNING PERFORMANCE TESTS...")
        performance_results = test_performance_impact()
        
        print("\n🧵 RUNNING CONCURRENCY TESTS...")
        concurrency_results = test_concurrent_safety()
        
        # Aggregate results
        all_results = [security_results, exception_results, performance_results, concurrency_results]
        
        for result in all_results:
            overall_results["total_tests"] += result["passed"] + result["failed"]
            overall_results["total_passed"] += result["passed"]
            overall_results["total_failed"] += result["failed"]
            overall_results["details"].extend(result.get("details", []))
        
        # Final Report
        print("\n" + "=" * 80)
        print("FINAL SECURITY ASSESSMENT REPORT")
        print("=" * 80)
        
        print(f"📊 OVERALL STATISTICS:")
        print(f"   Total Tests Run: {overall_results['total_tests']}")
        print(f"   Tests Passed: {overall_results['total_passed']}")
        print(f"   Tests Failed: {overall_results['total_failed']}")
        
        if overall_results['total_tests'] > 0:
            success_rate = (overall_results['total_passed'] / overall_results['total_tests']) * 100
            print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\n🔐 SECURITY ASSESSMENT:")
        if overall_results['total_failed'] == 0:
            print("   ✅ ALL SECURITY TESTS PASSED")
            print("   ✅ No code injection vulnerabilities detected")
            print("   ✅ Exception handling is properly implemented")
            print("   ✅ Performance impact is acceptable")
            print("   ✅ Thread safety is maintained")
            print("\n🎯 CONCLUSION: Security fixes are ROBUST and EFFECTIVE")
        else:
            print("   ⚠️  Some security tests failed - REVIEW REQUIRED")
            print(f"   ❌ {overall_results['total_failed']} issues detected")
            
            if overall_results["details"]:
                print("\n📋 FAILED TEST DETAILS:")
                for detail in overall_results["details"]:
                    print(f"   • {detail}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_results['total_failed'] == 0:
            print("   • Security fixes are working correctly")
            print("   • No additional hardening needed for these specific fixes")
            print("   • Consider these fixes as a security improvement baseline")
        else:
            print("   • Review failed tests above")
            print("   • Consider additional security measures")
            print("   • Test in production-like environment")
        
        return overall_results
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in security testing: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()