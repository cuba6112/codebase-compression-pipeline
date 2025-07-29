"""
Resilience Patterns for Codebase Compression Pipeline
=====================================================

Implements robust error recovery, retry mechanisms, circuit breakers,
and graceful degradation for production-ready pipeline operations.
"""

import asyncio
import time
import random
from enum import Enum
from typing import Callable, Any, Optional, Dict, List, TypeVar, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from functools import wraps
import traceback
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter"""
        delay = min(self.initial_delay * (self.exponential_base ** (attempt - 1)), self.max_delay)
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    success_threshold: int = 2
    

@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    consecutive_successes: int = 0


class RetryableError(Exception):
    """Base class for retryable errors"""
    pass


class NonRetryableError(Exception):
    """Base class for non-retryable errors"""
    pass


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to functions"""
    if config is None:
        config = RetryConfig()
        
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt == config.max_attempts:
                        logger.error(f"Failed after {config.max_attempts} attempts: {func.__name__}")
                        raise
                        
                    delay = config.calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                except NonRetryableError:
                    logger.error(f"Non-retryable error in {func.__name__}")
                    raise
                    
            raise last_exception
            
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt == config.max_attempts:
                        logger.error(f"Failed after {config.max_attempts} attempts: {func.__name__}")
                        raise
                        
                    delay = config.calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                except NonRetryableError:
                    logger.error(f"Non-retryable error in {func.__name__}")
                    raise
                    
            raise last_exception
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            self._check_state()
            
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
            
    def _check_state(self):
        """Check if circuit should be opened or closed"""
        if self.state.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")
                
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (self.state.last_failure_time and 
                time.time() - self.state.last_failure_time >= self.config.recovery_timeout)
                
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            if self.state.state == CircuitState.HALF_OPEN:
                self.state.consecutive_successes += 1
                if self.state.consecutive_successes >= self.config.success_threshold:
                    self.state.state = CircuitState.CLOSED
                    self.state.failure_count = 0
                    self.state.consecutive_successes = 0
                    logger.info(f"Circuit breaker '{self.name}' is now CLOSED")
            elif self.state.state == CircuitState.CLOSED:
                self.state.failure_count = 0
                
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.state.failure_count += 1
            self.state.last_failure_time = time.time()
            self.state.consecutive_successes = 0
            
            if self.state.state == CircuitState.HALF_OPEN:
                self.state.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' is now OPEN (failed in HALF_OPEN)")
            elif (self.state.state == CircuitState.CLOSED and 
                  self.state.failure_count >= self.config.failure_threshold):
                self.state.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' is now OPEN (threshold reached)")


class BulkheadExecutor:
    """Bulkhead pattern to isolate resources and prevent cascading failures"""
    
    def __init__(self, name: str, max_concurrent: int = 10, queue_size: int = 100):
        self.name = name
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue: deque = deque(maxlen=queue_size)
        self.active_count = 0
        self.rejected_count = 0
        
    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with bulkhead protection"""
        if self.active_count >= self.max_concurrent and len(self.queue) >= self.queue_size:
            self.rejected_count += 1
            raise RuntimeError(f"Bulkhead '{self.name}' queue is full")
            
        async with self.semaphore:
            self.active_count += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self.active_count -= 1
                
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics"""
        return {
            'name': self.name,
            'active': self.active_count,
            'queue_length': len(self.queue),
            'rejected': self.rejected_count,
            'capacity': self.max_concurrent
        }


class TimeoutHandler:
    """Handles timeout with graceful cancellation"""
    
    @staticmethod
    async def with_timeout(coro: Awaitable[T], timeout: float, 
                          cleanup: Optional[Callable] = None) -> T:
        """Execute coroutine with timeout and optional cleanup"""
        try:
            return await asyncio.wait_for(coro, timeout)
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {timeout}s")
            if cleanup:
                try:
                    await cleanup() if asyncio.iscoroutinefunction(cleanup) else cleanup()
                except Exception as e:
                    logger.error(f"Cleanup failed: {e}")
            raise


class FallbackHandler:
    """Provides fallback mechanisms for graceful degradation"""
    
    def __init__(self):
        self.fallback_chain: List[Callable] = []
        
    def add_fallback(self, fallback: Callable) -> 'FallbackHandler':
        """Add a fallback function to the chain"""
        self.fallback_chain.append(fallback)
        return self
        
    async def execute_with_fallback(self, primary: Callable[..., Awaitable[T]], 
                                   *args, **kwargs) -> T:
        """Execute primary function with fallback chain"""
        exceptions = []
        
        # Try primary function
        try:
            return await primary(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed: {e}")
            exceptions.append(('primary', e))
            
        # Try fallbacks in order
        for i, fallback in enumerate(self.fallback_chain):
            try:
                logger.info(f"Attempting fallback {i+1}")
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                else:
                    return fallback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback {i+1} failed: {e}")
                exceptions.append((f'fallback_{i+1}', e))
                
        # All attempts failed
        raise RuntimeError(f"All attempts failed: {exceptions}")


class HealthChecker:
    """Monitors component health and triggers recovery actions"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.components: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self._task = None
        
    def register_component(self, name: str, 
                         health_check: Callable[[], Awaitable[bool]],
                         recovery_action: Optional[Callable] = None):
        """Register a component for health monitoring"""
        self.components[name] = {
            'check': health_check,
            'recovery': recovery_action,
            'healthy': True,
            'last_check': None,
            'consecutive_failures': 0
        }
        
    async def start(self):
        """Start health monitoring"""
        self.running = True
        self._task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Stop health monitoring"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            await self._check_all_components()
            await asyncio.sleep(self.check_interval)
            
    async def _check_all_components(self):
        """Check health of all registered components"""
        for name, component in self.components.items():
            try:
                is_healthy = await component['check']()
                component['last_check'] = datetime.now()
                
                if is_healthy:
                    if not component['healthy']:
                        logger.info(f"Component '{name}' recovered")
                    component['healthy'] = True
                    component['consecutive_failures'] = 0
                else:
                    component['healthy'] = False
                    component['consecutive_failures'] += 1
                    logger.warning(f"Component '{name}' is unhealthy")
                    
                    # Trigger recovery if available
                    if component['recovery'] and component['consecutive_failures'] >= 3:
                        logger.info(f"Attempting recovery for '{name}'")
                        try:
                            if asyncio.iscoroutinefunction(component['recovery']):
                                await component['recovery']()
                            else:
                                component['recovery']()
                        except Exception as e:
                            logger.error(f"Recovery failed for '{name}': {e}")
                            
            except Exception as e:
                logger.error(f"Health check failed for '{name}': {e}")
                component['healthy'] = False
                component['consecutive_failures'] += 1
                
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all components"""
        return {
            name: {
                'healthy': comp['healthy'],
                'last_check': comp['last_check'],
                'consecutive_failures': comp['consecutive_failures']
            }
            for name, comp in self.components.items()
        }


# Example usage patterns
async def example_resilient_operation():
    """Example of using resilience patterns together"""
    
    # Configure retry behavior
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        exponential_base=2.0,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )
    
    # Configure circuit breaker
    circuit_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        expected_exception=ConnectionError
    )
    
    # Create components
    circuit_breaker = CircuitBreaker("api_service", circuit_config)
    bulkhead = BulkheadExecutor("api_bulkhead", max_concurrent=5)
    fallback_handler = FallbackHandler()
    
    # Add fallback options
    fallback_handler.add_fallback(lambda: {"source": "cache", "data": "cached_value"})
    fallback_handler.add_fallback(lambda: {"source": "default", "data": "default_value"})
    
    # Define the main operation with retry
    @with_retry(retry_config)
    async def api_call():
        # Simulate API call
        await asyncio.sleep(0.1)
        if random.random() < 0.3:  # 30% failure rate
            raise ConnectionError("API unavailable")
        return {"source": "api", "data": "fresh_value"}
    
    # Execute with all patterns
    async def resilient_api_call():
        async def protected_call():
            return await circuit_breaker.call(
                bulkhead.execute,
                api_call
            )
            
        return await fallback_handler.execute_with_fallback(protected_call)
    
    # Use the resilient operation
    try:
        result = await resilient_api_call()
        logger.info(f"Success: {result}")
    except Exception as e:
        logger.error(f"All resilience measures failed: {e}")