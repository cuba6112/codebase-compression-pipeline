"""
Performance Optimizer
=====================

Optimize pipeline performance through profiling and tuning.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Optimize pipeline performance through profiling and tuning"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.profiling_enabled = False
    
    def profile_stage(self, stage_name: str):
        """Decorator to profile pipeline stages"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if self.profiling_enabled:
                    start_time = time.time()
                    start_memory = self._get_memory_usage()
                    
                    result = await func(*args, **kwargs)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    self.metrics[stage_name].append({
                        'duration': end_time - start_time,
                        'memory_delta': end_memory - start_memory,
                        'timestamp': datetime.now()
                    })
                    
                    return result
                else:
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def optimize_batch_size(self, 
                           file_sizes: List[int], 
                           available_memory: int) -> int:
        """Calculate optimal batch size based on file sizes and memory"""
        avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 1024 * 1024
        
        # Reserve 50% memory for processing overhead
        usable_memory = available_memory * 0.5
        
        # Calculate batch size with safety margin
        batch_size = max(1, int(usable_memory / (avg_file_size * 2)))
        
        # Cap at reasonable limits
        return min(batch_size, 1000)
    
    def optimize_compression_level(self, 
                                  content_type: str, 
                                  size: int) -> int:
        """Determine optimal compression level"""
        if content_type == 'already_compressed':  # e.g., images, videos
            return 0
        elif size < 1024:  # Small files
            return 1
        elif size < 1024 * 1024:  # Medium files
            return 6
        else:  # Large files
            return 9
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {}
        
        for stage, metrics in self.metrics.items():
            if metrics:
                durations = [m['duration'] for m in metrics]
                memory_deltas = [m['memory_delta'] for m in metrics]
                
                report[stage] = {
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
                    'total_executions': len(metrics)
                }
        
        return report
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss