"""
Pipeline Monitoring and Performance Analysis
===========================================

Real-time monitoring, bottleneck detection, and performance optimization
for the codebase compression pipeline.
"""

import asyncio
import time
import json
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Deque
from datetime import datetime, timedelta
import psutil
import statistics
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage"""
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    items_processed: int = 0
    bytes_processed: int = 0
    errors: int = 0
    memory_start: int = 0
    memory_peak: int = 0
    cpu_percent: float = 0.0
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def throughput_items_per_sec(self) -> float:
        if self.duration > 0:
            return self.items_processed / self.duration
        return 0.0
    
    @property
    def throughput_mb_per_sec(self) -> float:
        if self.duration > 0:
            return (self.bytes_processed / 1024 / 1024) / self.duration
        return 0.0


class PipelineMonitor:
    """Real-time pipeline monitoring and analysis"""
    
    def __init__(self, 
                 window_size: int = 1000,
                 sample_interval: float = 0.1):
        self.window_size = window_size
        self.sample_interval = sample_interval
        
        # Metrics storage
        self.metrics: Dict[str, Deque[MetricPoint]] = defaultdict(lambda: deque(maxlen=window_size))
        self.stage_metrics: Dict[str, StageMetrics] = {}
        self.active_stages: Dict[str, StageMetrics] = {}
        self._metrics_lock = threading.Lock()
        
        # System monitoring
        self.system_monitor = SystemMonitor(sample_interval)
        
        # Performance analysis
        self.bottleneck_detector = BottleneckDetector()
        self.optimizer = PerformanceOptimizer()
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background monitoring"""
        self.monitoring = True
        self.system_monitor.start()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        self.system_monitor.stop()
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            # Collect system metrics
            sys_metrics = self.system_monitor.get_current_metrics()
            
            # Update active stage metrics
            for stage_id, stage in self.active_stages.items():
                stage.cpu_percent = sys_metrics['cpu_percent']
                stage.memory_peak = max(stage.memory_peak, sys_metrics['memory_used'])
            
            time.sleep(self.sample_interval)
    
    def stage_start(self, stage_name: str) -> str:
        """Mark stage start"""
        stage_id = f"{stage_name}_{time.time()}"
        
        stage_metrics = StageMetrics(
            stage_name=stage_name,
            start_time=time.time(),
            memory_start=psutil.Process().memory_info().rss
        )
        
        self.active_stages[stage_id] = stage_metrics
        self.stage_metrics[stage_id] = stage_metrics
        
        return stage_id
    
    def stage_end(self, stage_id: str):
        """Mark stage completion"""
        if stage_id in self.active_stages:
            stage = self.active_stages[stage_id]
            stage.end_time = time.time()
            del self.active_stages[stage_id]
            
            # Record metrics
            self.record_metric(
                f"{stage.stage_name}_duration",
                stage.duration,
                {'stage_id': stage_id}
            )
            self.record_metric(
                f"{stage.stage_name}_throughput",
                stage.throughput_items_per_sec,
                {'stage_id': stage_id}
            )
    
    def record_metric(self, 
                     metric_name: str, 
                     value: float, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric value with thread safety"""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            metadata=metadata or {}
        )
        with self._metrics_lock:
            self.metrics[metric_name].append(point)
    
    def update_stage_progress(self, 
                            stage_id: str, 
                            items: int = 0, 
                            bytes_count: int = 0) -> None:
        """Update stage progress with thread safety"""
        with self._metrics_lock:
            if stage_id in self.active_stages:
                stage = self.active_stages[stage_id]
                stage.items_processed += items
                stage.bytes_processed += bytes_count
    
    def record_error(self, stage_id: str, error: Exception):
        """Record stage error"""
        if stage_id in self.active_stages:
            self.active_stages[stage_id].errors += 1
        
        self.record_metric(
            'errors',
            1,
            {
                'stage_id': stage_id,
                'error_type': type(error).__name__,
                'error_msg': str(error)
            }
        )
    
    def get_stage_summary(self, stage_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for stages"""
        stages = [s for s in self.stage_metrics.values() 
                 if stage_name is None or s.stage_name == stage_name]
        
        if not stages:
            return {}
        
        durations = [s.duration for s in stages if s.end_time]
        throughputs = [s.throughput_items_per_sec for s in stages if s.end_time]
        
        return {
            'count': len(stages),
            'avg_duration': statistics.mean(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'avg_throughput': statistics.mean(throughputs) if throughputs else 0,
            'total_items': sum(s.items_processed for s in stages),
            'total_bytes': sum(s.bytes_processed for s in stages),
            'total_errors': sum(s.errors for s in stages)
        }
    
    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        return self.bottleneck_detector.analyze(
            self.stage_metrics,
            self.metrics
        )
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get performance optimization suggestions"""
        return self.optimizer.analyze(
            self.stage_metrics,
            self.metrics,
            self.system_monitor.get_summary()
        )


class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.metrics = deque(maxlen=1000)
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
    
    def start(self):
        """Start system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self) -> None:
        """System monitoring loop"""
        while self.monitoring:
            try:
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_used': self.process.memory_info().rss,
                    'memory_percent': self.process.memory_percent(),
                    'num_threads': self.process.num_threads(),
                }
                
                # io_counters might not be available on all platforms
                try:
                    io_counters = self.process.io_counters()
                    metrics['io_read_bytes'] = io_counters.read_bytes
                    metrics['io_write_bytes'] = io_counters.write_bytes
                except (AttributeError, psutil.AccessDenied):
                    metrics['io_read_bytes'] = 0
                    metrics['io_write_bytes'] = 0
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.error(f"Error monitoring process: {e}")
                break
            
            # System-wide metrics
            metrics['system_cpu_percent'] = psutil.cpu_percent()
            metrics['system_memory_percent'] = psutil.virtual_memory().percent
            
            self.metrics.append(metrics)
            time.sleep(self.sample_interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get most recent metrics"""
        if self.metrics:
            return self.metrics[-1]
        return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_used'] for m in self.metrics]
        
        return {
            'cpu_avg': statistics.mean(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg': statistics.mean(memory_values),
            'memory_max': max(memory_values),
            'memory_growth': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
        }


class BottleneckDetector:
    """Detect performance bottlenecks in the pipeline"""
    
    def analyze(self, 
               stage_metrics: Dict[str, StageMetrics],
               metrics: Dict[str, deque]) -> List[Dict[str, Any]]:
        """Analyze metrics to find bottlenecks"""
        bottlenecks = []
        
        # Stage duration analysis
        stage_durations = defaultdict(list)
        for stage in stage_metrics.values():
            if stage.end_time:
                stage_durations[stage.stage_name].append(stage.duration)
        
        # Find slowest stages
        avg_durations = {
            name: statistics.mean(durations) 
            for name, durations in stage_durations.items()
        }
        
        if avg_durations:
            max_duration = max(avg_durations.values())
            for stage_name, avg_duration in avg_durations.items():
                if avg_duration > max_duration * 0.5:  # Stage takes >50% of max time
                    bottlenecks.append({
                        'type': 'slow_stage',
                        'stage': stage_name,
                        'severity': 'high' if avg_duration == max_duration else 'medium',
                        'avg_duration': avg_duration,
                        'suggestion': f"Optimize {stage_name} stage - taking {avg_duration:.2f}s on average"
                    })
        
        # Memory growth analysis
        for stage in stage_metrics.values():
            if stage.memory_peak > stage.memory_start * 2:  # 2x memory growth
                bottlenecks.append({
                    'type': 'memory_growth',
                    'stage': stage.stage_name,
                    'severity': 'high',
                    'memory_growth': stage.memory_peak - stage.memory_start,
                    'suggestion': f"High memory growth in {stage.stage_name} - consider streaming or chunking"
                })
        
        # Error rate analysis
        error_stages = defaultdict(int)
        if 'errors' in metrics:
            for point in metrics['errors']:
                if 'stage_id' in point.metadata:
                    stage_id = point.metadata['stage_id']
                    if stage_id in stage_metrics:
                        error_stages[stage_metrics[stage_id].stage_name] += 1
        
        for stage_name, error_count in error_stages.items():
            if error_count > 10:  # High error threshold
                bottlenecks.append({
                    'type': 'high_error_rate',
                    'stage': stage_name,
                    'severity': 'high',
                    'error_count': error_count,
                    'suggestion': f"High error rate in {stage_name} - {error_count} errors detected"
                })
        
        # I/O bottleneck detection
        io_wait_stages = []
        for stage in stage_metrics.values():
            if stage.cpu_percent < 50 and stage.throughput_mb_per_sec < 10:
                io_wait_stages.append(stage.stage_name)
        
        if io_wait_stages:
            bottlenecks.append({
                'type': 'io_bottleneck',
                'stages': io_wait_stages,
                'severity': 'medium',
                'suggestion': "Low CPU usage with low throughput suggests I/O bottleneck"
            })
        
        return bottlenecks


class PerformanceOptimizer:
    """Suggest performance optimizations"""
    
    def analyze(self,
               stage_metrics: Dict[str, StageMetrics],
               metrics: Dict[str, deque],
               system_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze and suggest optimizations"""
        suggestions = []
        
        # CPU utilization optimization
        if system_summary.get('cpu_avg', 0) < 50:
            suggestions.append({
                'category': 'resource_utilization',
                'suggestion': 'Low CPU utilization - consider increasing worker count',
                'impact': 'high',
                'implementation': 'Increase num_workers in configuration'
            })
        elif system_summary.get('cpu_max', 0) > 90:
            suggestions.append({
                'category': 'resource_utilization',
                'suggestion': 'High CPU utilization - may be CPU bound',
                'impact': 'medium',
                'implementation': 'Consider reducing compression level or using faster algorithms'
            })
        
        # Memory optimization
        memory_growth = system_summary.get('memory_growth', 0)
        if memory_growth > 1024 * 1024 * 1024:  # >1GB growth
            suggestions.append({
                'category': 'memory_management',
                'suggestion': 'Significant memory growth detected',
                'impact': 'high',
                'implementation': 'Implement streaming processing or reduce batch sizes'
            })
        
        # Stage-specific optimizations
        stage_throughputs = defaultdict(list)
        for stage in stage_metrics.values():
            if stage.end_time and stage.items_processed > 0:
                stage_throughputs[stage.stage_name].append(stage.throughput_items_per_sec)
        
        for stage_name, throughputs in stage_throughputs.items():
            avg_throughput = statistics.mean(throughputs)
            
            if avg_throughput < 100:  # Low throughput threshold
                if 'parsing' in stage_name.lower():
                    suggestions.append({
                        'category': 'stage_optimization',
                        'stage': stage_name,
                        'suggestion': 'Low parsing throughput',
                        'impact': 'high',
                        'implementation': 'Consider using faster parsers or regex-based extraction'
                    })
                elif 'compression' in stage_name.lower():
                    suggestions.append({
                        'category': 'stage_optimization',
                        'stage': stage_name,
                        'suggestion': 'Low compression throughput',
                        'impact': 'medium',
                        'implementation': 'Reduce compression level or use faster algorithm (LZ4)'
                    })
        
        # Batch size optimization
        batch_metrics = [m for m in metrics.get('batch_size', []) if m]
        if batch_metrics:
            batch_sizes = [m.value for m in batch_metrics]
            avg_batch_size = statistics.mean(batch_sizes)
            
            if avg_batch_size < 10:
                suggestions.append({
                    'category': 'batching',
                    'suggestion': 'Small batch sizes detected',
                    'impact': 'medium',
                    'implementation': 'Increase batch_size to reduce overhead'
                })
            elif avg_batch_size > 1000:
                suggestions.append({
                    'category': 'batching',
                    'suggestion': 'Large batch sizes may cause memory spikes',
                    'impact': 'low',
                    'implementation': 'Consider reducing batch_size for smoother memory usage'
                })
        
        # Cache effectiveness
        cache_hits = sum(1 for m in metrics.get('cache_hit', []) if m.value == 1)
        cache_misses = sum(1 for m in metrics.get('cache_hit', []) if m.value == 0)
        
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses)
            if hit_rate < 0.5:
                suggestions.append({
                    'category': 'caching',
                    'suggestion': f'Low cache hit rate: {hit_rate:.1%}',
                    'impact': 'medium',
                    'implementation': 'Increase cache size or TTL'
                })
        
        return suggestions


class MetricsExporter:
    """Export metrics in various formats"""
    
    @staticmethod
    def to_prometheus(monitor: PipelineMonitor) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Stage metrics
        for stage_id, stage in monitor.stage_metrics.items():
            if stage.end_time:
                lines.append(
                    f'pipeline_stage_duration_seconds{{stage="{stage.stage_name}"}} {stage.duration}'
                )
                lines.append(
                    f'pipeline_stage_items_total{{stage="{stage.stage_name}"}} {stage.items_processed}'
                )
                lines.append(
                    f'pipeline_stage_bytes_total{{stage="{stage.stage_name}"}} {stage.bytes_processed}'
                )
                lines.append(
                    f'pipeline_stage_errors_total{{stage="{stage.stage_name}"}} {stage.errors}'
                )
        
        # Current metrics
        for metric_name, points in monitor.metrics.items():
            if points:
                latest = points[-1]
                lines.append(f'pipeline_{metric_name} {latest.value}')
        
        return '\n'.join(lines)
    
    @staticmethod
    def to_json(monitor: PipelineMonitor) -> str:
        """Export metrics as JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'metrics': {},
            'system': monitor.system_monitor.get_summary()
        }
        
        # Stage summaries
        for stage_name in set(s.stage_name for s in monitor.stage_metrics.values()):
            data['stages'][stage_name] = monitor.get_stage_summary(stage_name)
        
        # Recent metrics
        for metric_name, points in monitor.metrics.items():
            if points:
                recent_points = list(points)[-100:]  # Last 100 points
                data['metrics'][metric_name] = [
                    {
                        'timestamp': p.timestamp,
                        'value': p.value,
                        'metadata': p.metadata
                    }
                    for p in recent_points
                ]
        
        # Analysis results
        data['bottlenecks'] = monitor.get_bottlenecks()
        data['optimizations'] = monitor.get_optimization_suggestions()
        
        return json.dumps(data, indent=2)
    
    @staticmethod
    def to_grafana_dashboard(monitor: PipelineMonitor) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration"""
        return {
            'dashboard': {
                'title': 'Codebase Compression Pipeline',
                'panels': [
                    {
                        'title': 'Stage Durations',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'pipeline_stage_duration_seconds',
                                'legendFormat': '{{stage}}'
                            }
                        ]
                    },
                    {
                        'title': 'Throughput',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'rate(pipeline_stage_items_total[5m])',
                                'legendFormat': '{{stage}} items/sec'
                            }
                        ]
                    },
                    {
                        'title': 'Memory Usage',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'pipeline_memory_usage_bytes',
                                'legendFormat': 'Memory'
                            }
                        ]
                    },
                    {
                        'title': 'Error Rate',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'rate(pipeline_stage_errors_total[5m])',
                                'legendFormat': '{{stage}} errors/sec'
                            }
                        ]
                    }
                ]
            }
        }


# Example usage with context manager
class MonitoredPipeline:
    """Context manager for monitored pipeline execution"""
    
    def __init__(self, monitor: PipelineMonitor):
        self.monitor = monitor
        self.stage_stack = []
    
    def __enter__(self):
        self.monitor.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop_monitoring()
        
        # Export final metrics
        report_path = Path('./monitoring_report.json')
        try:
            with open(report_path, 'w') as f:
                f.write(MetricsExporter.to_json(self.monitor))
            logger.info(f"Monitoring report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save monitoring report: {e}")
    
    def stage(self, stage_name: str):
        """Context manager for monitoring a stage"""
        return MonitoredStage(self.monitor, stage_name)


class MonitoredStage:
    """Context manager for monitoring a single stage"""
    
    def __init__(self, monitor: PipelineMonitor, stage_name: str):
        self.monitor = monitor
        self.stage_name = stage_name
        self.stage_id = None
    
    def __enter__(self):
        self.stage_id = self.monitor.stage_start(self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.monitor.record_error(self.stage_id, exc_val)
        self.monitor.stage_end(self.stage_id)
    
    def update_progress(self, items: int = 0, bytes_count: int = 0):
        """Update stage progress"""
        self.monitor.update_stage_progress(self.stage_id, items, bytes_count)


# Usage example
async def example_monitored_pipeline():
    """Example of using monitoring with the pipeline"""
    monitor = PipelineMonitor()
    
    with MonitoredPipeline(monitor) as pipeline:
        # Stage 1: File discovery
        with pipeline.stage('file_discovery') as stage:
            files = ['file1.py', 'file2.py', 'file3.py']
            stage.update_progress(items=len(files))
        
        # Stage 2: Parsing
        with pipeline.stage('parsing') as stage:
            for file in files:
                # Simulate parsing
                await asyncio.sleep(0.1)
                stage.update_progress(items=1, bytes_count=1024)
        
        # Stage 3: Compression
        with pipeline.stage('compression') as stage:
            # Simulate compression
            await asyncio.sleep(0.2)
            stage.update_progress(items=len(files), bytes_count=512 * len(files))
        
        # Get real-time analysis
        bottlenecks = monitor.get_bottlenecks()
        suggestions = monitor.get_optimization_suggestions()
        
        logger.info(f"Bottlenecks: {bottlenecks}")
        logger.info(f"Suggestions: {suggestions}")