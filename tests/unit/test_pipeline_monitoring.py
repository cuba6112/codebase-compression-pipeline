"""
Unit tests for pipeline monitoring and performance analysis
==========================================================

Tests for pipeline_monitoring.py including:
- PipelineMonitor functionality
- SystemMonitor resource tracking
- BottleneckDetector analysis
- PerformanceOptimizer suggestions
- MetricsExporter output formats
- Context managers and stage tracking
"""

import pytest
import asyncio
import time
import json
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import deque, defaultdict
from pathlib import Path
import tempfile

# Import the classes to test
from pipeline_monitoring import (
    MetricPoint, StageMetrics, PipelineMonitor, SystemMonitor,
    BottleneckDetector, PerformanceOptimizer, MetricsExporter,
    MonitoredPipeline, MonitoredStage
)


class TestMetricPoint:
    """Test MetricPoint dataclass functionality"""
    
    def test_metric_point_creation(self):
        """Test basic MetricPoint creation"""
        point = MetricPoint(
            timestamp=1234567890.0,
            value=42.5,
            metadata={'stage': 'parsing', 'file_count': 10}
        )
        
        assert point.timestamp == 1234567890.0
        assert point.value == 42.5
        assert point.metadata['stage'] == 'parsing'
        assert point.metadata['file_count'] == 10
    
    def test_metric_point_default_metadata(self):
        """Test MetricPoint with default empty metadata"""
        point = MetricPoint(timestamp=time.time(), value=123.4)
        
        assert isinstance(point.metadata, dict)
        assert len(point.metadata) == 0
    
    def test_metric_point_equality(self):
        """Test MetricPoint equality comparison"""
        timestamp = time.time()
        point1 = MetricPoint(timestamp=timestamp, value=10.0)
        point2 = MetricPoint(timestamp=timestamp, value=10.0)
        point3 = MetricPoint(timestamp=timestamp, value=20.0)
        
        assert point1 == point2
        assert point1 != point3


class TestStageMetrics:
    """Test StageMetrics dataclass and computed properties"""
    
    def test_stage_metrics_creation(self):
        """Test basic StageMetrics creation"""
        start_time = time.time()
        metrics = StageMetrics(
            stage_name="parsing",
            start_time=start_time,
            items_processed=100,
            bytes_processed=50000,
            errors=2,
            memory_start=1000000,
            memory_peak=1500000,
            cpu_percent=75.5
        )
        
        assert metrics.stage_name == "parsing"
        assert metrics.start_time == start_time
        assert metrics.items_processed == 100
        assert metrics.bytes_processed == 50000
        assert metrics.errors == 2
        assert metrics.memory_start == 1000000
        assert metrics.memory_peak == 1500000
        assert metrics.cpu_percent == 75.5
        assert metrics.end_time is None
    
    def test_stage_metrics_duration_ongoing(self):
        """Test duration calculation for ongoing stage"""
        start_time = time.time() - 5.0  # Started 5 seconds ago
        metrics = StageMetrics(stage_name="test", start_time=start_time)
        
        duration = metrics.duration
        assert 4.8 <= duration <= 5.2  # Allow some timing variance
    
    def test_stage_metrics_duration_completed(self):
        """Test duration calculation for completed stage"""
        start_time = time.time()
        end_time = start_time + 3.0
        metrics = StageMetrics(
            stage_name="test",
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics.duration == 3.0
    
    def test_throughput_items_per_sec(self):
        """Test items per second throughput calculation"""
        metrics = StageMetrics(
            stage_name="test",
            start_time=time.time() - 4.0,  # 4 seconds ago
            end_time=time.time(),
            items_processed=200
        )
        
        throughput = metrics.throughput_items_per_sec
        assert 49.0 <= throughput <= 51.0  # ~50 items/sec
    
    def test_throughput_mb_per_sec(self):
        """Test MB per second throughput calculation"""
        metrics = StageMetrics(
            stage_name="test",
            start_time=time.time() - 2.0,  # 2 seconds ago
            end_time=time.time(),
            bytes_processed=4 * 1024 * 1024  # 4 MB
        )
        
        throughput = metrics.throughput_mb_per_sec
        assert 1.9 <= throughput <= 2.1  # ~2 MB/sec
    
    def test_zero_duration_throughput(self):
        """Test throughput calculation with zero duration"""
        metrics = StageMetrics(
            stage_name="test",
            start_time=time.time(),
            items_processed=100,
            bytes_processed=1000
        )
        
        # Should return 0 for zero duration to avoid division by zero
        assert metrics.throughput_items_per_sec == 0.0
        assert metrics.throughput_mb_per_sec == 0.0


class TestPipelineMonitor:
    """Test PipelineMonitor main functionality"""
    
    @pytest.fixture
    def monitor(self):
        """Create a PipelineMonitor instance for testing"""
        with patch('pipeline_monitoring.SystemMonitor'), \
             patch('pipeline_monitoring.BottleneckDetector'), \
             patch('pipeline_monitoring.PerformanceOptimizer'):
            return PipelineMonitor(window_size=100, sample_interval=0.01)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.window_size == 100
        assert monitor.sample_interval == 0.01
        assert isinstance(monitor.metrics, defaultdict)
        assert isinstance(monitor.stage_metrics, dict)
        assert isinstance(monitor.active_stages, dict)
        assert monitor.monitoring is False
        assert monitor.monitor_thread is None
    
    def test_record_metric(self, monitor):
        """Test metric recording functionality"""
        monitor.record_metric("test_metric", 42.5, {"test": "metadata"})
        
        assert "test_metric" in monitor.metrics
        assert len(monitor.metrics["test_metric"]) == 1
        
        point = monitor.metrics["test_metric"][0]
        assert point.value == 42.5
        assert point.metadata["test"] == "metadata"
    
    def test_record_metric_thread_safety(self, monitor):
        """Test thread safety of metric recording"""
        def worker(worker_id):
            for i in range(50):
                monitor.record_metric(f"worker_{worker_id}_metric", i, {"worker": worker_id})
        
        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have metrics from all workers
        total_metrics = sum(len(monitor.metrics[key]) for key in monitor.metrics)
        assert total_metrics == 250  # 5 workers * 50 metrics each
    
    def test_stage_start_and_end(self, monitor):
        """Test stage lifecycle tracking"""
        stage_id = monitor.stage_start("parsing")
        
        assert stage_id.startswith("parsing_")
        assert stage_id in monitor.active_stages
        assert stage_id in monitor.stage_metrics
        
        stage = monitor.active_stages[stage_id]
        assert stage.stage_name == "parsing"
        assert stage.start_time > 0
        assert stage.end_time is None
        
        # End the stage
        monitor.stage_end(stage_id)
        
        assert stage_id not in monitor.active_stages
        assert stage_id in monitor.stage_metrics
        assert monitor.stage_metrics[stage_id].end_time is not None
    
    def test_update_stage_progress(self, monitor):
        """Test stage progress updates"""
        stage_id = monitor.stage_start("compression")
        
        monitor.update_stage_progress(stage_id, items=50, bytes_count=25000)
        monitor.update_stage_progress(stage_id, items=30, bytes_count=15000)
        
        stage = monitor.active_stages[stage_id]
        assert stage.items_processed == 80
        assert stage.bytes_processed == 40000
        
        monitor.stage_end(stage_id)
    
    def test_update_stage_progress_thread_safety(self, monitor):
        """Test thread safety of stage progress updates"""
        stage_id = monitor.stage_start("parallel_processing")
        
        def worker():
            for i in range(100):
                monitor.update_stage_progress(stage_id, items=1, bytes_count=1024)
        
        # Run multiple threads updating the same stage
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        stage = monitor.active_stages[stage_id]
        assert stage.items_processed == 300  # 3 workers * 100 items each
        assert stage.bytes_processed == 300 * 1024  # 3 workers * 100 * 1024 bytes
        
        monitor.stage_end(stage_id)
    
    def test_record_error(self, monitor):
        """Test error recording"""
        stage_id = monitor.stage_start("error_prone_stage")
        test_error = ValueError("Test error message")
        
        monitor.record_error(stage_id, test_error)
        
        # Check stage error count
        assert monitor.active_stages[stage_id].errors == 1
        
        # Check error metric
        assert "errors" in monitor.metrics
        error_points = list(monitor.metrics["errors"])
        assert len(error_points) == 1
        assert error_points[0].metadata["error_type"] == "ValueError"
        assert "Test error message" in error_points[0].metadata["error_msg"]
        
        monitor.stage_end(stage_id)
    
    def test_get_stage_summary_single_stage(self, monitor):
        """Test stage summary for specific stage"""
        # Create and complete a stage
        stage_id = monitor.stage_start("test_stage")
        monitor.update_stage_progress(stage_id, items=100, bytes_count=50000)
        time.sleep(0.1)  # Ensure some duration
        monitor.stage_end(stage_id)
        
        summary = monitor.get_stage_summary("test_stage")
        
        assert summary["count"] == 1
        assert summary["total_items"] == 100
        assert summary["total_bytes"] == 50000
        assert summary["total_errors"] == 0
        assert summary["avg_duration"] > 0
        assert summary["avg_throughput"] > 0
    
    def test_get_stage_summary_multiple_stages(self, monitor):
        """Test stage summary across multiple stage instances"""
        # Create multiple instances of the same stage type
        for i in range(3):
            stage_id = monitor.stage_start("batch_processing")
            monitor.update_stage_progress(stage_id, items=50 + i * 10, bytes_count=25000)
            time.sleep(0.05)
            monitor.stage_end(stage_id)
        
        summary = monitor.get_stage_summary("batch_processing")
        
        assert summary["count"] == 3
        assert summary["total_items"] == 50 + 60 + 70  # 180 total
        assert summary["total_bytes"] == 3 * 25000  # 75000 total
        assert summary["avg_duration"] > 0
    
    def test_get_stage_summary_no_stages(self, monitor):
        """Test stage summary for non-existent stage"""
        summary = monitor.get_stage_summary("non_existent_stage")
        assert summary == {}
    
    def test_start_stop_monitoring(self, monitor):
        """Test monitoring thread lifecycle"""
        assert monitor.monitoring is False
        assert monitor.monitor_thread is None
        
        monitor.start_monitoring()
        assert monitor.monitoring is True
        assert monitor.monitor_thread is not None
        assert monitor.monitor_thread.daemon is True
        
        time.sleep(0.1)  # Let it run briefly
        
        monitor.stop_monitoring()
        assert monitor.monitoring is False
    
    def test_metrics_window_size_limit(self, monitor):
        """Test that metrics respect window size limit"""
        # Add more metrics than window size
        for i in range(150):  # Window size is 100
            monitor.record_metric("test_metric", i)
        
        # Should only keep the last 100 metrics
        assert len(monitor.metrics["test_metric"]) == 100
        
        # Should have the most recent values
        values = [point.value for point in monitor.metrics["test_metric"]]
        assert min(values) == 50  # Should start from 50 (150-100)
        assert max(values) == 149


class TestSystemMonitor:
    """Test SystemMonitor resource tracking"""
    
    @pytest.fixture
    def system_monitor(self):
        """Create a SystemMonitor instance for testing"""
        return SystemMonitor(sample_interval=0.01)
    
    def test_system_monitor_initialization(self, system_monitor):
        """Test system monitor initialization"""
        assert system_monitor.sample_interval == 0.01
        assert isinstance(system_monitor.metrics, deque)
        assert system_monitor.monitoring is False
        assert system_monitor.monitor_thread is None
    
    @patch('psutil.Process')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_monitor_loop_single_iteration(self, mock_vm, mock_cpu, mock_process, system_monitor):
        """Test single iteration of monitoring loop"""
        # Mock process metrics
        mock_proc_instance = Mock()
        mock_proc_instance.cpu_percent.return_value = 25.5
        mock_proc_instance.memory_info.return_value = Mock(rss=1024*1024*100)  # 100MB
        mock_proc_instance.memory_percent.return_value = 5.0
        mock_proc_instance.num_threads.return_value = 4
        mock_proc_instance.io_counters.return_value = Mock(read_bytes=1000, write_bytes=2000)
        mock_process.return_value = mock_proc_instance
        
        # Mock system metrics
        mock_cpu.return_value = 15.0
        mock_vm.return_value = Mock(percent=30.0)
        
        # Run single monitoring iteration
        system_monitor.monitoring = True
        system_monitor._monitor_loop()
        
        # Should have collected metrics
        assert len(system_monitor.metrics) > 0
        
        latest = system_monitor.metrics[-1]
        assert latest['cpu_percent'] == 25.5
        assert latest['memory_used'] == 1024*1024*100
        assert latest['memory_percent'] == 5.0
        assert latest['num_threads'] == 4
        assert latest['io_read_bytes'] == 1000
        assert latest['io_write_bytes'] == 2000
        assert latest['system_cpu_percent'] == 15.0
        assert latest['system_memory_percent'] == 30.0
    
    @patch('psutil.Process')
    def test_monitor_loop_io_counters_unavailable(self, mock_process, system_monitor):
        """Test monitoring when I/O counters are unavailable"""
        mock_proc_instance = Mock()
        mock_proc_instance.cpu_percent.return_value = 25.0
        mock_proc_instance.memory_info.return_value = Mock(rss=1024*1024)
        mock_proc_instance.memory_percent.return_value = 2.0
        mock_proc_instance.num_threads.return_value = 2
        mock_proc_instance.io_counters.side_effect = AttributeError("Not available")
        mock_process.return_value = mock_proc_instance
        
        system_monitor.monitoring = True
        system_monitor._monitor_loop()
        
        latest = system_monitor.metrics[-1]
        assert latest['io_read_bytes'] == 0
        assert latest['io_write_bytes'] == 0
    
    @patch('psutil.Process')
    def test_monitor_loop_process_error(self, mock_process, system_monitor):
        """Test monitoring with process errors"""
        mock_proc_instance = Mock()
        mock_proc_instance.cpu_percent.side_effect = psutil.NoSuchProcess(123)
        mock_process.return_value = mock_proc_instance
        
        system_monitor.monitoring = True
        # Should exit gracefully on process error
        system_monitor._monitor_loop()
    
    def test_get_current_metrics_empty(self, system_monitor):
        """Test getting current metrics when no data available"""
        metrics = system_monitor.get_current_metrics()
        assert metrics == {}
    
    def test_get_current_metrics_with_data(self, system_monitor):
        """Test getting current metrics with data"""
        # Add test data
        test_metrics = {
            'timestamp': time.time(),
            'cpu_percent': 50.0,
            'memory_used': 1024*1024*200
        }
        system_monitor.metrics.append(test_metrics)
        
        current = system_monitor.get_current_metrics()
        assert current == test_metrics
    
    def test_get_summary_empty(self, system_monitor):
        """Test getting summary with no data"""
        summary = system_monitor.get_summary()
        assert summary == {}
    
    def test_get_summary_with_data(self, system_monitor):
        """Test getting summary statistics"""
        # Add test data points
        for i in range(10):
            system_monitor.metrics.append({
                'timestamp': time.time(),
                'cpu_percent': 10.0 + i * 5,  # 10, 15, 20, ..., 55
                'memory_used': 1000000 + i * 100000  # Increasing memory
            })
        
        summary = system_monitor.get_summary()
        
        assert summary['cpu_avg'] == 32.5  # Average of 10-55
        assert summary['cpu_max'] == 55.0
        assert summary['memory_avg'] == 1450000  # Average memory
        assert summary['memory_max'] == 1900000  # Max memory
        assert summary['memory_growth'] == 900000  # Growth from first to last
    
    def test_start_stop_monitoring(self, system_monitor):
        """Test monitoring thread lifecycle"""
        assert system_monitor.monitoring is False
        
        system_monitor.start()
        assert system_monitor.monitoring is True
        assert system_monitor.monitor_thread is not None
        
        time.sleep(0.1)  # Let it run briefly
        
        system_monitor.stop()
        assert system_monitor.monitoring is False


class TestBottleneckDetector:
    """Test BottleneckDetector analysis functionality"""
    
    @pytest.fixture
    def detector(self):
        """Create a BottleneckDetector instance for testing"""
        return BottleneckDetector()
    
    @pytest.fixture
    def sample_stage_metrics(self):
        """Create sample stage metrics for testing"""
        metrics = {}
        
        # Fast stage
        fast_stage = StageMetrics(
            stage_name="fast_stage",
            start_time=time.time() - 2.0,
            end_time=time.time() - 1.0,
            items_processed=100,
            bytes_processed=50000,
            memory_start=1000000,
            memory_peak=1100000,
            cpu_percent=80.0
        )
        metrics["fast_1"] = fast_stage
        
        # Slow stage
        slow_stage = StageMetrics(
            stage_name="slow_stage", 
            start_time=time.time() - 10.0,
            end_time=time.time(),
            items_processed=50,
            bytes_processed=25000,
            memory_start=1000000,
            memory_peak=1200000,
            cpu_percent=85.0
        )
        metrics["slow_1"] = slow_stage
        
        # Memory-hungry stage
        memory_stage = StageMetrics(
            stage_name="memory_stage",
            start_time=time.time() - 3.0,
            end_time=time.time() - 1.0,
            items_processed=75,
            bytes_processed=40000,
            memory_start=1000000,
            memory_peak=5000000,  # 5x growth
            cpu_percent=60.0
        )
        metrics["memory_1"] = memory_stage
        
        # I/O bound stage
        io_stage = StageMetrics(
            stage_name="io_stage",
            start_time=time.time() - 5.0,
            end_time=time.time() - 1.0,
            items_processed=20,
            bytes_processed=10000,
            memory_start=1000000,
            memory_peak=1050000,
            cpu_percent=25.0  # Low CPU usage
        )
        metrics["io_1"] = io_stage
        
        return metrics
    
    def test_detect_slow_stages(self, detector, sample_stage_metrics):
        """Test detection of slow stages"""
        bottlenecks = detector.analyze(sample_stage_metrics, {})
        
        # Should detect slow_stage as bottleneck
        slow_bottlenecks = [b for b in bottlenecks if b['type'] == 'slow_stage']
        assert len(slow_bottlenecks) > 0
        
        slow_bottleneck = slow_bottlenecks[0]
        assert slow_bottleneck['stage'] == 'slow_stage'
        assert slow_bottleneck['severity'] in ['high', 'medium']
        assert 'suggestion' in slow_bottleneck
    
    def test_detect_memory_growth(self, detector, sample_stage_metrics):
        """Test detection of high memory growth"""
        bottlenecks = detector.analyze(sample_stage_metrics, {})
        
        # Should detect memory_stage as having high memory growth
        memory_bottlenecks = [b for b in bottlenecks if b['type'] == 'memory_growth']
        assert len(memory_bottlenecks) > 0
        
        memory_bottleneck = memory_bottlenecks[0]
        assert memory_bottleneck['stage'] == 'memory_stage'
        assert memory_bottleneck['severity'] == 'high'
        assert memory_bottleneck['memory_growth'] == 4000000  # 5M - 1M
    
    def test_detect_io_bottleneck(self, detector, sample_stage_metrics):
        """Test detection of I/O bottlenecks"""
        bottlenecks = detector.analyze(sample_stage_metrics, {})
        
        # Should detect I/O bottleneck
        io_bottlenecks = [b for b in bottlenecks if b['type'] == 'io_bottleneck']
        assert len(io_bottlenecks) > 0
        
        io_bottleneck = io_bottlenecks[0]
        assert 'io_stage' in io_bottleneck['stages']
        assert io_bottleneck['severity'] == 'medium'
    
    def test_detect_high_error_rate(self, detector):
        """Test detection of high error rates"""
        # Create stage with errors
        stage_metrics = {
            "error_stage_1": StageMetrics(
                stage_name="error_prone_stage",
                start_time=time.time() - 5.0,
                end_time=time.time(),
                errors=15  # High error count
            )
        }
        
        # Create error metrics
        error_metrics = deque()
        for i in range(15):
            error_metrics.append(MetricPoint(
                timestamp=time.time() - i,
                value=1,
                metadata={'stage_id': 'error_stage_1'}
            ))
        
        metrics = {'errors': error_metrics}
        
        bottlenecks = detector.analyze(stage_metrics, metrics)
        
        # Should detect high error rate
        error_bottlenecks = [b for b in bottlenecks if b['type'] == 'high_error_rate']
        assert len(error_bottlenecks) > 0
        
        error_bottleneck = error_bottlenecks[0]
        assert error_bottleneck['stage'] == 'error_prone_stage'
        assert error_bottleneck['error_count'] == 15
    
    def test_analyze_empty_metrics(self, detector):
        """Test analysis with empty metrics"""
        bottlenecks = detector.analyze({}, {})
        assert isinstance(bottlenecks, list)
        # May or may not have bottlenecks, but shouldn't crash


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer suggestion functionality"""
    
    @pytest.fixture
    def optimizer(self):
        """Create a PerformanceOptimizer instance for testing"""
        return PerformanceOptimizer()
    
    def test_low_cpu_utilization_suggestion(self, optimizer):
        """Test suggestion for low CPU utilization"""
        system_summary = {'cpu_avg': 30.0, 'memory_growth': 0}
        
        suggestions = optimizer.analyze({}, {}, system_summary)
        
        cpu_suggestions = [s for s in suggestions if s['category'] == 'resource_utilization']
        assert len(cpu_suggestions) > 0
        assert 'worker count' in cpu_suggestions[0]['suggestion']
        assert cpu_suggestions[0]['impact'] == 'high'
    
    def test_high_cpu_utilization_suggestion(self, optimizer):
        """Test suggestion for high CPU utilization"""
        system_summary = {'cpu_max': 95.0, 'memory_growth': 0}
        
        suggestions = optimizer.analyze({}, {}, system_summary)
        
        cpu_suggestions = [s for s in suggestions if s['category'] == 'resource_utilization']
        assert len(cpu_suggestions) > 0
        assert 'CPU bound' in cpu_suggestions[0]['suggestion']
    
    def test_memory_growth_suggestion(self, optimizer):
        """Test suggestion for high memory growth"""
        system_summary = {
            'cpu_avg': 50.0,
            'memory_growth': 2 * 1024 * 1024 * 1024  # 2GB growth
        }
        
        suggestions = optimizer.analyze({}, {}, system_summary)
        
        memory_suggestions = [s for s in suggestions if s['category'] == 'memory_management']
        assert len(memory_suggestions) > 0
        assert 'memory growth' in memory_suggestions[0]['suggestion']
        assert 'streaming' in memory_suggestions[0]['implementation']
    
    def test_low_parsing_throughput_suggestion(self, optimizer):
        """Test suggestion for low parsing throughput"""
        # Create stage with low throughput
        stage_metrics = {
            "parsing_1": StageMetrics(
                stage_name="parsing_stage",
                start_time=time.time() - 10.0,
                end_time=time.time(),
                items_processed=50  # Low throughput: 5 items/sec
            )
        }
        
        suggestions = optimizer.analyze(stage_metrics, {}, {'cpu_avg': 50.0})
        
        parsing_suggestions = [s for s in suggestions 
                              if s['category'] == 'stage_optimization' 
                              and 'parsing' in s.get('stage', '')]
        assert len(parsing_suggestions) > 0
        assert 'parsing throughput' in parsing_suggestions[0]['suggestion']
        assert 'faster parsers' in parsing_suggestions[0]['implementation']
    
    def test_small_batch_size_suggestion(self, optimizer):
        """Test suggestion for small batch sizes"""
        # Create batch size metrics
        batch_metrics = deque()
        for i in range(10):
            batch_metrics.append(MetricPoint(timestamp=time.time() - i, value=5))  # Small batches
        
        metrics = {'batch_size': batch_metrics}
        
        suggestions = optimizer.analyze({}, metrics, {'cpu_avg': 50.0})
        
        batch_suggestions = [s for s in suggestions if s['category'] == 'batching']
        assert len(batch_suggestions) > 0
        assert 'Small batch sizes' in batch_suggestions[0]['suggestion']
        assert 'Increase batch_size' in batch_suggestions[0]['implementation']
    
    def test_low_cache_hit_rate_suggestion(self, optimizer):
        """Test suggestion for low cache hit rate"""
        # Create cache hit/miss metrics
        cache_metrics = deque()
        # 80% misses, 20% hits (low hit rate)
        for i in range(80):
            cache_metrics.append(MetricPoint(timestamp=time.time() - i, value=0))  # Miss
        for i in range(20):
            cache_metrics.append(MetricPoint(timestamp=time.time() - i, value=1))  # Hit
        
        metrics = {'cache_hit': cache_metrics}
        
        suggestions = optimizer.analyze({}, metrics, {'cpu_avg': 50.0})
        
        cache_suggestions = [s for s in suggestions if s['category'] == 'caching']
        assert len(cache_suggestions) > 0
        assert 'cache hit rate' in cache_suggestions[0]['suggestion']
        assert 'cache size' in cache_suggestions[0]['implementation']
    
    def test_analyze_empty_data(self, optimizer):
        """Test analysis with empty data"""
        suggestions = optimizer.analyze({}, {}, {})
        assert isinstance(suggestions, list)
        # Should not crash, may or may not have suggestions


class TestMetricsExporter:
    """Test MetricsExporter output formats"""
    
    @pytest.fixture
    def mock_monitor(self):
        """Create a mock PipelineMonitor for testing exports"""
        monitor = Mock()
        
        # Mock stage metrics
        stage1 = StageMetrics(
            stage_name="parsing",
            start_time=time.time() - 5.0,
            end_time=time.time(),
            items_processed=100,
            bytes_processed=50000,
            errors=2
        )
        stage2 = StageMetrics(
            stage_name="compression",
            start_time=time.time() - 3.0,
            end_time=time.time(),
            items_processed=100,
            bytes_processed=25000,
            errors=0
        )
        
        monitor.stage_metrics = {"stage1": stage1, "stage2": stage2}
        
        # Mock current metrics
        metrics = defaultdict(deque)
        metrics["test_metric"].append(MetricPoint(timestamp=time.time(), value=42.0))
        monitor.metrics = metrics
        
        # Mock system monitor
        system_monitor = Mock()
        system_monitor.get_summary.return_value = {"cpu_avg": 50.0, "memory_avg": 1000000}
        monitor.system_monitor = system_monitor
        
        # Mock analysis methods
        monitor.get_stage_summary.return_value = {"count": 1, "avg_duration": 2.5}
        monitor.get_bottlenecks.return_value = [{"type": "slow_stage", "stage": "parsing"}]
        monitor.get_optimization_suggestions.return_value = [{"category": "batching", "suggestion": "Increase batch size"}]
        
        return monitor
    
    def test_to_prometheus_format(self, mock_monitor):
        """Test Prometheus format export"""
        prometheus_output = MetricsExporter.to_prometheus(mock_monitor)
        
        # Should contain stage metrics
        assert 'pipeline_stage_duration_seconds{stage="parsing"}' in prometheus_output
        assert 'pipeline_stage_items_total{stage="parsing"} 100' in prometheus_output
        assert 'pipeline_stage_bytes_total{stage="parsing"} 50000' in prometheus_output
        assert 'pipeline_stage_errors_total{stage="parsing"} 2' in prometheus_output
        
        # Should contain current metrics
        assert 'pipeline_test_metric 42.0' in prometheus_output
    
    def test_to_json_format(self, mock_monitor):
        """Test JSON format export"""
        json_output = MetricsExporter.to_json(mock_monitor)
        
        # Should be valid JSON
        data = json.loads(json_output)
        
        # Should contain expected sections
        assert 'timestamp' in data
        assert 'stages' in data
        assert 'metrics' in data
        assert 'system' in data
        assert 'bottlenecks' in data
        assert 'optimizations' in data
        
        # Should contain system summary
        assert data['system']['cpu_avg'] == 50.0
        
        # Should contain bottlenecks and optimizations
        assert len(data['bottlenecks']) > 0
        assert len(data['optimizations']) > 0
    
    def test_to_grafana_dashboard(self, mock_monitor):
        """Test Grafana dashboard configuration generation"""
        dashboard = MetricsExporter.to_grafana_dashboard(mock_monitor)
        
        # Should contain dashboard structure
        assert 'dashboard' in dashboard
        assert 'title' in dashboard['dashboard']
        assert 'panels' in dashboard['dashboard']
        
        panels = dashboard['dashboard']['panels']
        assert len(panels) > 0
        
        # Should have expected panel types
        panel_titles = [panel['title'] for panel in panels]
        assert 'Stage Durations' in panel_titles
        assert 'Throughput' in panel_titles
        assert 'Memory Usage' in panel_titles
        assert 'Error Rate' in panel_titles


class TestContextManagers:
    """Test MonitoredPipeline and MonitoredStage context managers"""
    
    @pytest.fixture
    def mock_monitor(self):
        """Create a mock monitor for context manager testing"""
        monitor = Mock()
        monitor.start_monitoring = Mock()
        monitor.stop_monitoring = Mock()
        monitor.stage_start = Mock(return_value="stage_123")
        monitor.stage_end = Mock()
        monitor.record_error = Mock()
        monitor.update_stage_progress = Mock()
        return monitor
    
    def test_monitored_pipeline_context(self, mock_monitor):
        """Test MonitoredPipeline context manager"""
        with patch('builtins.open', mock_open()):
            with patch('pathlib.Path.write_text'):
                with MonitoredPipeline(mock_monitor) as pipeline:
                    assert isinstance(pipeline, MonitoredPipeline)
                    mock_monitor.start_monitoring.assert_called_once()
                
                mock_monitor.stop_monitoring.assert_called_once()
    
    def test_monitored_pipeline_context_with_exception(self, mock_monitor):
        """Test MonitoredPipeline context manager with exception"""
        with patch('builtins.open', mock_open()):
            with patch('pathlib.Path.write_text'):
                try:
                    with MonitoredPipeline(mock_monitor):
                        raise ValueError("Test exception")
                except ValueError:
                    pass
                
                mock_monitor.stop_monitoring.assert_called_once()
    
    def test_monitored_stage_context(self, mock_monitor):
        """Test MonitoredStage context manager"""
        with MonitoredStage(mock_monitor, "test_stage") as stage:
            assert isinstance(stage, MonitoredStage)
            mock_monitor.stage_start.assert_called_once_with("test_stage")
            
            # Test progress update
            stage.update_progress(items=50, bytes_count=25000)
            mock_monitor.update_stage_progress.assert_called_once_with("stage_123", 50, 25000)
        
        mock_monitor.stage_end.assert_called_once_with("stage_123")
    
    def test_monitored_stage_context_with_exception(self, mock_monitor):
        """Test MonitoredStage context manager with exception"""
        test_error = RuntimeError("Test stage error")
        
        try:
            with MonitoredStage(mock_monitor, "error_stage"):
                raise test_error
        except RuntimeError:
            pass
        
        mock_monitor.record_error.assert_called_once_with("stage_123", test_error)
        mock_monitor.stage_end.assert_called_once_with("stage_123")
    
    @pytest.mark.asyncio
    async def test_example_monitored_pipeline(self):
        """Test the example monitored pipeline function"""
        with patch('pipeline_monitoring.PipelineMonitor') as MockMonitor:
            mock_monitor = Mock()
            MockMonitor.return_value = mock_monitor
            
            # Mock the context managers
            mock_monitored_pipeline = Mock()
            mock_monitored_stage = Mock()
            mock_monitored_stage.__enter__ = Mock(return_value=mock_monitored_stage)
            mock_monitored_stage.__exit__ = Mock(return_value=None)
            mock_monitored_stage.update_progress = Mock()
            
            with patch('pipeline_monitoring.MonitoredPipeline') as MockMonitoredPipeline:
                MockMonitoredPipeline.return_value.__enter__ = Mock(return_value=mock_monitored_pipeline)
                MockMonitoredPipeline.return_value.__exit__ = Mock(return_value=None)
                mock_monitored_pipeline.stage = Mock(return_value=mock_monitored_stage)
                
                # Mock the analysis methods
                mock_monitor.get_bottlenecks.return_value = []
                mock_monitor.get_optimization_suggestions.return_value = []
                
                # Import and run the example function
                from pipeline_monitoring import example_monitored_pipeline
                await example_monitored_pipeline()
                
                # Verify monitoring was used
                MockMonitor.assert_called_once()
                MockMonitoredPipeline.assert_called_once()


# Test helper functions
from unittest.mock import mock_open

def create_mock_stage_metrics(stage_name: str, duration: float, items: int = 100) -> StageMetrics:
    """Helper function to create mock stage metrics"""
    start_time = time.time() - duration
    end_time = time.time()
    
    return StageMetrics(
        stage_name=stage_name,
        start_time=start_time,
        end_time=end_time,
        items_processed=items,
        bytes_processed=items * 1000,
        memory_start=1000000,
        memory_peak=1100000,
        cpu_percent=70.0
    )


if __name__ == '__main__':
    pytest.main([__file__])