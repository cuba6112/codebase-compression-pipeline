"""
Codebase compression pipeline modules.
"""

# Import pipeline stages
from .stages.compression import StreamingCompressor
from .stages.metadata import MetadataStore
from .workers.parallel_processor import ParallelProcessor

__all__ = [
    'StreamingCompressor',
    'MetadataStore',
    'ParallelProcessor',
]