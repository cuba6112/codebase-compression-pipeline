"""
Pipeline stages for the codebase compression system.
"""

from .compression import StreamingCompressor
from .metadata import MetadataStore

__all__ = [
    'StreamingCompressor',
    'MetadataStore',
]