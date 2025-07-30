"""
Token Counter Module for LLM Context Optimization
================================================

Provides token counting functionality for measuring compression effectiveness
and managing LLM context window constraints.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import time

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    logger.info("tiktoken library loaded successfully")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available. Install with: pip install tiktoken")

# Fallback: Simple word-based estimation
AVERAGE_TOKENS_PER_WORD = 1.3  # Approximate ratio for most languages


@dataclass
class TokenStats:
    """Statistics for token counting"""
    total_tokens: int
    word_count: int
    char_count: int
    compression_ratio: Optional[float] = None
    processing_time: float = 0.0
    encoding_used: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_tokens": self.total_tokens,
            "word_count": self.word_count, 
            "char_count": self.char_count,
            "compression_ratio": self.compression_ratio,
            "processing_time": self.processing_time,
            "encoding_used": self.encoding_used
        }


class TokenCounter:
    """
    Token counter with support for multiple tokenization methods.
    Prioritizes tiktoken for accuracy, falls back to word-based estimation.
    """
    
    def __init__(self, model_name: str = "gpt-4o", fallback_enabled: bool = True):
        """
        Initialize token counter
        
        Args:
            model_name: OpenAI model name for tiktoken encoding
            fallback_enabled: Whether to use word-based fallback if tiktoken unavailable
        """
        self.model_name = model_name
        self.fallback_enabled = fallback_enabled
        self.encoding = None
        self.encoding_name = "fallback"
        
        # Try to initialize tiktoken
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
                self.encoding_name = f"tiktoken-{model_name}"
                logger.info(f"Initialized tiktoken encoding for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to get tiktoken encoding for {model_name}: {e}")
                if fallback_enabled:
                    logger.info("Falling back to word-based estimation")
                else:
                    raise
    
    def count_tokens(self, text: str) -> TokenStats:
        """
        Count tokens in text using the best available method
        
        Args:
            text: Text to count tokens for
            
        Returns:
            TokenStats object with detailed statistics
        """
        start_time = time.time()
        
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        
        # Token counting
        if self.encoding is not None:
            # Use tiktoken for accurate counting
            try:
                tokens = self.encoding.encode(text)
                total_tokens = len(tokens)
                encoding_used = self.encoding_name
            except Exception as e:
                logger.warning(f"tiktoken encoding failed: {e}")
                if self.fallback_enabled:
                    total_tokens = max(1, int(word_count * AVERAGE_TOKENS_PER_WORD))
                    encoding_used = "word_fallback"
                else:
                    raise
        else:
            # Fallback to word-based estimation
            if self.fallback_enabled:
                total_tokens = max(1, int(word_count * AVERAGE_TOKENS_PER_WORD))
                encoding_used = "word_fallback"
            else:
                raise ValueError("No tokenization method available")
        
        processing_time = time.time() - start_time
        
        return TokenStats(
            total_tokens=total_tokens,
            word_count=word_count,
            char_count=char_count,
            processing_time=processing_time,
            encoding_used=encoding_used
        )
    
    def count_tokens_batch(self, texts: List[str]) -> List[TokenStats]:
        """
        Count tokens for multiple texts efficiently
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of TokenStats for each text
        """
        results = []
        for text in texts:
            stats = self.count_tokens(text)
            results.append(stats)
        return results
    
    def calculate_compression_ratio(self, original_stats: TokenStats, 
                                  compressed_stats: TokenStats) -> float:
        """
        Calculate compression ratio between original and compressed text
        
        Args:
            original_stats: Token stats for original text
            compressed_stats: Token stats for compressed text
            
        Returns:
            Compression ratio (original_tokens / compressed_tokens)
        """
        if compressed_stats.total_tokens == 0:
            return float('inf')
        
        ratio = original_stats.total_tokens / compressed_stats.total_tokens
        
        # Update the compressed stats with the ratio
        compressed_stats.compression_ratio = ratio
        
        return ratio
    
    def format_token_summary(self, stats: TokenStats, title: str = "Token Statistics") -> str:
        """
        Format token statistics as a readable summary
        
        Args:
            stats: Token statistics to format
            title: Title for the summary
            
        Returns:
            Formatted string summary
        """
        lines = [
            f"## {title}",
            "",
            f"- **Total Tokens**: {stats.total_tokens:,}",
            f"- **Word Count**: {stats.word_count:,}",
            f"- **Character Count**: {stats.char_count:,}",
            f"- **Encoding Method**: {stats.encoding_used}",
            f"- **Processing Time**: {stats.processing_time:.3f}s"
        ]
        
        if stats.compression_ratio is not None:
            lines.extend([
                f"- **Compression Ratio**: {stats.compression_ratio:.2f}:1",
                f"- **Space Savings**: {(1 - 1/stats.compression_ratio)*100:.1f}%"
            ])
        
        return "\n".join(lines)
    
    def estimate_context_usage(self, stats: TokenStats, 
                             context_limit: int = 128000) -> Dict[str, Any]:
        """
        Estimate context window usage for LLM processing
        
        Args:
            stats: Token statistics
            context_limit: Maximum context window size
            
        Returns:
            Dictionary with usage statistics
        """
        usage_percent = (stats.total_tokens / context_limit) * 100
        remaining_tokens = max(0, context_limit - stats.total_tokens)
        
        return {
            "total_tokens": stats.total_tokens,
            "context_limit": context_limit,
            "usage_percent": usage_percent,
            "remaining_tokens": remaining_tokens,
            "fits_in_context": stats.total_tokens <= context_limit,
            "estimated_chunks_needed": max(1, (stats.total_tokens + context_limit - 1) // context_limit)
        }


class BatchTokenCounter:
    """Efficient batch processing for large numbers of files"""
    
    def __init__(self, token_counter: TokenCounter):
        self.counter = token_counter
        self.batch_stats: List[TokenStats] = []
        
    def add_text(self, text: str, identifier: str = None) -> TokenStats:
        """Add text to batch and get its stats"""
        stats = self.counter.count_tokens(text)
        self.batch_stats.append(stats)
        return stats
    
    def get_aggregate_stats(self) -> TokenStats:
        """Get aggregate statistics for all processed texts"""
        if not self.batch_stats:
            return TokenStats(0, 0, 0, encoding_used=self.counter.encoding_name)
        
        total_tokens = sum(s.total_tokens for s in self.batch_stats)
        total_words = sum(s.word_count for s in self.batch_stats)
        total_chars = sum(s.char_count for s in self.batch_stats)
        total_time = sum(s.processing_time for s in self.batch_stats)
        
        return TokenStats(
            total_tokens=total_tokens,
            word_count=total_words,
            char_count=total_chars,
            processing_time=total_time,
            encoding_used=self.counter.encoding_name
        )
    
    def clear(self):
        """Clear batch statistics"""
        self.batch_stats.clear()


# Utility functions for easy integration
def count_tokens_simple(text: str, model: str = "gpt-4o") -> int:
    """
    Simple function to count tokens in text
    
    Args:
        text: Text to count
        model: Model name for encoding
        
    Returns:
        Token count
    """
    counter = TokenCounter(model)
    stats = counter.count_tokens(text)
    return stats.total_tokens


def estimate_file_tokens(file_path: Path, model: str = "gpt-4o") -> TokenStats:
    """
    Estimate tokens for a file
    
    Args:
        file_path: Path to text file
        model: Model name for encoding
        
    Returns:
        Token statistics
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        counter = TokenCounter(model)
        return counter.count_tokens(content)
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return TokenStats(0, 0, 0, encoding_used="error")


# Create default instance for convenience
default_counter = None

def get_default_counter() -> TokenCounter:
    """Get or create default token counter instance"""
    global default_counter
    if default_counter is None:
        default_counter = TokenCounter()
    return default_counter