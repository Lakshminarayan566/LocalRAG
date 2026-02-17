"""
Configuration management for PrivaRepo
Handles all system settings, model parameters, and chunking strategies
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Dict, Optional
from pathlib import Path


class TreeSitterConfig(BaseSettings):
    """Configuration for Tree-sitter code chunking"""
    
    # Language support
    supported_languages: List[str] = ["python", "javascript", "typescript", "java", "cpp"]
    
    # Chunking strategy
    chunk_by_functions: bool = True
    chunk_by_classes: bool = True
    chunk_by_methods: bool = True
    include_docstrings: bool = True
    include_comments: bool = True
    
    # Size constraints
    min_chunk_size: int = 50  # minimum characters
    max_chunk_size: int = 2000  # maximum characters
    overlap_size: int = 200  # overlap between chunks
    
    # Context inclusion
    include_imports: bool = True
    include_class_context: bool = True  # Include class definition with methods


class ChromaDBConfig(BaseSettings):
    """Configuration for ChromaDB vector store"""
    
    persist_directory: str = "./chroma_db"
    collection_name: str = "code_intelligence"
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    distance_metric: str = "cosine"  # cosine, l2, or ip
    
    # Retrieval settings
    top_k: int = 8
    similarity_threshold: float = 0.1


class OllamaConfig(BaseSettings):
    """Configuration for Ollama LLM"""
    
    # Model settings
    model_name: str = "llama3.2:3b"  # or "deepseek-coder", "mistral", etc.
    base_url: str = "http://localhost:11434"
    
    # Quantization (handled by Ollama)
    quantization: str = "4-bit"  # 4-bit, 8-bit, or none
    
    # Generation parameters
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 512
    
    # Performance optimization
    num_ctx: int = 4096  # context window
    num_gpu: int = 1
    num_thread: int = 8
    
    # Timeout settings
    request_timeout: int = 30  # seconds


class RAGASConfig(BaseSettings):
    """Configuration for RAGAS evaluation"""
    
    # Metrics to compute
    compute_faithfulness: bool = True
    compute_answer_relevancy: bool = True
    compute_context_precision: bool = True
    compute_context_recall: bool = True
    
    # Target benchmarks (from your project description)
    target_faithfulness: float = 0.92
    target_precision_improvement: float = 0.40  # 40% improvement


class SystemConfig(BaseSettings):
    """Global system configuration"""
    
    # Directories
    code_base_path: Optional[Path] = None
    cache_dir: str = "./.cache"
    log_dir: str = "./logs"
    
    # Performance
    max_workers: int = 4  # for parallel processing
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Logging
    log_level: str = "INFO"
    
    # Sub-configurations
    tree_sitter: TreeSitterConfig = Field(default_factory=TreeSitterConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    ragas: RAGASConfig = Field(default_factory=RAGASConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = SystemConfig()


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    Load configuration from file or environment
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        SystemConfig instance
    """
    if config_path:
        # Load from custom config file
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return SystemConfig(**config_dict)
    
    return SystemConfig()


def save_config(config: SystemConfig, output_path: str):
    """
    Save configuration to file
    
    Args:
        config: SystemConfig instance
        output_path: Path to save config
    """
    import json
    with open(output_path, 'w') as f:
        json.dump(config.model_dump(), f, indent=2, default=str)
