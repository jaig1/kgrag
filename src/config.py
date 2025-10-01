"""
Configuration management for the KG-Enhanced RAG System.

This module handles loading environment variables and provides
a centralized configuration class for all system parameters.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration class that manages all system parameters.
    
    Loads configuration from environment variables with sensible defaults.
    """
    
    def __init__(self):
        # Project root directory
        self.PROJECT_ROOT = Path(__file__).parent.parent
        
        # API Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Model Configuration
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-turbo-preview")
        
        # Processing Parameters
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
        self.TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
        
        # Dataset Configuration
        self.DATASET_SIZE = int(os.getenv("DATASET_SIZE", "50"))
        self.ARTICLES_PER_CATEGORY = int(os.getenv("ARTICLES_PER_CATEGORY", "10"))
        
        # File Paths (relative to project root)
        self.RAW_DATA_PATH = self.PROJECT_ROOT / os.getenv("RAW_DATA_PATH", "data/raw")
        self.PROCESSED_DATA_PATH = self.PROJECT_ROOT / os.getenv("PROCESSED_DATA_PATH", "data/processed")
        self.VECTOR_DB_PATH = self.PROJECT_ROOT / os.getenv("VECTOR_DB_PATH", "data/processed/chroma_db")
        self.KNOWLEDGE_GRAPH_PATH = self.PROJECT_ROOT / os.getenv("KNOWLEDGE_GRAPH_PATH", "data/processed/knowledge_graph.json")
        
        # System Configuration
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
        self.RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
        
        # BBC News Dataset Configuration
        self.BBC_DATASET_NAME = "SetFit/bbc-news"
        self.BBC_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.RAW_DATA_PATH,
            self.PROCESSED_DATA_PATH,
            self.VECTOR_DB_PATH.parent  # ChromaDB will create its own directory
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_bbc_subset_path(self) -> Path:
        """Get the path for the BBC news subset JSON file."""
        return self.RAW_DATA_PATH / "bbc_news_subset.json"
    
    def validate_config(self) -> bool:
        """
        Validate the configuration settings.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        validations = [
            (self.OPENAI_API_KEY is not None, "OpenAI API key is required"),
            (self.CHUNK_SIZE > 0, "Chunk size must be positive"),
            (self.CHUNK_OVERLAP >= 0, "Chunk overlap must be non-negative"),
            (self.CHUNK_OVERLAP < self.CHUNK_SIZE, "Chunk overlap must be less than chunk size"),
            (self.TOP_K_RESULTS > 0, "Top K results must be positive"),
            (self.DATASET_SIZE > 0, "Dataset size must be positive"),
            (self.ARTICLES_PER_CATEGORY > 0, "Articles per category must be positive"),
            (len(self.BBC_CATEGORIES) * self.ARTICLES_PER_CATEGORY == self.DATASET_SIZE, 
             "Dataset size should equal categories * articles per category")
        ]
        
        for is_valid, message in validations:
            if not is_valid:
                print(f"Configuration Error: {message}")
                return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"""Config(
    PROJECT_ROOT={self.PROJECT_ROOT},
    EMBEDDING_MODEL={self.EMBEDDING_MODEL},
    CHAT_MODEL={self.CHAT_MODEL},
    CHUNK_SIZE={self.CHUNK_SIZE},
    CHUNK_OVERLAP={self.CHUNK_OVERLAP},
    TOP_K_RESULTS={self.TOP_K_RESULTS},
    DATASET_SIZE={self.DATASET_SIZE}
)"""

# Global configuration instance
config = Config()