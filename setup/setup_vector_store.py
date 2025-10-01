#!/usr/bin/env python3
"""
Runner script for Vector Database Population

This script populates ChromaDB with the processed document chunks,
creating embeddings and enabling vector similarity search for RAG.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the vector database population function
from src.storage.populate_vectordb import main

if __name__ == "__main__":
    # Run in full mode to populate with all documents
    success = main(test_mode=False)
    if success:
        print("🎉 Vector database populated successfully!")
        print("📚 ChromaDB now contains all document embeddings for RAG queries.")
    else:
        print("❌ Vector database population failed!")
    
    sys.exit(0 if success else 1)