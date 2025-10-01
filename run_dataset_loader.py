#!/usr/bin/env python3
"""
Runner script for the BBC News Dataset Loader

This script can be executed directly to download and process
the BBC News dataset for the KG-Enhanced RAG system.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.ingestion.dataset_loader import main

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)