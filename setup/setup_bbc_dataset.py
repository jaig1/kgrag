#!/usr/bin/env python3
"""
Runner script for the BBC News Dataset Loader

This script can be executed directly to download and process
the BBC News dataset for the KG-Enhanced RAG system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.dataset_loader import main

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)