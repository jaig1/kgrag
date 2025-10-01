#!/usr/bin/env python3
"""
Runner script for the Document Processing Pipeline

This script executes the complete document processing pipeline
including preprocessing, chunking, and saving processed results.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.processor import main

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)