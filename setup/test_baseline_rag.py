#!/usr/bin/env python3
"""
Baseline RAG System Test Script

This script validates the baseline RAG system functionality
by running test queries and verifying responses.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.baseline_rag import test_baseline_rag

if __name__ == "__main__":
    print("🧪 TESTING BASELINE RAG SYSTEM")
    print("=" * 50)
    print("This test validates the baseline vector-based RAG system")
    print("by running sample queries and checking responses.\n")
    
    try:
        test_baseline_rag()
        print("\n✅ Baseline RAG system test completed successfully!")
        print("🎉 System is ready for production use.")
    except Exception as e:
        print(f"\n❌ Baseline RAG test failed: {str(e)}")
        print("🔧 Please check the vector database and embeddings setup.")
        sys.exit(1)
    
    sys.exit(0)