#!/usr/bin/env python3
"""
Complete KG-Enhanced RAG Setup Script

This script runs the complete setup pipeline in the correct order:
1. Environment verification
2. BBC dataset download and processing
3. Document processing and chunking  
4. Vector database population
5. Knowledge graph construction
6. System validation

Run this script for a complete fresh setup.
"""

import sys
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Run a setup script and report results."""
    print(f"\n{'='*60}")
    print(f"📋 Step: {description}")
    print(f"🔧 Running: {script_name}")
    print('='*60)
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        script_path = project_root / "setup" / script_name
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], cwd=project_root, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {str(e)}")
        return False

def main():
    """Run complete setup pipeline."""
    print("🚀 KG-Enhanced RAG Complete Setup Pipeline")
    print("=" * 60)
    print("This script will set up the complete KG-Enhanced RAG system")
    print("from scratch, including all data files and dependencies.\n")
    
    # Define setup steps in correct order
    setup_steps = [
        ("verify_environment_setup.py", "Environment Verification"),
        ("setup_bbc_dataset.py", "BBC News Dataset Download & Processing"), 
        ("setup_vector_database.py", "Document Processing & Chunking"),
        ("setup_vector_store.py", "Vector Database Population"),
        ("setup_knowledge_graph.py", "Knowledge Graph Construction"),
    ]
    
    # Run setup steps
    total_steps = len(setup_steps)
    completed_steps = 0
    
    for script_name, description in setup_steps:
        if run_script(script_name, description):
            completed_steps += 1
        else:
            print(f"\n💥 Setup failed at step: {description}")
            print(f"📊 Progress: {completed_steps}/{total_steps} steps completed")
            print("🔧 Please fix the error and run the setup again.")
            return False
    
    # Run validation tests
    print(f"\n{'='*60}")
    print("🧪 RUNNING VALIDATION TESTS")
    print('='*60)
    
    test_steps = [
        ("test_baseline_rag.py", "Baseline RAG System Test"),
        ("test_kg_enhanced_rag.py", "KG-Enhanced RAG System Test"),
    ]
    
    test_passed = 0
    for script_name, description in test_steps:
        if run_script(script_name, description):
            test_passed += 1
        else:
            print(f"⚠️ Test failed: {description}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 SETUP COMPLETE!")
    print('='*60)
    print(f"📊 Setup Steps: {completed_steps}/{total_steps} completed")
    print(f"🧪 Tests: {test_passed}/{len(test_steps)} passed")
    
    if completed_steps == total_steps:
        print("\n✅ All setup steps completed successfully!")
        print("🚀 Your KG-Enhanced RAG system is ready!")
        print("\n🎯 To start the Streamlit app:")
        print("   streamlit run src/interface/demo_app.py")
        print("\n📁 Data files created in:")
        print("   - data/raw/bbc_news_subset.json (50 articles)")
        print("   - data/processed/ (processed chunks, entities, graph)")
        print("   - data/chroma_db/ (vector database)")
        return True
    else:
        print("\n❌ Setup incomplete. Please fix errors and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)