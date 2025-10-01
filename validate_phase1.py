#!/usr/bin/env python3
"""
Phase 1 Validation Script

Validates all deliverables for Phase 1: Environment Setup & Data Acquisition
"""

import sys
import json
from pathlib import Path
import importlib.util

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def validate_project_structure():
    """Validate that all required directories and files exist."""
    print("🔍 Validating Project Structure...")
    
    required_dirs = [
        "src", "src/ingestion", "src/storage", "src/retrieval", 
        "src/evaluation", "src/interface", "data", "data/raw", 
        "data/processed", "notebooks", "tests"
    ]
    
    required_files = [
        "requirements.txt", ".env.example", ".gitignore", "README.md",
        "src/__init__.py", "src/config.py", "src/ingestion/__init__.py",
        "src/ingestion/dataset_loader.py", "run_dataset_loader.py",
        "notebooks/01_data_exploration.ipynb"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print("❌ Project structure validation failed:")
        if missing_dirs:
            print(f"   Missing directories: {missing_dirs}")
        if missing_files:
            print(f"   Missing files: {missing_files}")
        return False
    
    print("✅ Project structure is valid")
    return True

def validate_configuration():
    """Validate configuration system."""
    print("🔍 Validating Configuration System...")
    
    try:
        from src.config import config
        
        # Test basic configuration loading
        assert config.PROJECT_ROOT.exists(), "Project root path doesn't exist"
        assert config.DATASET_SIZE == 50, "Dataset size should be 50"
        assert len(config.BBC_CATEGORIES) == 5, "Should have 5 BBC categories"
        assert config.CHUNK_SIZE > 0, "Chunk size should be positive"
        
        # Test validation method
        is_valid = config.validate_config()
        assert is_valid, "Configuration validation failed"
        
        print("✅ Configuration system is valid")
        return True
    
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def validate_dataset():
    """Validate that the dataset was downloaded and processed correctly."""
    print("🔍 Validating Dataset...")
    
    try:
        from src.config import config
        dataset_path = config.get_bbc_subset_path()
        
        if not dataset_path.exists():
            print("❌ Dataset file doesn't exist. Run dataset loader first.")
            return False
        
        # Load and validate dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate metadata
        metadata = data['dataset_info']
        assert metadata['total_articles'] == 50, f"Expected 50 articles, got {metadata['total_articles']}"
        assert len(metadata['categories']) == 5, f"Expected 5 categories, got {len(metadata['categories'])}"
        
        # Validate articles
        articles = data['articles']
        assert len(articles) == 50, f"Expected 50 articles, got {len(articles)}"
        
        # Check category distribution
        category_counts = {}
        for article in articles:
            category = article['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category in metadata['categories']:
            assert category_counts.get(category, 0) == 10, f"Category {category} should have 10 articles, got {category_counts.get(category, 0)}"
        
        # Validate article structure
        sample_article = articles[0]
        required_fields = ['article_id', 'category', 'text', 'label', 'label_text', 'original_index']
        for field in required_fields:
            assert field in sample_article, f"Article missing required field: {field}"
        
        print("✅ Dataset is valid")
        return True
    
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        return False

def validate_dependencies():
    """Validate that all required dependencies are available."""
    print("🔍 Validating Dependencies...")
    
    required_packages = [
        'openai', 'chromadb', 'networkx', 'datasets', 'pandas', 
        'numpy', 'matplotlib', 'plotly', 'python-dotenv', 'spacy',
        'streamlit', 'jupyter', 'scikit-learn', 'tqdm', 'click'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.replace('-', '_')  # Handle package name differences
        try:
            if package_name == 'python_dotenv':
                import dotenv
            elif package_name == 'scikit_learn':
                import sklearn
            else:
                importlib.import_module(package_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        return False
    
    print("✅ All dependencies are available")
    return True

def validate_notebook():
    """Validate that the data exploration notebook exists and is properly formatted."""
    print("🔍 Validating Jupyter Notebook...")
    
    notebook_path = project_root / "notebooks" / "01_data_exploration.ipynb"
    
    if not notebook_path.exists():
        print("❌ Data exploration notebook doesn't exist")
        return False
    
    try:
        # Basic validation - check if it's valid JSON
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = f.read()
        
        # Check for key notebook components
        if 'cells' not in notebook_content:
            print("❌ Notebook doesn't contain cells")
            return False
        
        if 'metadata' not in notebook_content:
            print("❌ Notebook missing metadata")
            return False
        
        print("✅ Jupyter notebook is valid")
        return True
    
    except Exception as e:
        print(f"❌ Notebook validation failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 Phase 1 Validation - Environment Setup & Data Acquisition")
    print("=" * 70)
    
    validations = [
        ("Project Structure", validate_project_structure),
        ("Dependencies", validate_dependencies),
        ("Configuration System", validate_configuration),
        ("Dataset", validate_dataset),
        ("Jupyter Notebook", validate_notebook)
    ]
    
    results = []
    for name, validator in validations:
        try:
            result = validator()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation failed with error: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("📊 Validation Summary")
    print("=" * 30)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} validations passed")
    
    if passed == len(results):
        print("\n🎉 Phase 1 implementation is complete and validated!")
        print("🚀 Ready to proceed to Phase 2: Vector Database Implementation")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} validation(s) failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)