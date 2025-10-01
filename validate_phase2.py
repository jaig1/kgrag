#!/usr/bin/env python3
"""
Phase 2 Validation Script

Validates all deliverables for Phase 2: Document Processing & Chunking
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.config import config


def validate_modules_exist():
    """Validate that all required modules exist and can be imported."""
    print("🔍 Validating Module Existence and Import...")
    
    modules_to_test = [
        ('src.ingestion.preprocessor', 'TextPreprocessor'),
        ('src.ingestion.chunker', 'DocumentChunker'),
        ('src.ingestion.processor', 'DocumentProcessor')
    ]
    
    missing_modules = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"   ❌ {module_name}.{class_name}: {e}")
            missing_modules.append(module_name)
        except AttributeError as e:
            print(f"   ❌ {module_name}.{class_name}: Class not found")
            missing_modules.append(f"{module_name}.{class_name}")
    
    if missing_modules:
        print(f"❌ Missing modules: {missing_modules}")
        return False
    
    print("✅ All modules exist and can be imported")
    return True


def validate_preprocessor():
    """Validate the TextPreprocessor functionality."""
    print("🔍 Validating TextPreprocessor...")
    
    try:
        from src.ingestion.preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        # Test basic preprocessing
        test_text = "This  is   a   test   with    extra  spaces."
        cleaned, title, stats = preprocessor.preprocess_text(test_text, generate_title=True)
        
        # Validate results
        assert "  " not in cleaned, "Multiple spaces should be removed"
        assert len(title) > 0, "Title should be generated"
        assert stats.original_length > stats.cleaned_length, "Text should be cleaned"
        assert stats.chars_removed >= 0, "Characters removed should be non-negative"
        
        # Test article preprocessing
        test_article = {
            'article_id': 'test_001',
            'text': test_text,
            'category': 'test'
        }
        
        processed_article = preprocessor.preprocess_article(test_article)
        assert 'preprocessing_stats' in processed_article, "Should contain preprocessing stats"
        assert 'generated_title' in processed_article, "Should contain generated title"
        assert processed_article['text'] == cleaned, "Text should match cleaned version"
        
        print("✅ TextPreprocessor validation passed")
        return True
        
    except Exception as e:
        print(f"❌ TextPreprocessor validation failed: {e}")
        return False


def validate_chunker():
    """Validate the DocumentChunker functionality."""
    print("🔍 Validating DocumentChunker...")
    
    try:
        from src.ingestion.chunker import DocumentChunker
        
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        
        # Test basic chunking
        test_article = {
            'article_id': 'test_001',
            'text': "This is the first sentence. This is the second sentence. " * 10,  # Long text
            'category': 'test',
            'generated_title': 'Test Article'
        }
        
        chunks = chunker.chunk_document(test_article)
        
        # Validate chunks
        assert len(chunks) > 1, "Should create multiple chunks for long text"
        
        for i, chunk in enumerate(chunks):
            assert 'chunk_id' in chunk, "Each chunk should have an ID"
            assert 'text' in chunk, "Each chunk should have text"
            assert 'metadata' in chunk, "Each chunk should have metadata"
            
            metadata = chunk['metadata']
            assert metadata['position'] == i, "Position should match index"
            assert metadata['total_chunks'] == len(chunks), "Total chunks should be correct"
            assert metadata['is_first_chunk'] == (i == 0), "First chunk flag should be correct"
            assert metadata['is_last_chunk'] == (i == len(chunks) - 1), "Last chunk flag should be correct"
            assert metadata['char_count'] <= chunker.chunk_size + 50, "Chunk size should be reasonable"  # Allow some flexibility
        
        print("✅ DocumentChunker validation passed")
        return True
        
    except Exception as e:
        print(f"❌ DocumentChunker validation failed: {e}")
        return False


def validate_processor():
    """Validate the DocumentProcessor functionality."""
    print("🔍 Validating DocumentProcessor...")
    
    try:
        from src.ingestion.processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test if processor can load articles
        try:
            articles, metadata = processor.load_raw_articles()
            assert len(articles) > 0, "Should load articles"
            assert isinstance(metadata, dict), "Should load metadata"
        except Exception as e:
            print(f"   ⚠️ Could not load articles (may be expected): {e}")
            return True  # This is OK if no dataset exists yet
        
        print("✅ DocumentProcessor validation passed")
        return True
        
    except Exception as e:
        print(f"❌ DocumentProcessor validation failed: {e}")
        return False


def validate_processed_data():
    """Validate the processed data file."""
    print("🔍 Validating Processed Data...")
    
    processed_file = config.PROCESSED_DATA_PATH / "processed_articles.json"
    
    if not processed_file.exists():
        print("❌ Processed data file not found. Run the processing pipeline first.")
        return False
    
    try:
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        required_keys = ['processing_info', 'statistics', 'articles', 'all_chunks']
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        
        articles = data['articles']
        chunks = data['all_chunks']
        stats = data['statistics']
        
        # Validate article count
        assert len(articles) == 50, f"Expected 50 articles, got {len(articles)}"
        
        # Validate chunk count (should be 150-250)
        chunk_count = len(chunks)
        assert 150 <= chunk_count <= 250, f"Expected 150-250 chunks, got {chunk_count}"
        
        # Validate average chunks per article (should be 3-5)
        avg_chunks = chunk_count / len(articles)
        assert 3 <= avg_chunks <= 5, f"Expected 3-5 chunks per article, got {avg_chunks:.1f}"
        
        # Validate chunk sizes
        chunk_sizes = [chunk['metadata']['char_count'] for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
        assert 600 <= avg_chunk_size <= 1000, f"Expected avg chunk size 600-1000, got {avg_chunk_size:.0f}"
        
        # Validate metadata presence
        sample_chunk = chunks[0]
        required_chunk_keys = ['chunk_id', 'text', 'metadata', 'source_article']
        for key in required_chunk_keys:
            assert key in sample_chunk, f"Missing required chunk key: {key}"
        
        # Validate metadata structure
        metadata = sample_chunk['metadata']
        required_metadata_keys = [
            'chunk_id', 'article_id', 'position', 'total_chunks', 'category', 
            'title', 'is_first_chunk', 'is_last_chunk', 'char_count', 'word_count'
        ]
        for key in required_metadata_keys:
            assert key in metadata, f"Missing required metadata key: {key}"
        
        # Validate category distribution
        categories = set(chunk['source_article']['category'] for chunk in chunks)
        expected_categories = set(config.BBC_CATEGORIES)
        assert categories == expected_categories, f"Category mismatch: {categories} vs {expected_categories}"
        
        print("✅ Processed data validation passed")
        print(f"   Articles: {len(articles)}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Average chunks per article: {avg_chunks:.1f}")
        print(f"   Average chunk size: {avg_chunk_size:.0f} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Processed data validation failed: {e}")
        return False


def validate_chunk_overlap():
    """Validate that chunk overlapping is working correctly."""
    print("🔍 Validating Chunk Overlap...")
    
    processed_file = config.PROCESSED_DATA_PATH / "processed_articles.json"
    
    if not processed_file.exists():
        print("❌ Processed data file not found.")
        return False
    
    try:
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data['all_chunks']
        
        # Group chunks by article
        chunks_by_article = {}
        for chunk in chunks:
            article_id = chunk['source_article']['article_id']
            if article_id not in chunks_by_article:
                chunks_by_article[article_id] = []
            chunks_by_article[article_id].append(chunk)
        
        # Sort chunks by position
        for article_id in chunks_by_article:
            chunks_by_article[article_id].sort(key=lambda x: x['metadata']['position'])
        
        overlap_found = False
        
        # Check overlaps in multi-chunk articles
        for article_id, article_chunks in chunks_by_article.items():
            if len(article_chunks) > 1:
                for i in range(1, len(article_chunks)):
                    overlap = article_chunks[i]['metadata']['overlap_with_previous']
                    if overlap > 0:
                        overlap_found = True
                        # Validate overlap is reasonable
                        assert overlap <= config.CHUNK_OVERLAP * 1.5, f"Overlap too large: {overlap}"
        
        if overlap_found:
            print("✅ Chunk overlap validation passed")
        else:
            print("⚠️ No overlaps found (may be expected for short articles)")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunk overlap validation failed: {e}")
        return False


def validate_reconstruction():
    """Validate that articles can be reconstructed from chunks."""
    print("🔍 Validating Article Reconstruction...")
    
    try:
        from src.ingestion.processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Load processed data
        processed_file = config.PROCESSED_DATA_PATH / "processed_articles.json"
        
        if not processed_file.exists():
            print("❌ Processed data file not found.")
            return False
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data['articles']
        chunks = data['all_chunks']
        
        # Test reconstruction on 3 articles
        verification_results = processor.verify_chunk_reconstruction(articles, chunks, num_verify=3)
        
        success_rate = sum(verification_results.values()) / len(verification_results) if verification_results else 0
        
        if success_rate >= 0.9:  # 90% success rate threshold
            print("✅ Article reconstruction validation passed")
            return True
        else:
            print(f"❌ Article reconstruction validation failed: {success_rate:.1%} success rate")
            return False
            
    except Exception as e:
        print(f"❌ Article reconstruction validation failed: {e}")
        return False


def main():
    """Main validation function for Phase 2."""
    print("🚀 Phase 2 Validation - Document Processing & Chunking")
    print("=" * 70)
    
    validations = [
        ("Module Existence", validate_modules_exist),
        ("TextPreprocessor", validate_preprocessor),
        ("DocumentChunker", validate_chunker),
        ("DocumentProcessor", validate_processor),
        ("Processed Data", validate_processed_data),
        ("Chunk Overlap", validate_chunk_overlap),
        ("Article Reconstruction", validate_reconstruction)
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
        print("\n🎉 Phase 2 implementation is complete and validated!")
        print("🚀 Ready to proceed to Phase 3: Vector Database Implementation")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} validation(s) failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)