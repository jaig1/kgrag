"""
Test suite for VectorStore functionality.

Tests similarity search, metadata filtering, retrieval by ID, and performance.
"""

import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.storage.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test suite for VectorStore class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        print("🧪 Setting up VectorStore test environment...")
        
        # Initialize configuration
        cls.config = Config()
        
        # Create test vector store
        cls.test_db_path = Path("data/test_chroma_db_unit")
        cls.vector_store = VectorStore(
            db_path=cls.test_db_path,
            collection_name="unit_test_chunks",
            api_key=cls.config.OPENAI_API_KEY
        )
        
        # Reset collection for clean tests
        cls.vector_store.reset_collection()
        
        # Add test documents
        cls.test_documents = [
            "This is a comprehensive business report about quarterly earnings and market performance indicators.",
            "Sports news covering the championship football match with detailed player statistics and game analysis.",
            "Technology article discussing artificial intelligence breakthroughs and machine learning applications in industry.",
            "Entertainment news featuring celebrity interviews and movie premiere coverage from Hollywood events.",
            "Political analysis of recent election results and their impact on government policy decisions.",
            "Business update on stock market trends and investment opportunities in emerging markets.",
            "Sports coverage of tennis tournament with match results and player rankings updates.",
            "Technology review of new smartphone features and consumer electronics innovation trends."
        ]
        
        cls.test_metadatas = [
            {"article_id": "biz_01", "chunk_id": "chunk_biz_01", "category": "business", "position": 1, "title": "Quarterly Report", "is_first_chunk": True, "is_last_chunk": False},
            {"article_id": "sport_01", "chunk_id": "chunk_sport_01", "category": "sport", "position": 1, "title": "Championship Match", "is_first_chunk": True, "is_last_chunk": True},
            {"article_id": "tech_01", "chunk_id": "chunk_tech_01", "category": "tech", "position": 1, "title": "AI Breakthroughs", "is_first_chunk": True, "is_last_chunk": False},
            {"article_id": "ent_01", "chunk_id": "chunk_ent_01", "category": "entertainment", "position": 1, "title": "Celebrity News", "is_first_chunk": True, "is_last_chunk": True},
            {"article_id": "pol_01", "chunk_id": "chunk_pol_01", "category": "politics", "position": 1, "title": "Election Analysis", "is_first_chunk": True, "is_last_chunk": False},
            {"article_id": "biz_02", "chunk_id": "chunk_biz_02", "category": "business", "position": 2, "title": "Market Update", "is_first_chunk": False, "is_last_chunk": True},
            {"article_id": "sport_02", "chunk_id": "chunk_sport_02", "category": "sport", "position": 1, "title": "Tennis Tournament", "is_first_chunk": True, "is_last_chunk": False},
            {"article_id": "tech_02", "chunk_id": "chunk_tech_02", "category": "tech", "position": 1, "title": "Smartphone Review", "is_first_chunk": True, "is_last_chunk": True}
        ]
        
        cls.test_ids = [metadata["chunk_id"] for metadata in cls.test_metadatas]
        
        # Add documents to vector store
        print("📚 Adding test documents to vector store...")
        cls.vector_store.add_documents(
            documents=cls.test_documents,
            metadatas=cls.test_metadatas,
            ids=cls.test_ids,
            batch_size=4
        )
        
        print(f"✅ Test setup complete with {len(cls.test_documents)} documents")
    
    def test_basic_similarity_search(self):
        """Test that similarity search returns relevant results."""
        print("\n🔍 Testing basic similarity search...")
        
        query = "business financial report"
        results = self.vector_store.similarity_search(query, top_k=3)
        
        # Verify structure
        self.assertIsInstance(results, dict)
        self.assertIn('documents', results)
        self.assertIn('metadatas', results)
        self.assertIn('distances', results)
        self.assertEqual(results['query'], query)
        
        # Verify results
        self.assertGreater(len(results['documents']), 0)
        self.assertLessEqual(len(results['documents']), 3)
        
        # Check that business documents are ranked highly
        business_in_top_2 = any(
            metadata['category'] == 'business' 
            for metadata in results['metadatas'][:2]
        )
        self.assertTrue(business_in_top_2, "Business documents should rank highly for business query")
        
        print(f"   ✅ Search returned {len(results['documents'])} relevant results")
    
    def test_metadata_filtering_by_category(self):
        """Test metadata filtering by category."""
        print("\n🏷️  Testing metadata filtering by category...")
        
        query = "latest news update"
        category_filter = {"category": "tech"}
        
        results = self.vector_store.similarity_search(
            query, 
            top_k=5, 
            filter_metadata=category_filter
        )
        
        # Verify all results match filter
        self.assertGreater(len(results['documents']), 0)
        
        for metadata in results['metadatas']:
            self.assertEqual(metadata['category'], 'tech', 
                           f"All results should be tech category, got {metadata['category']}")
        
        print(f"   ✅ All {len(results['documents'])} results match category filter")
    
    def test_retrieval_by_chunk_id(self):
        """Test retrieving specific documents by chunk ID."""
        print("\n🆔 Testing retrieval by chunk ID...")
        
        test_chunk_id = "chunk_tech_01"
        result = self.vector_store.get_by_id(test_chunk_id)
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('id', result)
        self.assertIn('document', result)
        self.assertIn('metadata', result)
        
        # Verify correct document retrieved
        self.assertEqual(result['id'], test_chunk_id)
        self.assertEqual(result['metadata']['chunk_id'], test_chunk_id)
        self.assertEqual(result['metadata']['category'], 'tech')
        
        # Test non-existent ID
        non_existent_result = self.vector_store.get_by_id("non_existent_chunk")
        self.assertIsNone(non_existent_result)
        
        print(f"   ✅ Successfully retrieved document by ID")
    
    def test_top_k_parameter(self):
        """Test that top_k parameter works correctly."""
        print("\n🔢 Testing top_k parameter...")
        
        query = "news article content"
        
        # Test different k values
        for k in [1, 3, 5]:
            results = self.vector_store.similarity_search(query, top_k=k)
            
            actual_count = len(results['documents'])
            expected_count = min(k, len(self.test_documents))
            
            self.assertEqual(actual_count, expected_count,
                           f"top_k={k} should return {expected_count} results, got {actual_count}")
        
        print(f"   ✅ top_k parameter works correctly")
    
    def test_compound_metadata_filters(self):
        """Test multiple metadata filters (category AND position)."""
        print("\n🔗 Testing compound metadata filters...")
        
        query = "information content"
        compound_filter = {"$and": [{"category": {"$eq": "business"}}, {"position": {"$eq": 1}}]}
        
        results = self.vector_store.similarity_search(
            query,
            top_k=10,
            filter_metadata=compound_filter
        )
        
        # Verify all results match both filters
        self.assertGreater(len(results['documents']), 0)
        
        for metadata in results['metadatas']:
            self.assertEqual(metadata['category'], 'business')
            self.assertEqual(metadata['position'], 1)
        
        print(f"   ✅ All {len(results['documents'])} results match compound filter")
    
    def test_query_latency(self):
        """Test that query latency is under 1 second."""
        print("\n⏱️  Testing query latency...")
        
        query = "test search performance"
        latencies = []
        
        # Run multiple searches to get average latency
        for i in range(5):
            start_time = time.time()
            results = self.vector_store.similarity_search(query, top_k=3)
            latency = time.time() - start_time
            latencies.append(latency)
            
            self.assertLess(latency, 1.0, f"Search {i+1} took {latency:.3f}s, should be under 1.0s")
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        self.assertLess(avg_latency, 1.0, f"Average latency {avg_latency:.3f}s should be under 1.0s")
        
        print(f"   ✅ Average latency: {avg_latency:.3f}s, Max: {max_latency:.3f}s")
    
    def test_search_by_metadata_only(self):
        """Test metadata-only search functionality."""
        print("\n📋 Testing metadata-only search...")
        
        # Search for all sport documents
        sport_docs = self.vector_store.search_by_metadata(
            metadata_filter={"category": "sport"},
            limit=10
        )
        
        # Verify results
        self.assertGreater(len(sport_docs), 0)
        
        for doc in sport_docs:
            self.assertIn('id', doc)
            self.assertIn('document', doc)
            self.assertIn('metadata', doc)
            self.assertEqual(doc['metadata']['category'], 'sport')
        
        print(f"   ✅ Found {len(sport_docs)} documents via metadata search")
    
    def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions for text-embedding-3-small."""
        print("\n📏 Testing embedding dimensions...")
        
        # Generate a test embedding
        test_text = "Sample text for dimension testing"
        embedding = self.vector_store.embedder.embed_text(test_text)
        
        # text-embedding-3-small should produce 1536-dimensional vectors
        expected_dimensions = 1536
        actual_dimensions = len(embedding)
        
        self.assertEqual(actual_dimensions, expected_dimensions,
                        f"Expected {expected_dimensions} dimensions, got {actual_dimensions}")
        
        # Verify embedding values are floats
        self.assertTrue(all(isinstance(val, float) for val in embedding),
                       "All embedding values should be floats")
        
        print(f"   ✅ Embeddings have correct dimensions: {actual_dimensions}")
    
    def test_collection_statistics(self):
        """Test collection statistics functionality."""
        print("\n📊 Testing collection statistics...")
        
        stats = self.vector_store.get_collection_stats()
        
        # Verify statistics structure
        self.assertIn('total_documents', stats)
        self.assertIn('collection_name', stats)
        self.assertIn('metadata_fields', stats)
        self.assertIn('categories_found', stats)
        
        # Verify document count
        self.assertEqual(stats['total_documents'], len(self.test_documents))
        
        # Verify categories
        expected_categories = {'business', 'sport', 'tech', 'entertainment', 'politics'}
        found_categories = set(stats['categories_found'])
        self.assertTrue(expected_categories.issubset(found_categories),
                       f"Expected categories {expected_categories}, found {found_categories}")
        
        print(f"   ✅ Collection statistics are accurate")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        print("\n🧹 Cleaning up test environment...")
        
        # Reset the test collection
        cls.vector_store.reset_collection()
        
        print("   ✅ Test cleanup complete")


def run_vector_store_tests():
    """Run all vector store tests."""
    print("🚀 VECTOR STORE TEST SUITE")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVectorStore)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"📊 Total tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"🎯 Success rate: {passed/total_tests*100:.1f}%")
    
    if failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    success = failures == 0 and errors == 0
    print(f"\n{'✅ ALL TESTS PASSED!' if success else '❌ SOME TESTS FAILED!'}")
    
    return success


if __name__ == "__main__":
    success = run_vector_store_tests()
    sys.exit(0 if success else 1)