"""
Vector Database Population Script

Loads processed article chunks and populates ChromaDB with embeddings and metadata.
Supports both test mode (small subset) and full mode (all chunks).
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.storage.vector_store import VectorStore
from src.storage.embedder import Embedder


def load_processed_chunks(config: Config, test_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Load processed article chunks from Phase 2.
    
    Args:
        config: Configuration object
        test_mode: If True, return only a small subset of chunks
        
    Returns:
        List of chunk dictionaries
    """
    processed_file = config.PROCESSED_DATA_PATH / 'processed_articles.json'
    
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed articles file not found: {processed_file}")
    
    print(f"📂 Loading processed articles from {processed_file}")
    
    with open(processed_file, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    print(f"   Loaded {len(articles)} articles")
    
    # Check if chunks are stored at top level or in articles
    if 'all_chunks' in data:
        print(f"   Using chunks from 'all_chunks' field")
        raw_chunks = data['all_chunks']
        all_chunks = []
        
        # Process each chunk to standardize format
        for chunk in raw_chunks:
            metadata = chunk.get('metadata', {})
            source_article = chunk.get('source_article', {})
            
            chunk_data = {
                'chunk_id': chunk.get('chunk_id', ''),
                'text': chunk.get('text', ''),
                'position': metadata.get('position', 0),
                'is_first_chunk': metadata.get('is_first_chunk', False),
                'is_last_chunk': metadata.get('is_last_chunk', False),
                'article_id': metadata.get('article_id', source_article.get('article_id', 'unknown')),
                'title': metadata.get('title', source_article.get('title', 'Unknown Title')),
                'category': metadata.get('category', source_article.get('category', 'unknown')),
                'url': '',  # Not available in this format
                'published_date': ''  # Not available in this format
            }
            all_chunks.append(chunk_data)
                
    else:
        # Extract chunks from individual articles (legacy format)
        print(f"   Extracting chunks from individual articles")
        all_chunks = []
        
        for article in articles:
            article_chunks = article.get('chunks', [])
            if len(article_chunks) == 0:
                print(f"   ⚠️  Article {article.get('article_id', 'unknown')} has no chunks")
            
            # Add article-level metadata to each chunk
            for chunk in article_chunks:
                chunk_data = {
                    'chunk_id': chunk.get('chunk_id', ''),
                    'text': chunk.get('text', ''),
                    'position': chunk.get('position', 0),
                    'is_first_chunk': chunk.get('is_first_chunk', False),
                    'is_last_chunk': chunk.get('is_last_chunk', False),
                    'article_id': article.get('article_id', ''),
                    'title': article.get('generated_title', ''),
                    'category': article.get('category', ''),
                    'url': article.get('url', ''),
                    'published_date': article.get('published_date', '')
                }
                all_chunks.append(chunk_data)
    
    total_chunks = len(all_chunks)
    print(f"   Extracted {total_chunks:,} total chunks from all articles")
    
    # Return subset for testing
    if test_mode:
        test_chunks = all_chunks[:10]  # Use first 10 chunks for testing
        print(f"🧪 Test mode: Using {len(test_chunks)} chunks for testing")
        return test_chunks
    
    return all_chunks


def prepare_vectordb_data(chunks: List[Dict[str, Any]]) -> tuple:
    """
    Prepare data for vector database insertion.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    print(f"🔄 Preparing {len(chunks):,} chunks for vector database...")
    
    documents = []
    metadatas = []
    ids = []
    
    for chunk in chunks:
        # Document text
        documents.append(chunk['text'])
        
        # Metadata (ensure all values are JSON serializable)
        metadata = {
            'article_id': str(chunk['article_id']),
            'chunk_id': str(chunk['chunk_id']),
            'category': str(chunk['category']),
            'position': int(chunk['position']),
            'title': str(chunk['title']),
            'is_first_chunk': bool(chunk['is_first_chunk']),
            'is_last_chunk': bool(chunk['is_last_chunk']),
            'url': str(chunk['url']),
            'published_date': str(chunk['published_date'])
        }
        metadatas.append(metadata)
        
        # Unique ID
        ids.append(chunk['chunk_id'])
    
    print(f"✅ Prepared {len(documents):,} documents with metadata and IDs")
    
    return documents, metadatas, ids


def test_search_functionality(vector_store: VectorStore, test_mode: bool = False) -> Dict[str, Any]:
    """
    Test basic search functionality with sample queries.
    
    Args:
        vector_store: VectorStore instance
        test_mode: If True, use simple test queries
        
    Returns:
        Dictionary with test results
    """
    print(f"\n🔍 Testing search functionality...")
    
    if test_mode:
        test_queries = [
            "business news and market updates",
            "sports and football results", 
            "technology and artificial intelligence"
        ]
    else:
        test_queries = [
            "business earnings and financial performance",
            "sports championship and tournament results",
            "technology innovation and digital transformation",
            "entertainment industry and celebrity news",
            "political developments and government policy"
        ]
    
    search_results = {}
    
    for i, query in enumerate(test_queries):
        print(f"\n   Query {i+1}: '{query}'")
        
        start_time = time.time()
        results = vector_store.similarity_search(query, top_k=3)
        search_time = time.time() - start_time
        
        print(f"   ⏱️  Search time: {search_time:.3f}s")
        print(f"   📊 Found {len(results['documents'])} results:")
        
        query_results = []
        for j, (doc, metadata, distance) in enumerate(zip(
            results['documents'], 
            results['metadatas'], 
            results['distances']
        )):
            result_info = {
                'rank': j + 1,
                'category': metadata['category'],
                'article_id': metadata['article_id'],
                'distance': distance,
                'text_preview': doc[:100] + '...' if len(doc) > 100 else doc
            }
            query_results.append(result_info)
            
            print(f"      {j+1}. [{metadata['category']}] {metadata['article_id']} (dist: {distance:.3f})")
            print(f"         {doc[:80]}...")
        
        search_results[query] = {
            'search_time': search_time,
            'results': query_results
        }
    
    return search_results


def verify_metadata_filtering(vector_store: VectorStore) -> Dict[str, Any]:
    """
    Verify that metadata filtering works correctly.
    
    Args:
        vector_store: VectorStore instance
        
    Returns:
        Dictionary with filtering test results
    """
    print(f"\n🏷️  Testing metadata filtering...")
    
    # Test different category filters
    categories_to_test = ['business', 'sport', 'tech', 'entertainment', 'politics']
    filtering_results = {}
    
    for category in categories_to_test:
        print(f"\n   Testing category filter: '{category}'")
        
        # Search with category filter
        results = vector_store.similarity_search(
            query="latest news update",
            top_k=5,
            filter_metadata={"category": category}
        )
        
        # Verify all results match the filter
        matching_results = [
            r for r in results['metadatas'] 
            if r.get('category') == category
        ]
        
        filter_success = len(matching_results) == len(results['metadatas'])
        
        print(f"      Found {len(results['documents'])} results")
        print(f"      Filter accuracy: {len(matching_results)}/{len(results['metadatas'])} ({'✅' if filter_success else '❌'})")
        
        filtering_results[category] = {
            'total_results': len(results['documents']),
            'matching_filter': len(matching_results),
            'filter_success': filter_success
        }
    
    # Test compound filters (ChromaDB format)
    print(f"\n   Testing compound filters...")
    compound_results = vector_store.similarity_search(
        query="news",
        top_k=3,
        filter_metadata={"$and": [{"category": {"$eq": "business"}}, {"position": {"$eq": 0}}]}  # First chunk of business articles
    )
    
    compound_matches = [
        r for r in compound_results['metadatas']
        if r.get('category') == 'business' and r.get('position') == 0
    ]
    
    compound_success = len(compound_matches) == len(compound_results['metadatas'])
    print(f"      Compound filter (category='business' AND position=0): {len(compound_results['documents'])} results ({'✅' if compound_success else '❌'})")
    
    filtering_results['compound_filter'] = {
        'total_results': len(compound_results['documents']),
        'matching_filter': len(compound_matches),
        'filter_success': compound_success
    }
    
    return filtering_results


def save_embedding_stats(config: Config, 
                        embedder: Embedder, 
                        vector_store: VectorStore,
                        search_results: Dict[str, Any],
                        filtering_results: Dict[str, Any],
                        total_chunks: int,
                        processing_time: float) -> None:
    """
    Save embedding and vector store statistics.
    
    Args:
        config: Configuration object
        embedder: Embedder instance
        vector_store: VectorStore instance
        search_results: Search test results
        filtering_results: Filtering test results
        total_chunks: Number of chunks processed
        processing_time: Total processing time
    """
    print(f"\n💾 Saving embedding statistics...")
    
    # Collect all statistics
    embedding_stats = embedder.get_stats()
    collection_stats = vector_store.get_collection_stats()
    
    # Calculate search performance metrics
    search_times = [result['search_time'] for result in search_results.values()]
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    max_search_time = max(search_times) if search_times else 0
    
    # Calculate filtering accuracy
    filter_accuracies = [
        result['filter_success'] for result in filtering_results.values()
    ]
    filter_success_rate = (sum(filter_accuracies) / len(filter_accuracies) * 100) if filter_accuracies else 0
    
    stats = {
        'processing_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_chunks_processed': total_chunks,
            'total_processing_time_seconds': processing_time,
            'chunks_per_second': total_chunks / processing_time if processing_time > 0 else 0
        },
        'embedding_stats': embedding_stats,
        'collection_stats': collection_stats,
        'search_performance': {
            'average_search_time': avg_search_time,
            'maximum_search_time': max_search_time,
            'under_1_second': all(t < 1.0 for t in search_times),
            'test_queries_count': len(search_results)
        },
        'filtering_performance': {
            'filter_success_rate_percent': filter_success_rate,
            'categories_tested': list(filtering_results.keys()),
            'compound_filter_success': filtering_results.get('compound_filter', {}).get('filter_success', False)
        },
        'validation_results': {
            'all_chunks_embedded': embedding_stats['total_texts_processed'] == total_chunks,
            'collection_size_matches': collection_stats['total_documents'] == total_chunks,
            'search_latency_acceptable': avg_search_time < 1.0,
            'filtering_works': filter_success_rate > 80.0
        }
    }
    
    # Save to file
    stats_file = config.PROCESSED_DATA_PATH / 'embedding_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"   ✅ Statistics saved to {stats_file}")
    print(f"   📊 File size: {stats_file.stat().st_size:,} bytes")
    
    return stats


def print_final_summary(stats: Dict[str, Any]) -> None:
    """Print final processing summary."""
    print(f"\n" + "="*60)
    print(f"PHASE 4 VECTOR DATABASE POPULATION SUMMARY")
    print(f"="*60)
    
    # Processing summary
    proc_stats = stats['processing_metadata']
    print(f"📊 Processing Overview:")
    print(f"   Total chunks: {proc_stats['total_chunks_processed']:,}")
    print(f"   Processing time: {proc_stats['total_processing_time_seconds']:.1f}s")
    print(f"   Rate: {proc_stats['chunks_per_second']:.1f} chunks/second")
    
    # Embedding summary
    emb_stats = stats['embedding_stats']
    print(f"\n🔌 Embedding Generation:")
    print(f"   New embeddings: {emb_stats['new_embeddings_generated']:,}")
    print(f"   Cache hits: {emb_stats['cache_hits']:,}")
    print(f"   Cache hit rate: {emb_stats['cache_hit_rate_percent']:.1f}%")
    print(f"   API calls: {emb_stats['api_calls_made']:,}")
    
    # Collection summary
    coll_stats = stats['collection_stats']
    print(f"\n📚 Vector Store:")
    print(f"   Collection: {coll_stats['collection_name']}")
    print(f"   Documents stored: {coll_stats['total_documents']:,}")
    print(f"   Categories: {', '.join(coll_stats['categories_found'])}")
    
    # Performance summary
    search_stats = stats['search_performance']
    filter_stats = stats['filtering_performance']
    print(f"\n⚡ Performance:")
    print(f"   Avg search time: {search_stats['average_search_time']:.3f}s")
    print(f"   Under 1 second: {'✅' if search_stats['under_1_second'] else '❌'}")
    print(f"   Filter accuracy: {filter_stats['filter_success_rate_percent']:.1f}%")
    
    # Validation summary
    validation = stats['validation_results']
    passed_checks = sum(validation.values())
    total_checks = len(validation)
    
    print(f"\n✅ Validation Results:")
    print(f"   All chunks embedded: {'✅' if validation['all_chunks_embedded'] else '❌'}")
    print(f"   Collection size matches: {'✅' if validation['collection_size_matches'] else '❌'}")
    print(f"   Search latency OK: {'✅' if validation['search_latency_acceptable'] else '❌'}")
    print(f"   Filtering works: {'✅' if validation['filtering_works'] else '❌'}")
    print(f"   Overall: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")


def main(test_mode: bool = False) -> bool:
    """
    Main function to populate vector database.
    
    Args:
        test_mode: If True, process only a small subset of data
        
    Returns:
        True if successful, False otherwise
    """
    start_time = time.time()
    
    print(f"🚀 Vector Database Population Pipeline")
    print(f"=" * 50)
    if test_mode:
        print("🧪 TEST MODE: Processing subset of data")
    else:
        print("Executing Phase 4: Vector Database Setup")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        config.validate_config()
        
        # Load processed chunks
        chunks = load_processed_chunks(config, test_mode)
        total_chunks = len(chunks)
        
        # Prepare data for vector database
        documents, metadatas, ids = prepare_vectordb_data(chunks)
        
        # Initialize vector store
        db_path = Path("data") / ("test_chroma_db" if test_mode else "chroma_db")
        collection_name = "test_bbc_chunks" if test_mode else "bbc_news_chunks"
        
        print(f"\n🗄️  Initializing vector store...")
        vector_store = VectorStore(
            db_path=db_path,
            collection_name=collection_name,
            api_key=config.OPENAI_API_KEY
        )
        
        # Reset collection in test mode
        if test_mode:
            vector_store.reset_collection()
        
        # Add documents to vector store
        print(f"\n📚 Populating vector database...")
        vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            batch_size=50 if test_mode else 100,
            progress_interval=5 if test_mode else 20
        )
        
        # Verify insertion
        final_count = vector_store.collection.count()
        print(f"✅ Vector database populated with {final_count:,} documents")
        
        if final_count != total_chunks:
            print(f"⚠️  Warning: Expected {total_chunks:,} documents, but collection has {final_count:,}")
        
        # Test search functionality
        search_results = test_search_functionality(vector_store, test_mode)
        
        # Test metadata filtering
        filtering_results = verify_metadata_filtering(vector_store)
        
        # Print vector store statistics
        vector_store.print_stats()
        
        # Save statistics
        processing_time = time.time() - start_time
        stats = save_embedding_stats(
            config, 
            vector_store.embedder, 
            vector_store,
            search_results,
            filtering_results,
            total_chunks,
            processing_time
        )
        
        # Print final summary
        print_final_summary(stats)
        
        print(f"\n✅ Phase 4 Complete! Vector database ready for use.")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        # Print debugging info
        import traceback
        print(f"\n🔍 Full error traceback:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    import sys
    
    # Check if we should run in full mode
    test_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        test_mode = False
        print("⚠️  Full mode requested - will process all chunks")
    
    success = main(test_mode)
    sys.exit(0 if success else 1)