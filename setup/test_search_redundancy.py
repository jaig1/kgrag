#!/usr/bin/env python3
"""
Test to compare original vs expanded vector searches to see if original is redundant.
"""

import sys
import os

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retrieval.hybrid_retriever import HybridRetriever


def analyze_search_redundancy():
    """Analyze whether original vector search adds value over expanded search."""
    
    print("🔍 ANALYZING SEARCH REDUNDANCY")
    print("=" * 50)
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    test_query = "Who is Andy Murray?"
    
    print(f"\n📋 Testing with: '{test_query}'")
    
    # Step 1: Get original vector search results
    print("\n🔍 Original Vector Search Results:")
    original_results = retriever._vector_search(test_query, k=10)
    
    print(f"   Found {len(original_results)} results")
    for i, result in enumerate(original_results[:3]):
        print(f"   {i+1}. Score: {result['score']:.4f} | {result['content'][:80]}...")
    
    # Step 2: Get expanded query
    print("\n🔍 Query Expansion:")
    expansion_result = retriever.query_expander.expand_query(test_query)
    
    if expansion_result['success']:
        expanded_query = expansion_result['expanded_query']
        print(f"   Original: {test_query}")
        print(f"   Expanded: {expanded_query[:150]}...")
        
        # Step 3: Get expanded vector search results
        print("\n🔍 Expanded Vector Search Results:")
        expanded_results = retriever._vector_search(expanded_query, k=10)
        
        print(f"   Found {len(expanded_results)} results")
        for i, result in enumerate(expanded_results[:3]):
            print(f"   {i+1}. Score: {result['score']:.4f} | {result['content'][:80]}...")
        
        # Step 4: Analyze overlap
        print("\n📊 Overlap Analysis:")
        original_chunk_ids = {r['chunk_id'] for r in original_results}
        expanded_chunk_ids = {r['chunk_id'] for r in expanded_results}
        
        overlap = original_chunk_ids.intersection(expanded_chunk_ids)
        only_original = original_chunk_ids - expanded_chunk_ids
        only_expanded = expanded_chunk_ids - original_chunk_ids
        
        print(f"   Original unique chunks: {len(original_chunk_ids)}")
        print(f"   Expanded unique chunks: {len(expanded_chunk_ids)}")
        print(f"   Overlapping chunks: {len(overlap)}")
        print(f"   Only in original: {len(only_original)}")
        print(f"   Only in expanded: {len(only_expanded)}")
        
        overlap_percentage = len(overlap) / len(original_chunk_ids) * 100 if original_chunk_ids else 0
        print(f"   Overlap percentage: {overlap_percentage:.1f}%")
        
        # Step 5: Quality comparison
        print("\n📈 Quality Analysis:")
        
        # Compare top result scores
        top_original_score = original_results[0]['score'] if original_results else 0
        top_expanded_score = expanded_results[0]['score'] if expanded_results else 0
        
        print(f"   Top original score: {top_original_score:.4f}")
        print(f"   Top expanded score: {top_expanded_score:.4f}")
        
        # Recommendation
        print("\n💡 RECOMMENDATION:")
        
        if overlap_percentage > 70:
            print("   ❌ HIGH REDUNDANCY: Original search appears largely redundant")
            print("   💡 Consider removing original search, use expanded only")
        elif len(only_original) > len(only_expanded):
            print("   ✅ VALUE ADDED: Original search finds unique valuable results")
            print("   💡 Keep both searches for maximum coverage")
        elif top_expanded_score > top_original_score * 1.2:
            print("   ⚠️  EXPANDED SUPERIOR: Expanded search significantly outperforms original")
            print("   💡 Consider making expanded search primary, original secondary")
        else:
            print("   ⚖️  BALANCED: Both searches contribute complementary value")
            print("   💡 Current hybrid approach is justified")
    
    else:
        print("   ❌ Query expansion failed, cannot compare")


if __name__ == "__main__":
    # Set environment variables
    import os
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass
    
    analyze_search_redundancy()