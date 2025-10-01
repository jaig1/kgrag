"""
Test the new singular knowledge graph-informed approach.
"""
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from retrieval.hybrid_retriever import HybridRetriever

def test_singular_approach():
    """Test the new singular knowledge graph-informed retrieval approach."""
    
    print("🧪 TESTING SINGULAR KNOWLEDGE GRAPH-INFORMED APPROACH")
    print("=" * 60)
    
    # Initialize retriever
    print("🚀 Initializing HybridRetriever...")
    retriever = HybridRetriever()
    
    # Test query
    query = "How is artificial intelligence impacting different industries?"
    print(f"\n📋 Testing with query: '{query}'")
    
    # Time the retrieval
    start_time = time.time()
    result = retriever.retrieve(query)
    end_time = time.time()
    
    # Results
    print(f"\n📊 RESULTS:")
    print(f"   Success: {result['success']}")
    print(f"   Retrieval time: {end_time - start_time:.3f}s")
    print(f"   Results count: {len(result['results'])}")
    
    # Show graph enhancement
    graph_data = result.get('graph_data', {})
    if graph_data.get('success', False):
        print(f"\n🕸️  GRAPH ENHANCEMENT:")
        print(f"   Entities extracted: {len(graph_data.get('extracted_entities', []))}")
        print(f"   Related entities: {len(graph_data.get('related_entities', []))}")
        print(f"   Connected articles: {len(graph_data.get('connected_articles', []))}")
        
        # Show extracted entities
        if graph_data.get('extracted_entities'):
            print(f"   Extracted entities: {', '.join(graph_data['extracted_entities'])}")
        
        # Show related entities (first 5)
        if graph_data.get('related_entities'):
            related_preview = graph_data['related_entities'][:5]
            print(f"   Related entities: {', '.join(related_preview)}")
    
    # Show top results
    print(f"\n📄 TOP 3 RESULTS:")
    for i, doc in enumerate(result['results'][:3], 1):
        content_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
        print(f"   {i}. Score: {doc['score']:.4f}")
        print(f"      Content: {content_preview}")
        print(f"      Article: {doc['article_id']}")
    
    # Show query enhancement
    print(f"\n🔍 QUERY ENHANCEMENT:")
    print(f"   Original: {result['query']}")
    enhanced = result.get('enhanced_query', result['query'])
    enhanced_preview = enhanced[:100] + "..." if len(enhanced) > 100 else enhanced
    print(f"   Enhanced: {enhanced_preview}")
    
    print(f"\n✅ Test complete! Retrieved {len(result['results'])} results in {end_time - start_time:.3f}s")
    
    return result

if __name__ == "__main__":
    test_singular_approach()