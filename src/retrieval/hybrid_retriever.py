#!/usr/bin/env python3
"""
Phase 6 - Script 4: Hybrid Retriever

This module combines vector-based retrieval with knowledge graph-enhanced retrieval
using Reciprocal Rank Fusion (RRF) to provide comprehensive and contextually rich results.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retrieval.query_expander import QueryExpander
from storage.vector_store import VectorStore


class HybridRetriever:
    """
    Combines vector-based and graph-enhanced retrieval using Reciprocal Rank Fusion.
    
    This retriever:
    1. Uses QueryExpander to enhance queries with graph context
    2. Performs vector similarity search on both original and expanded queries
    3. Applies Reciprocal Rank Fusion to merge results
    4. Returns ranked, diverse, and contextually relevant results
    """
    
    def __init__(self, vector_store_path: str = "data/processed/vector_store.pkl"):
        """Initialize the HybridRetriever with required components."""
        print("🚀 Initializing HybridRetriever...")
        
        # Initialize components
        self.query_expander = QueryExpander()
        self.vector_store = VectorStore(
            db_path="data/chroma_db",
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Get vector store statistics
        try:
            stats = self.vector_store.get_collection_stats()
            print(f"📊 Vector store stats: {stats.get('count', 0)} documents")
        except Exception as e:
            print(f"⚠️  Could not get vector store stats: {e}")
        
        # Configuration parameters
        self.config = {
            'max_results': 8,         # Maximum final results to return
            'max_hops': 2,            # Maximum graph traversal hops
            'max_related_entities': 8, # Maximum related entities for expansion
            'include_entity_context': True, # Whether to include entity context
        }
        
        # Statistics tracking
        self.stats = {
            'queries_processed': 0,
            'graph_searches': 0,
            'vector_searches': 0,
            'total_retrieval_time': 0.0,
            'average_retrieval_time': 0.0,
        }
        
        print(f"✅ HybridRetriever initialized successfully")
        print(f"   Configuration: max_results={self.config['max_results']}, max_hops={self.config['max_hops']}")
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Perform knowledge graph-informed retrieval using a singular approach.
        
        Step 1: Knowledge Graph Search - Extract entities and discover relationships
        Step 2: Graph-Informed Vector Search - Use graph insights for enhanced vector search
        
        Args:
            query: User query string
            
        Returns:
            Dict containing retrieved results and metadata
        """
        start_time = time.time()
        
        print(f"\n🔍 KNOWLEDGE GRAPH-INFORMED RETRIEVAL: '{query}'")
        print("-" * 70)
        
        try:
            # Validate input query
            if not query or not query.strip():
                return {
                    'query': query,
                    'results': [],
                    'graph_data': {},
                    'retrieval_metadata': {
                        'retrieval_time': time.time() - start_time,
                        'error': 'Empty or invalid query provided'
                    },
                    'success': False,
                    'error': 'Empty or invalid query provided'
                }
            
            # Step 1: Knowledge Graph Search
            print("�️  Step 1: Knowledge Graph Search & Entity Discovery...")
            
            graph_data = self._knowledge_graph_search(query)
            self.stats['graph_searches'] += 1
            
            print(f"   Extracted entities: {len(graph_data.get('extracted_entities', []))}")
            print(f"   Related entities found: {len(graph_data.get('related_entities', []))}")
            print(f"   Connected articles: {len(graph_data.get('connected_articles', []))}")
            
            # Step 2: Graph-Informed Vector Search
            print(f"\n🔍 Step 2: Graph-Informed Vector Search...")
            
            enhanced_query = self._create_enhanced_query(query, graph_data)
            print(f"   Enhanced query: {enhanced_query[:120]}...")
            
            # Single optimized vector search with graph insights
            results = self._vector_search(enhanced_query, k=self.config['max_results'])
            self.stats['vector_searches'] += 1
            
            print(f"   Found {len(results)} results from enhanced search")
            
            # Calculate metadata
            retrieval_time = time.time() - start_time
            
            retrieval_metadata = {
                'retrieval_time': retrieval_time,
                'results_count': len(results),
                'entities_extracted': len(graph_data.get('extracted_entities', [])),
                'entities_related': len(graph_data.get('related_entities', [])),
                'articles_connected': len(graph_data.get('connected_articles', [])),
                'graph_search_success': graph_data.get('success', False),
                'enhancement_applied': len(graph_data.get('related_entities', [])) > 0
            }
            
            # Update statistics
            self.stats['queries_processed'] += 1
            self.stats['total_retrieval_time'] += retrieval_time
            self.stats['average_retrieval_time'] = (
                self.stats['total_retrieval_time'] / self.stats['queries_processed']
            )
            
            print(f"✅ Knowledge graph-informed retrieval complete in {retrieval_time:.3f}s")
            
            return {
                'query': query,
                'results': results,
                'graph_data': graph_data,
                'enhanced_query': enhanced_query,
                'retrieval_metadata': retrieval_metadata,
                'success': True
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ Knowledge graph-informed retrieval failed: {str(e)}")
            
            return {
                'query': query,
                'results': [],
                'graph_data': {},
                'retrieval_metadata': {
                    'retrieval_time': error_time,
                    'error': str(e)
                },
                'success': False,
                'error': str(e)
            }
    
    def _vector_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of search results with scores and content
        """
        # Validate query input
        if not query or not query.strip():
            print(f"⚠️  Empty or invalid query, returning no results")
            return []
        
        try:
            # Use VectorStore's similarity_search method
            search_results = self.vector_store.similarity_search(query.strip(), top_k=k)
            
            # Convert to standard format
            results = []
            documents = search_results.get('documents', [])
            metadatas = search_results.get('metadatas', [])
            distances = search_results.get('distances', [])
            ids = search_results.get('ids', [])
            
            for i, (doc, metadata, distance, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                # Convert distance to similarity score (lower distance = higher similarity)
                score = 1.0 / (1.0 + distance) if distance is not None else 0.0
                
                results.append({
                    'rank': i + 1,
                    'score': score,
                    'content': doc,
                    'metadata': metadata or {},
                    'chunk_id': doc_id,
                    'article_id': metadata.get('article_id', 'unknown') if metadata else 'unknown'
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Vector search error: {str(e)}")
            return []
    
    def _knowledge_graph_search(self, query: str) -> Dict[str, Any]:
        """
        Perform knowledge graph search to extract entities and discover relationships.
        
        Args:
            query: User query string
            
        Returns:
            Dict containing graph search results
        """
        try:
            print("📋   Analyzing query for entities...")
            
            # Step 1: Extract entities from query
            analysis_result = self.query_expander.query_analyzer.analyze_query(query)
            
            if not analysis_result or 'extracted_entities' not in analysis_result:
                return {
                    'extracted_entities': [],
                    'related_entities': [],
                    'connected_articles': [],
                    'entity_contexts': {},
                    'success': False,
                    'error': 'Failed to analyze query'
                }
            
            # Parse extracted entities
            extracted_entities_raw = analysis_result['extracted_entities']
            extracted_entities = []
            
            if isinstance(extracted_entities_raw, dict):
                for category, entities in extracted_entities_raw.items():
                    if isinstance(entities, list):
                        extracted_entities.extend(entities)
                    elif entities:
                        extracted_entities.append(entities)
            elif isinstance(extracted_entities_raw, list):
                extracted_entities = extracted_entities_raw
            
            print(f"📋   Found {len(extracted_entities)} entities: {extracted_entities}")
            
            # Step 2: Graph traversal for related entities (only if entities found)
            related_entities = []
            connected_articles = set()
            entity_contexts = {}
            
            if extracted_entities:
                print("📋   Performing graph traversal...")
                
                # Use graph traversal to find related entities
                multi_result = self.query_expander.graph_traversal.find_related_entities(
                    extracted_entities, 
                    max_hops=self.config['max_hops']
                )
                
                if multi_result and 'related_entities' in multi_result:
                    related_entities = multi_result['related_entities'][:self.config['max_related_entities']]
                    
                    # Collect connected articles from traversal results
                    for entity, traversal_result in multi_result.get('traversal_results', {}).items():
                        if 'related_articles' in traversal_result:
                            connected_articles.update(traversal_result['related_articles'])
                    
                    print(f"📋   Discovered {len(related_entities)} related entities")
                    print(f"📋   Found {len(connected_articles)} connected articles")
                    
                    # Get context for key entities if requested
                    if self.config['include_entity_context'] and related_entities:
                        print("📋   Retrieving entity contexts...")
                        
                        for entity in related_entities[:5]:  # Limit to top 5 for performance
                            context = self.query_expander.graph_traversal.get_entity_context(entity)
                            if 'error' not in context:
                                entity_contexts[entity] = {
                                    'type': context['entity_attributes'].get('type', 'unknown'),
                                    'name': context['entity_attributes'].get('name', entity),
                                    'articles_count': context['articles_count'],
                                    'entities_count': context['entities_count']
                                }
                else:
                    print("📋   No related entities found via graph traversal")
            else:
                print("📋   No entities extracted, skipping graph traversal")
            
            return {
                'extracted_entities': extracted_entities,
                'related_entities': related_entities,
                'connected_articles': list(connected_articles),
                'entity_contexts': entity_contexts,
                'query_intent': analysis_result.get('intent_type', 'unknown'),
                'complexity_score': analysis_result.get('complexity_score', 1),
                'success': True
            }
            
        except Exception as e:
            print(f"❌   Graph search error: {str(e)}")
            return {
                'extracted_entities': [],
                'related_entities': [],
                'connected_articles': [],
                'entity_contexts': {},
                'success': False,
                'error': str(e)
            }
    
    def _create_enhanced_query(self, original_query: str, graph_data: Dict[str, Any]) -> str:
        """
        Create an enhanced query using graph insights.
        
        Args:
            original_query: Original user query
            graph_data: Results from knowledge graph search
            
        Returns:
            Enhanced query string
        """
        # Start with original query
        enhanced_parts = [original_query]
        
        # Add extracted entities
        extracted_entities = graph_data.get('extracted_entities', [])
        if extracted_entities:
            enhanced_parts.append(f"Entities: {' '.join(extracted_entities)}")
        
        # Add related entities (grouped by type if we have context)
        related_entities = graph_data.get('related_entities', [])
        entity_contexts = graph_data.get('entity_contexts', {})
        
        if related_entities:
            # Group entities by type if we have context
            entities_by_type = {}
            entities_without_context = []
            
            for entity in related_entities[:6]:  # Limit for query length
                if entity in entity_contexts:
                    entity_type = entity_contexts[entity]['type']
                    entity_name = entity_contexts[entity]['name']
                    
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity_name)
                else:
                    # Clean entity name (remove type prefix if present)
                    clean_name = entity.split(':', 1)[-1] if ':' in entity else entity
                    entities_without_context.append(clean_name)
            
            # Add grouped entities
            for entity_type, names in entities_by_type.items():
                if names:
                    enhanced_parts.append(f"{entity_type.capitalize()}: {' '.join(names[:3])}")
            
            # Add ungrouped entities
            if entities_without_context:
                enhanced_parts.append(f"Related: {' '.join(entities_without_context[:3])}")
        
        # Join all parts
        enhanced_query = ' | '.join(enhanced_parts)
        
        return enhanced_query
    

    
    def get_result_summary(self, results: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of retrieval results.
        
        Args:
            results: Results from retrieve() method
            
        Returns:
            String summary of the results
        """
        if not results['success']:
            return f"Retrieval failed: {results.get('error', 'Unknown error')}"
        
        metadata = results['retrieval_metadata']
        graph_data = results['graph_data']
        
        summary_parts = []
        
        # Basic stats
        summary_parts.append(f"Retrieved {metadata['results_count']} results in {metadata['retrieval_time']:.3f}s")
        
        # Graph enhancement info
        if metadata.get('enhancement_applied', False):
            summary_parts.append(f"Graph-enhanced search")
            summary_parts.append(f"Entities: {metadata['entities_extracted']} extracted, {metadata['entities_related']} related")
            summary_parts.append(f"Connected articles: {metadata['articles_connected']}")
        else:
            summary_parts.append("Standard vector search (no graph enhancement)")
        
        # Query enhancement
        if len(results.get('enhanced_query', '')) > len(results.get('query', '')):
            summary_parts.append("Query enhanced with graph insights")
        
        return " | ".join(summary_parts)
    
    def print_statistics(self) -> None:
        """Print current retrieval statistics."""
        print(f"\n📊 KNOWLEDGE GRAPH-INFORMED RETRIEVAL STATISTICS")
        print("=" * 60)
        print(f"🔍 Queries processed: {self.stats['queries_processed']}")
        print(f"�️  Graph searches: {self.stats['graph_searches']}")
        print(f"� Vector searches: {self.stats['vector_searches']}")
        print(f"⏱️  Total retrieval time: {self.stats['total_retrieval_time']:.3f}s")
        
        if self.stats['queries_processed'] > 0:
            print(f"📈 Average retrieval time: {self.stats['average_retrieval_time']:.3f}s")
            graph_usage_rate = (self.stats['graph_searches'] / self.stats['queries_processed']) * 100
            print(f"📈 Graph search usage rate: {graph_usage_rate:.1f}%")


# Test function for single retrieval
def test_single_knowledge_graph_retrieval() -> Dict[str, Any]:
    """Test HybridRetriever with the new knowledge graph-informed approach."""
    
    print("🧪 TESTING KNOWLEDGE GRAPH-INFORMED RETRIEVAL")
    print("=" * 60)
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Test with a query that should benefit from graph enhancement
    test_query = "What is Microsoft's role in the technology industry?"
    
    print(f"\n📋 Testing with query: '{test_query}'")
    
    # Perform knowledge graph-informed retrieval
    result = retriever.retrieve(query=test_query)
    
    # Display results
    print(f"\n📊 RETRIEVAL RESULTS:")
    print(f"   Success: {result['success']}")
    
    if result['success']:
        metadata = result['retrieval_metadata']
        graph_data = result['graph_data']
        
        print(f"   Retrieval time: {metadata['retrieval_time']:.3f}s")
        print(f"   Results count: {metadata['results_count']}")
        
        print(f"\n�️  GRAPH ENHANCEMENT:")
        print(f"   Entities extracted: {metadata['entities_extracted']}")
        print(f"   Related entities: {metadata['entities_related']}")
        print(f"   Connected articles: {metadata['articles_connected']}")
        print(f"   Enhancement applied: {metadata['enhancement_applied']}")
        
        print(f"\n📝 RESULT SUMMARY:")
        summary = retriever.get_result_summary(result)
        print(f"   {summary}")
        
        print(f"\n📄 TOP 3 RESULTS:")
        for i, res in enumerate(result['results'][:3]):
            print(f"   {i+1}. Score: {res['score']:.4f}")
            print(f"      Content: {res['content'][:100]}...")
            print(f"      Article: {res.get('article_id', 'unknown')}")
        
        # Show enhancement details
        print(f"\n🔍 QUERY ENHANCEMENT:")
        print(f"   Original: {result['query']}")
        print(f"   Enhanced: {result.get('enhanced_query', 'N/A')[:150]}...")
        
        if graph_data.get('extracted_entities'):
            print(f"   Extracted entities: {', '.join(graph_data['extracted_entities'])}")
        
        if graph_data.get('related_entities'):
            print(f"   Related entities: {', '.join(graph_data['related_entities'][:5])}")
    
    else:
        print(f"\n❌ Retrieval failed:")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Print statistics
    retriever.print_statistics()
    
    print(f"\n✅ Knowledge graph-informed retrieval test complete!")
    
    return result


if __name__ == "__main__":
    # Test with single query
    result = test_single_knowledge_graph_retrieval()