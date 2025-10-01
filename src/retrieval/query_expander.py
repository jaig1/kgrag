#!/usr/bin/env python3
"""
Phase 6 - Script 3: Query Expander

This module expands user queries with related entities and context discovered
from the knowledge graph, enhancing the retrieval process with graph-based insights.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import time

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retrieval.query_analyzer import QueryAnalyzer
from retrieval.graph_traversal import GraphTraversal


class QueryExpander:
    """
    Expands queries using knowledge graph traversal to find related entities,
    contexts, and semantic relationships that can enhance retrieval.
    """
    
    def __init__(self):
        """Initialize the QueryExpander with required components."""
        print("🚀 Initializing QueryExpander...")
        
        # Initialize components
        self.query_analyzer = QueryAnalyzer()
        self.graph_traversal = GraphTraversal()
        
        # Statistics tracking
        self.stats = {
            'queries_expanded': 0,
            'entities_discovered': 0,
            'total_expansion_time': 0.0,
            'average_expansion_time': 0.0
        }
        
        print(f"✅ QueryExpander initialized successfully")
    
    def expand_query(
        self, 
        query: str, 
        max_hops: int = 2,
        max_related_entities: int = 10,
        include_entity_context: bool = True
    ) -> Dict[str, Any]:
        """
        Expand a query using knowledge graph traversal.
        
        Args:
            query: Original user query
            max_hops: Maximum hops for graph traversal
            max_related_entities: Maximum related entities to include
            include_entity_context: Whether to include detailed entity context
            
        Returns:
            Dict containing expanded query information
        """
        start_time = time.time()
        
        print(f"\n🔍 EXPANDING QUERY: '{query}'")
        print("-" * 50)
        
        try:
            # Step 1: Analyze the query to extract entities and intent
            print("📋 Step 1: Analyzing query...")
            analysis_result = self.query_analyzer.analyze_query(query)
            
            if not analysis_result or 'extracted_entities' not in analysis_result:
                return {
                    'original_query': query,
                    'expanded_query': query,
                    'entities_found': [],
                    'related_entities': [],
                    'entity_contexts': {},
                    'expansion_summary': 'No entities found for expansion',
                    'expansion_time': time.time() - start_time,
                    'success': False
                }
            
            extracted_entities_raw = analysis_result['extracted_entities']
            # Extract intent information from analysis result
            query_intent = {
                'primary_intent': analysis_result.get('intent_type', 'unknown'),
                'confidence': analysis_result.get('intent_confidence', 0.0),
                'requires_graph_traversal': analysis_result.get('requires_graph_traversal', False)
            }
            
            # Parse entities from the structured format
            extracted_entities = []
            if isinstance(extracted_entities_raw, dict):
                for category, entities in extracted_entities_raw.items():
                    if isinstance(entities, list):
                        extracted_entities.extend(entities)
                    elif entities:  # Single entity as string
                        extracted_entities.append(entities)
            elif isinstance(extracted_entities_raw, list):
                extracted_entities = extracted_entities_raw
            
            print(f"   Extracted entities: {extracted_entities}")
            print(f"   Query intent: {query_intent.get('primary_intent', 'unknown')} (confidence: {query_intent.get('confidence', 0.0):.2f})")
            print(f"   Requires graph traversal: {query_intent.get('requires_graph_traversal', False)}")
            
            # Step 2: Find related entities through graph traversal
            print("\n🕸️  Step 2: Graph traversal for related entities...")
            
            all_related_entities = []
            entity_relationships = {}
            traversal_stats = {
                'entities_per_source': {},
                'total_unique_entities': 0
            }
            
            if extracted_entities and query_intent.get('requires_graph_traversal', False):
                # Use multi-entity traversal to find relationships
                multi_result = self.graph_traversal.find_related_entities(
                    extracted_entities, 
                    max_hops=max_hops
                )
                
                entity_relationships = multi_result['entity_relationships']
                all_related_entities = multi_result['related_entities']
                traversal_stats['total_unique_entities'] = multi_result['total_related_count']
                
                for entity in extracted_entities:
                    if entity in entity_relationships:
                        count = len(entity_relationships[entity])
                        traversal_stats['entities_per_source'][entity] = count
                        print(f"   '{entity}' → {count} related entities")
                
                print(f"   Total unique related entities: {traversal_stats['total_unique_entities']}")
            else:
                print("   Skipping graph traversal (not required for this query type)")
            
            # Step 3: Select top related entities
            print(f"\n🏷️  Step 3: Selecting top {max_related_entities} related entities...")
            
            # Limit the number of related entities to include
            selected_related = all_related_entities[:max_related_entities]
            
            print(f"   Selected {len(selected_related)} entities for expansion")
            
            # Step 4: Get entity contexts if requested
            entity_contexts = {}
            
            if include_entity_context and selected_related:
                print(f"\n📖 Step 4: Retrieving entity contexts...")
                
                # Get context for up to 5 most relevant entities
                for entity in selected_related[:5]:
                    context = self.graph_traversal.get_entity_context(entity)
                    if 'error' not in context:
                        entity_contexts[entity] = {
                            'type': context['entity_attributes'].get('type', 'unknown'),
                            'name': context['entity_attributes'].get('name', entity),
                            'articles_count': context['articles_count'],
                            'entities_count': context['entities_count'],
                            'relationship_types': context['relationship_types']
                        }
                        print(f"   '{entity}': {context['articles_count']} articles, {context['entities_count']} connections")
                    else:
                        print(f"   ❌ Failed to get context for '{entity}': {context['error']}")
            
            # Step 5: Create expanded query
            print(f"\n✨ Step 5: Creating expanded query...")
            
            expanded_query = self._create_expanded_query(
                query, 
                extracted_entities, 
                selected_related, 
                entity_contexts,
                query_intent
            )
            
            # Calculate expansion metrics
            expansion_time = time.time() - start_time
            
            # Update statistics
            self.stats['queries_expanded'] += 1
            self.stats['entities_discovered'] += len(selected_related)
            self.stats['total_expansion_time'] += expansion_time
            self.stats['average_expansion_time'] = (
                self.stats['total_expansion_time'] / self.stats['queries_expanded']
            )
            
            # Create expansion summary
            expansion_summary = self._create_expansion_summary(
                extracted_entities,
                selected_related,
                entity_contexts,
                traversal_stats
            )
            
            print(f"✅ Query expansion complete in {expansion_time:.3f}s")
            
            return {
                'original_query': query,
                'expanded_query': expanded_query,
                'extracted_entities': extracted_entities,
                'related_entities': selected_related,
                'entity_relationships': entity_relationships,
                'entity_contexts': entity_contexts,
                'query_intent': query_intent,
                'expansion_summary': expansion_summary,
                'traversal_stats': traversal_stats,
                'expansion_time': expansion_time,
                'success': True
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ Query expansion failed: {str(e)}")
            
            return {
                'original_query': query,
                'expanded_query': query,
                'entities_found': [],
                'related_entities': [],
                'entity_contexts': {},
                'expansion_summary': f'Expansion failed: {str(e)}',
                'expansion_time': error_time,
                'success': False,
                'error': str(e)
            }
    
    def _create_expanded_query(
        self,
        original_query: str,
        extracted_entities: List[str],
        related_entities: List[str],
        entity_contexts: Dict[str, Any],
        query_intent: Dict[str, Any]
    ) -> str:
        """
        Create an expanded query incorporating related entities and context.
        
        Args:
            original_query: The original user query
            extracted_entities: Entities found in the original query
            related_entities: Related entities from graph traversal
            entity_contexts: Context information for entities
            query_intent: Classified intent of the query
            
        Returns:
            Expanded query string
        """
        
        # Start with original query
        expanded_parts = [f"Original query: {original_query}"]
        
        # Add extracted entities context
        if extracted_entities:
            extracted_str = ", ".join(extracted_entities)
            expanded_parts.append(f"Key entities: {extracted_str}")
        
        # Add related entities for broader context
        if related_entities:
            # Group related entities by type if we have context
            entities_by_type = {}
            entities_without_context = []
            
            for entity in related_entities[:8]:  # Limit to 8 for readability
                if entity in entity_contexts:
                    entity_type = entity_contexts[entity]['type']
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity_contexts[entity]['name'])
                else:
                    entities_without_context.append(entity)
            
            # Add grouped entities
            for entity_type, names in entities_by_type.items():
                if names:
                    names_str = ", ".join(names[:3])  # Limit to 3 per type
                    expanded_parts.append(f"Related {entity_type}s: {names_str}")
            
            # Add ungrouped entities
            if entities_without_context:
                others_str = ", ".join(entities_without_context[:3])
                expanded_parts.append(f"Related entities: {others_str}")
        
        # Add intent-specific context
        intent = query_intent.get('primary_intent', '')
        if intent == 'relational':
            expanded_parts.append("Focus on relationships and connections between entities")
        elif intent == 'temporal':
            expanded_parts.append("Consider temporal aspects and chronological information")
        elif intent == 'comparative':
            expanded_parts.append("Look for comparisons and contrasts between entities")
        elif intent == 'multi_hop':
            expanded_parts.append("Explore indirect connections and multi-step relationships")
        
        return " | ".join(expanded_parts)
    
    def _create_expansion_summary(
        self,
        extracted_entities: List[str],
        related_entities: List[str],
        entity_contexts: Dict[str, Any],
        traversal_stats: Dict[str, Any]
    ) -> str:
        """Create a summary of the expansion process."""
        
        summary_parts = []
        
        if extracted_entities:
            summary_parts.append(f"Found {len(extracted_entities)} entities in query")
        
        if related_entities:
            summary_parts.append(f"Discovered {len(related_entities)} related entities via graph traversal")
        
        if entity_contexts:
            summary_parts.append(f"Retrieved context for {len(entity_contexts)} key entities")
        
        if traversal_stats.get('total_unique_entities', 0) > 0:
            summary_parts.append(f"Total graph entities explored: {traversal_stats['total_unique_entities']}")
        
        if not summary_parts:
            return "No expansion performed"
        
        return "; ".join(summary_parts)
    
    def print_statistics(self) -> None:
        """Print current expansion statistics."""
        print(f"\n📊 QUERY EXPANSION STATISTICS")
        print("=" * 50)
        print(f"🔍 Queries expanded: {self.stats['queries_expanded']}")
        print(f"🏷️  Total entities discovered: {self.stats['entities_discovered']}")
        print(f"⏱️  Total expansion time: {self.stats['total_expansion_time']:.3f}s")
        
        if self.stats['queries_expanded'] > 0:
            avg_entities = self.stats['entities_discovered'] / self.stats['queries_expanded']
            print(f"📈 Average entities per query: {avg_entities:.1f}")
            print(f"📈 Average expansion time: {self.stats['average_expansion_time']:.3f}s")


# Test function for single query expansion
def test_single_query_expansion() -> Dict[str, Any]:
    """Test QueryExpander with a single query example."""
    
    print("🧪 TESTING SINGLE QUERY EXPANSION")
    print("=" * 50)
    
    # Initialize expander
    expander = QueryExpander()
    
    # Test with a complex query that should benefit from expansion
    test_query = "What is Microsoft's role in the technology industry?"
    
    print(f"\n📋 Testing with query: '{test_query}'")
    
    # Expand the query
    result = expander.expand_query(
        query=test_query,
        max_hops=2,
        max_related_entities=8,
        include_entity_context=True
    )
    
    # Display results
    print(f"\n📊 EXPANSION RESULTS:")
    print(f"   Success: {result['success']}")
    print(f"   Expansion time: {result['expansion_time']:.3f}s")
    
    if result['success']:
        print(f"\n🔤 Original Query:")
        print(f"   {result['original_query']}")
        
        print(f"\n✨ Expanded Query:")
        print(f"   {result['expanded_query']}")
        
        print(f"\n🏷️  Extracted Entities ({len(result.get('extracted_entities', []))}):")
        for entity in result.get('extracted_entities', []):
            print(f"   • {entity}")
        
        print(f"\n🕸️  Related Entities ({len(result.get('related_entities', []))}):")
        for i, entity in enumerate(result.get('related_entities', [])[:5]):  # Show first 5
            print(f"   {i+1}. {entity}")
        
        print(f"\n📖 Entity Contexts ({len(result.get('entity_contexts', {}))}):")
        for entity, context in result.get('entity_contexts', {}).items():
            print(f"   • {entity}: {context['type']} ({context['articles_count']} articles)")
        
        print(f"\n📝 Expansion Summary:")
        print(f"   {result.get('expansion_summary', 'No summary available')}")
        
    else:
        print(f"\n❌ Expansion failed:")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Print statistics
    expander.print_statistics()
    
    print(f"\n✅ Single query expansion test complete!")
    
    return result


if __name__ == "__main__":
    # Test with single query
    result = test_single_query_expansion()