#!/usr/bin/env python3
"""
Phase 6 - Script 2 Testing: Graph Traversal Validation

This script validates the GraphTraversal functionality with comprehensive tests
to ensure all components work correctly before integrating into the KG-Enhanced RAG.
"""

import sys
import os
from typing import Dict, List, Any
import time

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retrieval.graph_traversal import GraphTraversal


def test_end_to_end_graph_traversal() -> Dict[str, Any]:
    """
    Comprehensive end-to-end test of GraphTraversal functionality.
    
    Tests:
    1. Graph loading and initialization
    2. Single entity finding
    3. Multi-hop traversal with relationship discovery
    4. Multi-entity relationship analysis
    5. Entity context retrieval
    6. Error handling and edge cases
    
    Returns:
        Dict with test results and statistics
    """
    
    print("🧪 PHASE 6 SCRIPT 2: GRAPH TRAVERSAL VALIDATION")
    print("=" * 60)
    
    results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'test_details': [],
        'graph_stats': {},
        'performance': {}
    }
    
    start_time = time.time()
    
    try:
        # Test 1: Initialize GraphTraversal
        print("📋 TEST 1: GraphTraversal Initialization")
        print("-" * 40)
        
        traversal = GraphTraversal()
        
        if traversal.graph is None:
            print("❌ FAILED: Graph not loaded")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Graph Loading',
                'status': 'FAILED',
                'error': 'Graph is None'
            })
            return results
        else:
            nodes_count = len(traversal.graph.nodes())
            edges_count = len(traversal.graph.edges())
            print(f"✅ PASSED: Graph loaded successfully")
            print(f"   Nodes: {nodes_count:,}")
            print(f"   Edges: {edges_count:,}")
            
            results['tests_passed'] += 1
            results['graph_stats'] = {
                'nodes': nodes_count,
                'edges': edges_count
            }
            results['test_details'].append({
                'test': 'Graph Loading',
                'status': 'PASSED',
                'nodes': nodes_count,
                'edges': edges_count
            })
        
        # Test 2: Entity Finding
        print(f"\n📋 TEST 2: Entity Finding Capabilities")
        print("-" * 40)
        
        test_entities = ["Microsoft", "tennis", "London", "government"]
        entity_results = {}
        
        for entity in test_entities:
            nodes = traversal.find_entity_in_graph(entity)
            entity_results[entity] = len(nodes)
            print(f"   '{entity}': {len(nodes)} nodes found")
        
        if sum(entity_results.values()) > 0:
            print(f"✅ PASSED: Successfully found entities in graph")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Entity Finding',
                'status': 'PASSED',
                'entities_found': entity_results
            })
        else:
            print(f"❌ FAILED: No entities found in graph")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Entity Finding',
                'status': 'FAILED',
                'error': 'No entities found'
            })
        
        # Test 3: Multi-hop Traversal
        print(f"\n📋 TEST 3: Multi-hop Traversal")
        print("-" * 40)
        
        # Use entity with highest node count for traversal test
        best_entity = max(entity_results.items(), key=lambda x: x[1])
        test_entity = best_entity[0]
        
        print(f"Testing traversal from '{test_entity}' (has {best_entity[1]} nodes)")
        
        traversal_result = traversal.multi_hop_traversal(test_entity, max_hops=2)
        
        print(f"   Start nodes: {len(traversal_result['start_nodes'])}")
        print(f"   Related entities: {traversal_result['entities_count']}")
        print(f"   Related articles: {traversal_result['articles_count']}")
        print(f"   Traversal time: {traversal_result['traversal_time']:.3f}s")
        
        if traversal_result['entities_count'] > 0:
            print(f"✅ PASSED: Multi-hop traversal successful")
            results['tests_passed'] += 1
            results['performance']['traversal_time'] = traversal_result['traversal_time']
            results['test_details'].append({
                'test': 'Multi-hop Traversal',
                'status': 'PASSED',
                'start_entity': test_entity,
                'related_entities': traversal_result['entities_count'],
                'related_articles': traversal_result['articles_count'],
                'time': traversal_result['traversal_time']
            })
        else:
            print(f"❌ FAILED: No entities found via traversal")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Multi-hop Traversal',
                'status': 'FAILED',
                'error': 'No entities found in traversal'
            })
        
        # Test 4: Multi-entity Relationship Analysis
        print(f"\n📋 TEST 4: Multi-entity Relationships")
        print("-" * 40)
        
        # Use top 2 entities with most nodes
        sorted_entities = sorted(entity_results.items(), key=lambda x: x[1], reverse=True)
        multi_entities = [entity[0] for entity in sorted_entities[:2] if entity[1] > 0]
        
        if len(multi_entities) >= 2:
            print(f"Testing relationships between: {multi_entities}")
            
            multi_result = traversal.find_related_entities(multi_entities, max_hops=2)
            
            print(f"   Total unique related entities: {multi_result['total_related_count']}")
            
            for entity, related in multi_result['entity_relationships'].items():
                print(f"   '{entity}': {len(related)} related entities")
            
            if multi_result['total_related_count'] > 0:
                print(f"✅ PASSED: Multi-entity relationship analysis successful")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Multi-entity Relationships',
                    'status': 'PASSED',
                    'entities': multi_entities,
                    'total_related': multi_result['total_related_count']
                })
            else:
                print(f"❌ FAILED: No relationships found")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Multi-entity Relationships',
                    'status': 'FAILED',
                    'error': 'No relationships found'
                })
        else:
            print(f"⚠️ SKIPPED: Not enough entities found for relationship test")
            results['test_details'].append({
                'test': 'Multi-entity Relationships',
                'status': 'SKIPPED',
                'reason': 'Insufficient entities'
            })
        
        # Test 5: Entity Context Retrieval
        print(f"\n📋 TEST 5: Entity Context Retrieval")
        print("-" * 40)
        
        if traversal_result['related_entities']:
            sample_entity = traversal_result['related_entities'][0]
            print(f"Getting context for '{sample_entity}'")
            
            context = traversal.get_entity_context(sample_entity)
            
            if 'error' not in context:
                print(f"   Entity type: {context['entity_attributes'].get('type', 'unknown')}")
                print(f"   Connected articles: {context['articles_count']}")
                print(f"   Connected entities: {context['entities_count']}")
                print(f"   Relationship types: {len(context['relationship_types'])}")
                
                print(f"✅ PASSED: Entity context retrieval successful")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Entity Context',
                    'status': 'PASSED',
                    'entity': sample_entity,
                    'articles_count': context['articles_count'],
                    'entities_count': context['entities_count']
                })
            else:
                print(f"❌ FAILED: {context['error']}")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Entity Context',
                    'status': 'FAILED',
                    'error': context['error']
                })
        else:
            print(f"⚠️ SKIPPED: No entities available for context test")
            results['test_details'].append({
                'test': 'Entity Context',
                'status': 'SKIPPED',
                'reason': 'No entities available'
            })
        
        # Test 6: Error Handling
        print(f"\n📋 TEST 6: Error Handling")
        print("-" * 40)
        
        # Test with non-existent entity
        fake_result = traversal.multi_hop_traversal("NonExistentEntity12345", max_hops=1)
        
        if fake_result['entities_count'] == 0:
            print(f"✅ PASSED: Properly handles non-existent entities")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Error Handling',
                'status': 'PASSED',
                'validation': 'Handles non-existent entities'
            })
        else:
            print(f"❌ FAILED: Should return 0 entities for fake entity")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Error Handling',
                'status': 'FAILED',
                'error': 'Does not handle non-existent entities properly'
            })
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        results['tests_failed'] += 1
        results['test_details'].append({
            'test': 'Overall Execution',
            'status': 'FAILED',
            'error': str(e)
        })
    
    # Calculate performance metrics
    total_time = time.time() - start_time
    results['performance']['total_test_time'] = total_time
    
    # Final summary
    total_tests = results['tests_passed'] + results['tests_failed']
    success_rate = (results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 GRAPH TRAVERSAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {results['tests_passed']}")
    print(f"❌ Tests Failed: {results['tests_failed']}")
    print(f"📊 Success Rate: {success_rate:.1f}%")
    print(f"⏱️  Total Test Time: {total_time:.2f}s")
    
    if results['graph_stats']:
        print(f"\n📈 Graph Statistics:")
        print(f"   Nodes: {results['graph_stats']['nodes']:,}")
        print(f"   Edges: {results['graph_stats']['edges']:,}")
    
    results['success_rate'] = success_rate
    results['summary'] = {
        'status': 'PASSED' if success_rate >= 80 else 'FAILED',
        'total_tests': total_tests,
        'success_rate': success_rate
    }
    
    return results


if __name__ == "__main__":
    print("🚀 Starting Phase 6 Script 2 validation...")
    print()
    
    # Run comprehensive test
    results = test_end_to_end_graph_traversal()
    
    print(f"\n{'='*60}")
    if results['summary']['success_rate'] >= 80:
        print("🎉 SCRIPT 2 (GRAPH TRAVERSAL) VALIDATION: ✅ PASSED")
        print("Ready to proceed to Script 3 (QueryExpander)")
    else:
        print("❌ SCRIPT 2 (GRAPH TRAVERSAL) VALIDATION: FAILED") 
        print("Please fix issues before proceeding to Script 3")
    
    print(f"{'='*60}")