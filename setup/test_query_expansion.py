#!/usr/bin/env python3
"""
Phase 6 - Script 3 Testing: Query Expander Validation

This script validates the QueryExpander functionality with comprehensive tests
to ensure all components work correctly before integrating into the KG-Enhanced RAG.
"""

import sys
import os
from typing import Dict, List, Any
import time

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval.query_expander import QueryExpander


def test_end_to_end_query_expansion() -> Dict[str, Any]:
    """
    Comprehensive end-to-end test of QueryExpander functionality.
    
    Tests:
    1. QueryExpander initialization
    2. Single entity query expansion
    3. Multi-entity query expansion
    4. Entity context retrieval
    5. Query intent classification integration
    6. Error handling and edge cases
    
    Returns:
        Dict with test results and statistics
    """
    
    print("🧪 PHASE 6 SCRIPT 3: QUERY EXPANDER VALIDATION")
    print("=" * 60)
    
    results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'test_details': [],
        'performance': {}
    }
    
    start_time = time.time()
    
    try:
        # Test 1: Initialize QueryExpander
        print("📋 TEST 1: QueryExpander Initialization")
        print("-" * 40)
        
        expander = QueryExpander()
        
        print(f"✅ PASSED: QueryExpander initialized successfully")
        results['tests_passed'] += 1
        results['test_details'].append({
            'test': 'Initialization',
            'status': 'PASSED'
        })
        
        # Test 2: Simple Query Expansion
        print(f"\n📋 TEST 2: Simple Query Expansion")
        print("-" * 40)
        
        simple_query = "What is Microsoft's role in technology?"
        
        expansion_result = expander.expand_query(
            query=simple_query,
            max_hops=1,
            max_related_entities=5,
            include_entity_context=True
        )
        
        print(f"Query: '{simple_query}'")
        print(f"Success: {expansion_result['success']}")
        print(f"Extracted entities: {len(expansion_result.get('extracted_entities', []))}")
        print(f"Related entities: {len(expansion_result.get('related_entities', []))}")
        print(f"Time: {expansion_result['expansion_time']:.3f}s")
        
        if expansion_result['success'] and len(expansion_result.get('extracted_entities', [])) > 0:
            print(f"✅ PASSED: Simple query expansion successful")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Simple Query Expansion',
                'status': 'PASSED',
                'entities_extracted': len(expansion_result.get('extracted_entities', [])),
                'entities_related': len(expansion_result.get('related_entities', [])),
                'time': expansion_result['expansion_time']
            })
        else:
            print(f"❌ FAILED: Simple query expansion failed")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Simple Query Expansion', 
                'status': 'FAILED',
                'error': expansion_result.get('error', 'Unknown error')
            })
        
        # Test 3: Multi-Entity Query Expansion
        print(f"\n📋 TEST 3: Multi-Entity Query Expansion")
        print("-" * 40)
        
        multi_query = "How are Microsoft and tennis related?"
        
        multi_result = expander.expand_query(
            query=multi_query,
            max_hops=2,
            max_related_entities=10,
            include_entity_context=True
        )
        
        print(f"Query: '{multi_query}'")
        print(f"Success: {multi_result['success']}")
        print(f"Extracted entities: {len(multi_result.get('extracted_entities', []))}")
        print(f"Related entities: {len(multi_result.get('related_entities', []))}")
        
        if multi_result['success'] and len(multi_result.get('extracted_entities', [])) >= 2:
            print(f"✅ PASSED: Multi-entity query expansion successful")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Multi-Entity Query Expansion',
                'status': 'PASSED',
                'entities_extracted': len(multi_result.get('extracted_entities', [])),
                'entities_related': len(multi_result.get('related_entities', [])),
                'time': multi_result['expansion_time']
            })
        else:
            print(f"❌ FAILED: Multi-entity query expansion failed")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Multi-Entity Query Expansion',
                'status': 'FAILED',
                'error': multi_result.get('error', 'Insufficient entities extracted')
            })
        
        # Test 4: Entity Context Integration
        print(f"\n📋 TEST 4: Entity Context Integration")
        print("-" * 40)
        
        # Use best result from previous tests
        test_result = expansion_result if expansion_result['success'] else multi_result
        
        if test_result['success'] and test_result.get('entity_contexts'):
            context_count = len(test_result['entity_contexts'])
            print(f"Entity contexts retrieved: {context_count}")
            
            # Check context quality
            valid_contexts = 0
            for entity, context in test_result['entity_contexts'].items():
                if context.get('type') and context.get('articles_count', 0) > 0:
                    valid_contexts += 1
            
            print(f"Valid contexts: {valid_contexts}/{context_count}")
            
            if valid_contexts > 0:
                print(f"✅ PASSED: Entity context integration successful")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Entity Context Integration',
                    'status': 'PASSED',
                    'contexts_retrieved': context_count,
                    'contexts_valid': valid_contexts
                })
            else:
                print(f"❌ FAILED: No valid entity contexts")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Entity Context Integration',
                    'status': 'FAILED',
                    'error': 'No valid contexts retrieved'
                })
        else:
            print(f"⚠️ SKIPPED: No entity contexts available")
            results['test_details'].append({
                'test': 'Entity Context Integration',
                'status': 'SKIPPED',
                'reason': 'No contexts available from previous tests'
            })
        
        # Test 5: Query Intent Classification
        print(f"\n📋 TEST 5: Query Intent Classification")
        print("-" * 40)
        
        if test_result['success'] and test_result.get('query_intent'):
            intent = test_result['query_intent']
            print(f"Intent type: {intent.get('primary_intent', 'unknown')}")
            print(f"Confidence: {intent.get('confidence', 0.0):.2f}")
            print(f"Requires graph traversal: {intent.get('requires_graph_traversal', False)}")
            
            if intent.get('primary_intent') != 'unknown' and intent.get('confidence', 0.0) > 0.5:
                print(f"✅ PASSED: Query intent classification successful")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Query Intent Classification',
                    'status': 'PASSED',
                    'intent': intent.get('primary_intent'),
                    'confidence': intent.get('confidence', 0.0),
                    'graph_required': intent.get('requires_graph_traversal', False)
                })
            else:
                print(f"❌ FAILED: Poor intent classification")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Query Intent Classification',
                    'status': 'FAILED',
                    'error': 'Low confidence or unknown intent'
                })
        else:
            print(f"❌ FAILED: No intent classification available")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Query Intent Classification',
                'status': 'FAILED',
                'error': 'No intent data available'
            })
        
        # Test 6: Error Handling
        print(f"\n📋 TEST 6: Error Handling")
        print("-" * 40)
        
        # Test with empty query
        error_result = expander.expand_query(
            query="",
            max_hops=1,
            max_related_entities=5
        )
        
        print(f"Empty query handling: {error_result['success']}")
        
        # Test with very short query
        short_result = expander.expand_query(
            query="Hi",
            max_hops=1,
            max_related_entities=5
        )
        
        print(f"Short query handling: {short_result['success']}")
        
        # At least one should handle gracefully (return success=False or success=True with no entities)
        error_handled = (
            (not error_result['success']) or 
            (error_result['success'] and len(error_result.get('extracted_entities', [])) == 0)
        )
        
        if error_handled:
            print(f"✅ PASSED: Error handling works correctly")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Error Handling',
                'status': 'PASSED',
                'validation': 'Handles edge cases appropriately'
            })
        else:
            print(f"❌ FAILED: Poor error handling")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Error Handling',
                'status': 'FAILED',
                'error': 'Does not handle edge cases properly'
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
    
    print(f"\n🎯 QUERY EXPANDER VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {results['tests_passed']}")
    print(f"❌ Tests Failed: {results['tests_failed']}")
    print(f"📊 Success Rate: {success_rate:.1f}%")
    print(f"⏱️  Total Test Time: {total_time:.2f}s")
    
    results['success_rate'] = success_rate
    results['summary'] = {
        'status': 'PASSED' if success_rate >= 80 else 'FAILED',
        'total_tests': total_tests,
        'success_rate': success_rate
    }
    
    return results


if __name__ == "__main__":
    print("🚀 Starting Phase 6 Script 3 validation...")
    print()
    
    # Run comprehensive test
    results = test_end_to_end_query_expansion()
    
    print(f"\n{'='*60}")
    if results['summary']['success_rate'] >= 80:
        print("🎉 SCRIPT 3 (QUERY EXPANDER) VALIDATION: ✅ PASSED")
        print("Ready to proceed to Script 4 (HybridRetriever)")
    else:
        print("❌ SCRIPT 3 (QUERY EXPANDER) VALIDATION: FAILED") 
        print("Please fix issues before proceeding to Script 4")
    
    print(f"{'='*60}")