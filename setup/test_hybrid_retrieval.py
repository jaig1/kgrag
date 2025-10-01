#!/usr/bin/env python3
"""
Phase 6 - Script 4 Testing: Hybrid Retriever Validation

This script validates the HybridRetriever functionality with comprehensive tests
to ensure all components work correctly before integrating into the KG-Enhanced RAG.
"""

import sys
import os
from typing import Dict, List, Any
import time

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval.hybrid_retriever import HybridRetriever


def test_end_to_end_hybrid_retrieval() -> Dict[str, Any]:
    """
    Comprehensive end-to-end test of HybridRetriever functionality.
    
    Tests:
    1. HybridRetriever initialization
    2. Vector-only retrieval (no expansion)
    3. Hybrid retrieval with graph expansion
    4. Reciprocal Rank Fusion effectiveness
    5. Result quality and diversity
    6. Error handling and edge cases
    
    Returns:
        Dict with test results and statistics
    """
    
    print("🧪 PHASE 6 SCRIPT 4: HYBRID RETRIEVER VALIDATION")
    print("=" * 60)
    
    results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'test_details': [],
        'performance': {}
    }
    
    start_time = time.time()
    
    try:
        # Test 1: Initialize HybridRetriever
        print("📋 TEST 1: HybridRetriever Initialization")
        print("-" * 40)
        
        retriever = HybridRetriever()
        
        print(f"✅ PASSED: HybridRetriever initialized successfully")
        results['tests_passed'] += 1
        results['test_details'].append({
            'test': 'Initialization',
            'status': 'PASSED'
        })
        
        # Test 2: Vector-Only Retrieval
        print(f"\n📋 TEST 2: Vector-Only Retrieval")
        print("-" * 40)
        
        vector_query = "What is Microsoft doing?"
        
        vector_result = retriever.retrieve(
            query=vector_query,
            enable_expansion=False
        )
        
        print(f"Query: '{vector_query}'")
        print(f"Success: {vector_result['success']}")
        print(f"Results: {len(vector_result.get('fused_results', []))}")
        print(f"Time: {vector_result['retrieval_metadata'].get('retrieval_time', 0):.3f}s")
        
        if (vector_result['success'] and 
            len(vector_result.get('original_results', [])) > 0 and
            len(vector_result.get('expanded_results', [])) == 0):  # Should be empty when disabled
            
            print(f"✅ PASSED: Vector-only retrieval successful")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Vector-Only Retrieval',
                'status': 'PASSED',
                'results_count': len(vector_result.get('fused_results', [])),
                'time': vector_result['retrieval_metadata'].get('retrieval_time', 0)
            })
        else:
            print(f"❌ FAILED: Vector-only retrieval failed")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Vector-Only Retrieval',
                'status': 'FAILED',
                'error': 'Failed to retrieve or unexpected expansion'
            })
        
        # Test 3: Hybrid Retrieval with Expansion
        print(f"\n📋 TEST 3: Hybrid Retrieval with Expansion")
        print("-" * 40)
        
        hybrid_query = "How does Microsoft relate to technology?"
        
        hybrid_result = retriever.retrieve(
            query=hybrid_query,
            enable_expansion=True,
            expansion_params={
                'max_hops': 2,
                'max_related_entities': 5,
                'include_entity_context': True
            }
        )
        
        print(f"Query: '{hybrid_query}'")
        print(f"Success: {hybrid_result['success']}")
        print(f"Original results: {len(hybrid_result.get('original_results', []))}")
        print(f"Expanded results: {len(hybrid_result.get('expanded_results', []))}")
        print(f"Final results: {len(hybrid_result.get('fused_results', []))}")
        
        metadata = hybrid_result.get('retrieval_metadata', {})
        expansion_success = metadata.get('expansion_success', False)
        entities_found = metadata.get('entities_found', 0)
        
        if (hybrid_result['success'] and 
            expansion_success and
            entities_found > 0 and
            len(hybrid_result.get('fused_results', [])) > 0):
            
            print(f"✅ PASSED: Hybrid retrieval with expansion successful")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Hybrid Retrieval with Expansion',
                'status': 'PASSED',
                'original_results': len(hybrid_result.get('original_results', [])),
                'expanded_results': len(hybrid_result.get('expanded_results', [])),
                'final_results': len(hybrid_result.get('fused_results', [])),
                'entities_found': entities_found,
                'time': metadata.get('retrieval_time', 0)
            })
        else:
            print(f"❌ FAILED: Hybrid retrieval with expansion failed")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Hybrid Retrieval with Expansion',
                'status': 'FAILED',
                'error': f'Expansion success: {expansion_success}, Entities: {entities_found}'
            })
        
        # Test 4: Reciprocal Rank Fusion Effectiveness
        print(f"\n📋 TEST 4: Reciprocal Rank Fusion Analysis")
        print("-" * 40)
        
        if hybrid_result['success']:
            original_count = len(hybrid_result.get('original_results', []))
            expanded_count = len(hybrid_result.get('expanded_results', []))
            fused_count = len(hybrid_result.get('fused_results', []))
            unique_fused = metadata.get('fused_count', 0)
            
            print(f"Original results: {original_count}")
            print(f"Expanded results: {expanded_count}")
            print(f"Total available: {original_count + expanded_count}")
            print(f"Unique after RRF: {unique_fused}")
            print(f"Final returned: {fused_count}")
            
            # Check if RRF is working (should deduplicate and rerank)
            rrf_effective = (
                unique_fused <= (original_count + expanded_count) and  # Deduplication
                fused_count <= unique_fused  # Proper limiting
            )
            
            # Check for source diversity in top results
            sources_in_top3 = set()
            for result in hybrid_result.get('fused_results', [])[:3]:
                sources_in_top3.update(result.get('fusion_sources', []))
            
            source_diversity = len(sources_in_top3) > 1  # Should have both original and expanded
            
            print(f"RRF deduplication working: {rrf_effective}")
            print(f"Source diversity in top 3: {source_diversity} ({sources_in_top3})")
            
            if rrf_effective:
                print(f"✅ PASSED: Reciprocal Rank Fusion working effectively")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Reciprocal Rank Fusion',
                    'status': 'PASSED',
                    'deduplication_works': True,
                    'source_diversity': source_diversity,
                    'unique_fused': unique_fused
                })
            else:
                print(f"❌ FAILED: RRF not working effectively")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Reciprocal Rank Fusion',
                    'status': 'FAILED',
                    'error': 'RRF deduplication or limiting not working'
                })
        else:
            print(f"⚠️ SKIPPED: No hybrid results to analyze")
            results['test_details'].append({
                'test': 'Reciprocal Rank Fusion',
                'status': 'SKIPPED',
                'reason': 'No hybrid results available'
            })
        
        # Test 5: Result Quality and Content
        print(f"\n📋 TEST 5: Result Quality Assessment")
        print("-" * 40)
        
        if hybrid_result['success'] and hybrid_result.get('fused_results'):
            top_results = hybrid_result['fused_results'][:3]
            
            # Check result completeness
            complete_results = 0
            for result in top_results:
                if (result.get('content') and 
                    result.get('rrf_score', 0) > 0 and
                    result.get('article_id')):
                    complete_results += 1
            
            quality_score = complete_results / len(top_results) if top_results else 0
            
            print(f"Top 3 results analyzed")
            print(f"Complete results: {complete_results}/{len(top_results)}")
            print(f"Quality score: {quality_score:.2f}")
            
            # Check RRF score distribution (should be decreasing)
            scores = [r.get('rrf_score', 0) for r in top_results]
            scores_decreasing = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            
            print(f"RRF scores decreasing: {scores_decreasing}")
            print(f"Score range: {max(scores):.4f} to {min(scores):.4f}")
            
            if quality_score >= 0.8 and scores_decreasing:
                print(f"✅ PASSED: Result quality assessment successful")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Result Quality',
                    'status': 'PASSED',
                    'quality_score': quality_score,
                    'scores_decreasing': scores_decreasing,
                    'complete_results': complete_results
                })
            else:
                print(f"❌ FAILED: Result quality issues")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Result Quality',
                    'status': 'FAILED',
                    'error': f'Quality: {quality_score:.2f}, Scores decreasing: {scores_decreasing}'
                })
        else:
            print(f"❌ FAILED: No results to assess quality")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Result Quality',
                'status': 'FAILED',
                'error': 'No results available for quality assessment'
            })
        
        # Test 6: Error Handling
        print(f"\n📋 TEST 6: Error Handling")
        print("-" * 40)
        
        # Test with empty query
        empty_result = retriever.retrieve(query="", enable_expansion=True)
        print(f"Empty query handled: {empty_result['success']}")
        
        # Should either succeed with no results or fail gracefully
        empty_handled = (
            not empty_result['success'] or  # Graceful failure
            (empty_result['success'] and len(empty_result.get('fused_results', [])) == 0)  # No results
        )
        
        if empty_handled:
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
    
    print(f"\n🎯 HYBRID RETRIEVER VALIDATION SUMMARY")
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
    print("🚀 Starting Phase 6 Script 4 validation...")
    print()
    
    # Run comprehensive test
    results = test_end_to_end_hybrid_retrieval()
    
    print(f"\n{'='*60}")
    if results['summary']['success_rate'] >= 80:
        print("🎉 SCRIPT 4 (HYBRID RETRIEVER) VALIDATION: ✅ PASSED")
        print("Ready to proceed to Script 5 (KGEnhancedRAG)")
    else:
        print("❌ SCRIPT 4 (HYBRID RETRIEVER) VALIDATION: FAILED") 
        print("Please fix issues before proceeding to Script 5")
    
    print(f"{'='*60}")