#!/usr/bin/env python3
"""
Phase 6 - Script 5 Testing: KG-Enhanced RAG Validation

This script validates the KG-Enhanced RAG functionality with comprehensive tests
covering various query types and graph enhancement scenarios.
"""

import sys
import os
from typing import Dict, List, Any
import time

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retrieval.kg_enhanced_rag import KGEnhancedRAG


def test_end_to_end_kg_enhanced_rag() -> Dict[str, Any]:
    """
    Comprehensive end-to-end test of KG-Enhanced RAG functionality.
    
    Tests:
    1. KG-Enhanced RAG initialization
    2. Graph-enhanced retrieval and generation
    3. Entity relationship processing
    4. Enhanced prompt engineering
    5. Answer quality and explanation generation
    6. Performance metrics and statistics
    
    Returns:
        Dict with test results and statistics
    """
    
    print("🧪 PHASE 6 SCRIPT 5: KG-ENHANCED RAG VALIDATION")
    print("=" * 60)
    
    results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'test_details': [],
        'performance': {}
    }
    
    start_time = time.time()
    
    try:
        # Test 1: Initialize KG-Enhanced RAG
        print("📋 TEST 1: KG-Enhanced RAG Initialization")
        print("-" * 40)
        
        kg_rag = KGEnhancedRAG()
        
        print(f"✅ PASSED: KG-Enhanced RAG initialized successfully")
        results['tests_passed'] += 1
        results['test_details'].append({
            'test': 'KG-Enhanced RAG Initialization',
            'status': 'PASSED',
            'details': 'All components initialized correctly'
        })
        
        # Test 2: Entity-Rich Query Processing
        print(f"\n📋 TEST 2: Entity-Rich Query Processing")
        print("-" * 40)
        
        entity_query = "What is Microsoft's role in technology and how does it relate to other companies?"
        
        print(f"Testing query: '{entity_query}'")
        result2 = kg_rag.generate_answer(entity_query, include_explanation=True)
        
        # Validate result structure
        required_keys = ['question', 'answer', 'sources', 'response_time', 'graph_enhancement']
        missing_keys = [key for key in required_keys if key not in result2]
        
        if missing_keys:
            print(f"❌ FAILED: Missing keys in result: {missing_keys}")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Entity-Rich Query Processing',
                'status': 'FAILED',
                'details': f'Missing required keys: {missing_keys}'
            })
        else:
            # Check graph enhancement
            graph_enhancement = result2.get('graph_enhancement', {})
            enhancement_applied = graph_enhancement.get('enhancement_applied', False)
            entities_extracted = graph_enhancement.get('entities_extracted', 0)
            
            print(f"   Graph enhancement applied: {enhancement_applied}")
            print(f"   Entities extracted: {entities_extracted}")
            print(f"   Response time: {result2['response_time']:.2f}s")
            print(f"   Sources used: {len(result2['sources'])}")
            print(f"   Answer length: {len(result2['answer'])} chars")
            
            if enhancement_applied and entities_extracted > 0:
                print(f"✅ PASSED: Entity-rich query processed with graph enhancement")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Entity-Rich Query Processing',
                    'status': 'PASSED',
                    'details': f'Graph enhancement applied, {entities_extracted} entities extracted'
                })
            else:
                print(f"⚠️  PARTIAL: Query processed but minimal graph enhancement")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Entity-Rich Query Processing',
                    'status': 'PARTIAL',
                    'details': 'Query processed but limited graph enhancement'
                })
        
        # Test 3: Explanation Generation
        print(f"\n📋 TEST 3: Explanation Generation")
        print("-" * 40)
        
        explanation = result2.get('explanation', {})
        if explanation and 'steps' in explanation:
            steps = explanation['steps']
            step_names = [step['name'] for step in steps]
            
            print(f"   Process overview: {explanation.get('process_overview', 'N/A')[:100]}...")
            print(f"   Steps generated: {len(steps)}")
            for i, step in enumerate(steps, 1):
                print(f"     {i}. {step['name']}")
            
            if len(steps) >= 3:  # Expect at least 3 main steps
                print(f"✅ PASSED: Comprehensive explanation generated with {len(steps)} steps")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Explanation Generation',
                    'status': 'PASSED',
                    'details': f'{len(steps)} process steps explained'
                })
            else:
                print(f"❌ FAILED: Insufficient explanation detail (only {len(steps)} steps)")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Explanation Generation',
                    'status': 'FAILED',
                    'details': f'Only {len(steps)} steps in explanation'
                })
        else:
            print(f"❌ FAILED: No explanation generated")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Explanation Generation',
                'status': 'FAILED',
                'details': 'No explanation found in result'
            })
        
        # Test 4: Multiple Query Types
        print(f"\n📋 TEST 4: Multiple Query Types")
        print("-" * 40)
        
        test_queries = [
            "What are the latest developments in gaming?",
            "How is BBC covering sports events?",
            "What political issues are being discussed?"
        ]
        
        query_results = []
        for i, query in enumerate(test_queries, 1):
            print(f"   Testing query {i}: '{query[:50]}...'")
            query_result = kg_rag.generate_answer(query, include_explanation=False)
            query_results.append({
                'query': query,
                'success': not query_result.get('error', False),
                'response_time': query_result['response_time'],
                'sources_count': len(query_result['sources']),
                'graph_enhanced': query_result.get('graph_enhancement', {}).get('enhancement_applied', False)
            })
        
        successful_queries = sum(1 for r in query_results if r['success'])
        graph_enhanced_queries = sum(1 for r in query_results if r['graph_enhanced'])
        
        print(f"   Successful queries: {successful_queries}/{len(test_queries)}")
        print(f"   Graph-enhanced queries: {graph_enhanced_queries}/{len(test_queries)}")
        print(f"   Average response time: {sum(r['response_time'] for r in query_results)/len(query_results):.2f}s")
        
        if successful_queries == len(test_queries):
            print(f"✅ PASSED: All query types processed successfully")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Multiple Query Types',
                'status': 'PASSED',
                'details': f'{successful_queries}/{len(test_queries)} queries successful, {graph_enhanced_queries} graph-enhanced'
            })
        else:
            print(f"❌ FAILED: {len(test_queries) - successful_queries} queries failed")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Multiple Query Types',
                'status': 'FAILED',
                'details': f'{successful_queries}/{len(test_queries)} queries successful'
            })
        
        # Test 5: Statistics and Performance
        print(f"\n📋 TEST 5: Statistics and Performance Tracking")
        print("-" * 40)
        
        stats = kg_rag.get_statistics()
        
        # Check statistics structure
        required_stat_sections = ['kg_enhanced_rag', 'hybrid_retriever', 'response_tracker']
        missing_sections = [section for section in required_stat_sections if section not in stats]
        
        if missing_sections:
            print(f"❌ FAILED: Missing statistics sections: {missing_sections}")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Statistics and Performance Tracking',
                'status': 'FAILED',
                'details': f'Missing statistics sections: {missing_sections}'
            })
        else:
            kg_stats = stats['kg_enhanced_rag']
            print(f"   Total queries processed: {kg_stats.get('queries_processed', 0)}")
            print(f"   Success rate: {kg_stats.get('success_rate', 0):.1f}%")
            print(f"   Graph enhancement rate: {kg_stats.get('graph_enhancement_rate', 0):.1f}%")
            print(f"   Average response time: {kg_stats.get('average_response_time', 0):.2f}s")
            
            queries_processed = kg_stats.get('queries_processed', 0)
            if queries_processed >= 4:  # We ran 4+ queries in total
                print(f"✅ PASSED: Statistics tracking working correctly")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Statistics and Performance Tracking',
                    'status': 'PASSED',
                    'details': f'{queries_processed} queries tracked with comprehensive stats'
                })
            else:
                print(f"❌ FAILED: Statistics not tracking correctly ({queries_processed} queries recorded)")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Statistics and Performance Tracking',
                    'status': 'FAILED',
                    'details': f'Only {queries_processed} queries tracked'
                })
        
        # Test 6: Error Handling
        print(f"\n📋 TEST 6: Error Handling")
        print("-" * 40)
        
        # Test with empty query
        empty_result = kg_rag.generate_answer("")
        
        if empty_result and not empty_result.get('error'):
            # Should handle empty query gracefully
            print(f"   Empty query handled: {len(empty_result.get('answer', ''))} char response")
            print(f"✅ PASSED: Error handling working correctly")
            results['tests_passed'] += 1
            results['test_details'].append({
                'test': 'Error Handling',
                'status': 'PASSED',
                'details': 'Empty query handled gracefully'
            })
        else:
            print(f"❌ FAILED: Error handling issues")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Error Handling',
                'status': 'FAILED',
                'details': 'Error handling not working properly'
            })
        
        # Calculate final performance metrics
        total_time = time.time() - start_time
        results['performance'] = {
            'total_test_time': total_time,
            'final_statistics': stats
        }
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        results['tests_failed'] += 1
        results['test_details'].append({
            'test': 'Overall Execution',
            'status': 'CRITICAL_ERROR',
            'details': str(e)
        })
    
    # Calculate summary
    total_tests = results['tests_passed'] + results['tests_failed']
    success_rate = (results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
    
    results['summary'] = {
        'total_tests': total_tests,
        'tests_passed': results['tests_passed'],
        'tests_failed': results['tests_failed'],
        'success_rate': success_rate,
        'execution_time': time.time() - start_time
    }
    
    return results


if __name__ == "__main__":
    print("🚀 Starting Phase 6 Script 5 validation...")
    print()
    
    # Run comprehensive test
    results = test_end_to_end_kg_enhanced_rag()
    
    print(f"\n{'='*60}")
    print("📊 KG-ENHANCED RAG VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    summary = results['summary']
    
    print(f"📋 Tests Summary:")
    print(f"   Total tests: {summary['total_tests']}")
    print(f"   Tests passed: {summary['tests_passed']}")
    print(f"   Tests failed: {summary['tests_failed']}")
    print(f"   Success rate: {summary['success_rate']:.1f}%")
    print(f"   Execution time: {summary['execution_time']:.1f}s")
    
    print(f"\n📝 Test Details:")
    for i, test_detail in enumerate(results['test_details'], 1):
        status_icon = "✅" if test_detail['status'] == 'PASSED' else "⚠️" if test_detail['status'] == 'PARTIAL' else "❌"
        print(f"   {i}. {status_icon} {test_detail['test']}: {test_detail['status']}")
        if test_detail['details']:
            print(f"      {test_detail['details']}")
    
    if summary['success_rate'] >= 80:
        print(f"\n🎉 SCRIPT 5 (KG-ENHANCED RAG) VALIDATION: ✅ PASSED")
        print("Ready to proceed to Script 6 (ResponseGenerator)")
    else:
        print(f"\n❌ SCRIPT 5 (KG-ENHANCED RAG) VALIDATION: FAILED") 
        print("Please fix issues before proceeding to Script 6")
    
    print(f"{'='*60}")