#!/usr/bin/env python3
"""
Phase 6 - Script 6 Testing: ResponseGenerator Integration

This script validates the ResponseGenerator functionality by integrating it
with the KG-Enhanced RAG system to produce comprehensive enhanced responses.
"""

import sys
import os
from typing import Dict, List, Any
import time

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retrieval.kg_enhanced_rag import KGEnhancedRAG
from retrieval.response_generator import ResponseGenerator


def test_integrated_kg_rag_with_response_generator() -> Dict[str, Any]:
    """
    Test the complete pipeline: KG-Enhanced RAG → ResponseGenerator → Enhanced Response
    """
    
    print("🧪 PHASE 6 SCRIPT 6: INTEGRATED KG-RAG + RESPONSE GENERATOR")
    print("=" * 70)
    
    results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'test_details': [],
        'performance': {}
    }
    
    start_time = time.time()
    
    try:
        # Test 1: Initialize both systems
        print("📋 TEST 1: System Initialization")
        print("-" * 40)
        
        print("🚀 Initializing KG-Enhanced RAG...")
        kg_rag = KGEnhancedRAG()
        
        print("🎨 Initializing ResponseGenerator...")
        response_generator = ResponseGenerator()
        
        print(f"✅ PASSED: Both systems initialized successfully")
        results['tests_passed'] += 1
        results['test_details'].append({
            'test': 'System Initialization',
            'status': 'PASSED',
            'details': 'KG-Enhanced RAG and ResponseGenerator initialized'
        })
        
        # Test 2: End-to-End Processing with Entity-Rich Query
        print(f"\n📋 TEST 2: End-to-End Processing (Entity-Rich Query)")
        print("-" * 40)
        
        entity_query = "What role does Microsoft play in the technology sector?"
        
        print(f"Testing query: '{entity_query}'")
        
        # Step 1: Get KG-Enhanced RAG result
        print("🔍 Step 1: Running KG-Enhanced RAG...")
        kg_rag_result = kg_rag.generate_answer(entity_query, include_explanation=True)
        
        if not kg_rag_result.get('error'):
            print(f"   ✅ KG-RAG completed successfully")
            print(f"   Response time: {kg_rag_result['response_time']:.2f}s")
            print(f"   Sources: {len(kg_rag_result['sources'])}")
            print(f"   Graph enhanced: {kg_rag_result.get('graph_enhancement', {}).get('enhancement_applied', False)}")
            
            # Step 2: Enhance with ResponseGenerator
            print("🎨 Step 2: Enhancing with ResponseGenerator...")
            enhanced_result = response_generator.enhance_response(kg_rag_result)
            
            if enhanced_result.get('enhancement_applied'):
                print(f"   ✅ Response enhancement completed")
                
                # Validate enhanced response structure
                required_components = ['reasoning_path', 'confidence_indicators', 'visual_path', 'enhanced_answer']
                missing_components = [comp for comp in required_components if comp not in enhanced_result]
                
                if not missing_components:
                    reasoning_steps = len(enhanced_result.get('reasoning_path', {}).get('steps', []))
                    overall_confidence = enhanced_result.get('confidence_indicators', {}).get('overall_confidence', 0)
                    
                    print(f"   Reasoning steps generated: {reasoning_steps}")
                    print(f"   Overall confidence: {overall_confidence:.2f}")
                    
                    print(f"✅ PASSED: End-to-end processing successful")
                    results['tests_passed'] += 1
                    results['test_details'].append({
                        'test': 'End-to-End Processing (Entity-Rich Query)',
                        'status': 'PASSED',
                        'details': f'{reasoning_steps} reasoning steps, {overall_confidence:.2f} confidence'
                    })
                else:
                    print(f"❌ FAILED: Missing enhanced components: {missing_components}")
                    results['tests_failed'] += 1
                    results['test_details'].append({
                        'test': 'End-to-End Processing (Entity-Rich Query)',
                        'status': 'FAILED',
                        'details': f'Missing components: {missing_components}'
                    })
            else:
                print(f"❌ FAILED: Response enhancement not applied")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'End-to-End Processing (Entity-Rich Query)',
                    'status': 'FAILED',
                    'details': 'Response enhancement not applied'
                })
        else:
            print(f"❌ FAILED: KG-RAG processing error: {kg_rag_result.get('error')}")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'End-to-End Processing (Entity-Rich Query)',
                'status': 'FAILED',
                'details': f"KG-RAG error: {kg_rag_result.get('error')}"
            })
        
        # Test 3: Reasoning Path Quality
        print(f"\n📋 TEST 3: Reasoning Path Quality Assessment")
        print("-" * 40)
        
        if 'enhanced_result' in locals() and enhanced_result.get('reasoning_path'):
            reasoning_path = enhanced_result['reasoning_path']
            steps = reasoning_path.get('steps', [])
            
            print(f"📝 Analyzing {len(steps)} reasoning steps:")
            
            step_quality_scores = []
            for i, step in enumerate(steps, 1):
                description = step.get('description', '')
                details = step.get('details', '')
                confidence = step.get('confidence', 0)
                
                print(f"   Step {i}: {description}")
                print(f"           Confidence: {confidence:.2f}")
                print(f"           Detail length: {len(details)} chars")
                
                # Quality assessment criteria
                has_description = len(description) > 10
                has_details = len(details) > 20
                reasonable_confidence = 0.1 <= confidence <= 1.0
                
                step_quality = sum([has_description, has_details, reasonable_confidence]) / 3.0
                step_quality_scores.append(step_quality)
            
            avg_step_quality = sum(step_quality_scores) / len(step_quality_scores) if step_quality_scores else 0
            
            if avg_step_quality >= 0.8 and len(steps) >= 3:
                print(f"✅ PASSED: High-quality reasoning path (avg quality: {avg_step_quality:.2f})")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Reasoning Path Quality Assessment',
                    'status': 'PASSED',
                    'details': f'{len(steps)} steps, {avg_step_quality:.2f} avg quality'
                })
            else:
                print(f"❌ FAILED: Reasoning path quality insufficient (avg quality: {avg_step_quality:.2f})")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Reasoning Path Quality Assessment',
                    'status': 'FAILED',
                    'details': f'{len(steps)} steps, {avg_step_quality:.2f} avg quality'
                })
        else:
            print(f"❌ FAILED: No reasoning path available for assessment")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Reasoning Path Quality Assessment',
                'status': 'FAILED',
                'details': 'No reasoning path available'
            })
        
        # Test 4: Confidence Indicator Accuracy
        print(f"\n📋 TEST 4: Confidence Indicator Validation")
        print("-" * 40)
        
        if 'enhanced_result' in locals() and enhanced_result.get('confidence_indicators'):
            confidence_indicators = enhanced_result['confidence_indicators']
            
            overall_confidence = confidence_indicators.get('overall_confidence', 0)
            confidence_level = confidence_indicators.get('confidence_level', '')
            detailed_confidence = confidence_indicators.get('detailed_confidence', {})
            
            print(f"   Overall confidence: {overall_confidence:.2f}")
            print(f"   Confidence level: {confidence_level}")
            
            # Validate confidence components
            confidence_components = ['source_confidence', 'graph_confidence', 'entity_confidence']
            valid_components = []
            
            for component in confidence_components:
                value = detailed_confidence.get(component, -1)
                is_valid = 0 <= value <= 1.0
                valid_components.append(is_valid)
                print(f"   {component}: {value:.2f} {'✓' if is_valid else '✗'}")
            
            # Check confidence level consistency
            level_consistent = (
                (overall_confidence >= 0.75 and confidence_level in ['high', 'very_high']) or
                (0.6 <= overall_confidence < 0.75 and confidence_level == 'moderate') or
                (overall_confidence < 0.6 and confidence_level in ['low', 'very_low'])
            )
            
            if all(valid_components) and level_consistent:
                print(f"✅ PASSED: Confidence indicators are valid and consistent")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Confidence Indicator Validation',
                    'status': 'PASSED',
                    'details': f'Overall: {overall_confidence:.2f}, Level: {confidence_level}'
                })
            else:
                issues = []
                if not all(valid_components):
                    issues.append('invalid component values')
                if not level_consistent:
                    issues.append('inconsistent confidence level')
                
                print(f"❌ FAILED: Confidence indicator issues: {', '.join(issues)}")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Confidence Indicator Validation',
                    'status': 'FAILED',
                    'details': f'Issues: {", ".join(issues)}'
                })
        else:
            print(f"❌ FAILED: No confidence indicators available")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Confidence Indicator Validation',
                'status': 'FAILED',
                'details': 'No confidence indicators available'
            })
        
        # Test 5: Visual Path Generation
        print(f"\n📋 TEST 5: Visual Path Generation")
        print("-" * 40)
        
        if 'enhanced_result' in locals() and enhanced_result.get('visual_path'):
            visual_path = enhanced_result['visual_path']
            
            text_repr = visual_path.get('text_representation', '')
            ascii_graph = visual_path.get('ascii_graph', '')
            json_structure = visual_path.get('json_structure', {})
            
            print(f"   Text representation: {text_repr[:80]}...")
            print(f"   ASCII graph lines: {len(ascii_graph.split(chr(10))) if ascii_graph else 0}")
            print(f"   JSON nodes: {len(json_structure.get('nodes', []))}")
            print(f"   JSON edges: {len(json_structure.get('edges', []))}")
            
            # Validate visual path components
            has_text = len(text_repr) > 10
            has_ascii = len(ascii_graph) > 50
            has_json_nodes = len(json_structure.get('nodes', [])) >= 3
            has_json_edges = len(json_structure.get('edges', [])) >= 2
            
            visual_quality = sum([has_text, has_ascii, has_json_nodes, has_json_edges]) / 4.0
            
            if visual_quality >= 0.75:
                print(f"✅ PASSED: Visual path components generated successfully")
                results['tests_passed'] += 1
                results['test_details'].append({
                    'test': 'Visual Path Generation',
                    'status': 'PASSED',
                    'details': f'Quality score: {visual_quality:.2f}'
                })
            else:
                print(f"❌ FAILED: Visual path generation incomplete (quality: {visual_quality:.2f})")
                results['tests_failed'] += 1
                results['test_details'].append({
                    'test': 'Visual Path Generation',
                    'status': 'FAILED',
                    'details': f'Quality score: {visual_quality:.2f}'
                })
        else:
            print(f"❌ FAILED: No visual path available")
            results['tests_failed'] += 1
            results['test_details'].append({
                'test': 'Visual Path Generation',
                'status': 'FAILED',
                'details': 'No visual path available'
            })
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        
        if 'enhanced_result' in locals():
            kg_rag_time = kg_rag_result.get('response_time', 0)
            enhancement_time = enhanced_result.get('processing_metadata', {}).get('enhancement_time', 0)
            
            results['performance'] = {
                'total_pipeline_time': total_time,
                'kg_rag_time': kg_rag_time,
                'enhancement_time': enhancement_time,
                'enhancement_overhead': enhancement_time / kg_rag_time if kg_rag_time > 0 else 0
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
    print("🚀 Starting Phase 6 Script 6 integration validation...")
    print()
    
    # Run comprehensive integration test
    results = test_integrated_kg_rag_with_response_generator()
    
    print(f"\n{'='*70}")
    print("📊 RESPONSE GENERATOR INTEGRATION VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    summary = results['summary']
    
    print(f"📋 Tests Summary:")
    print(f"   Total tests: {summary['total_tests']}")
    print(f"   Tests passed: {summary['tests_passed']}")
    print(f"   Tests failed: {summary['tests_failed']}")
    print(f"   Success rate: {summary['success_rate']:.1f}%")
    print(f"   Execution time: {summary['execution_time']:.1f}s")
    
    # Performance metrics
    if 'performance' in results and results['performance']:
        perf = results['performance']
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Total pipeline time: {perf.get('total_pipeline_time', 0):.2f}s")
        print(f"   KG-RAG time: {perf.get('kg_rag_time', 0):.2f}s")
        print(f"   Enhancement time: {perf.get('enhancement_time', 0):.2f}s")
        print(f"   Enhancement overhead: {perf.get('enhancement_overhead', 0):.1%}")
    
    print(f"\n📝 Test Details:")
    for i, test_detail in enumerate(results['test_details'], 1):
        status_icon = "✅" if test_detail['status'] == 'PASSED' else "⚠️" if test_detail['status'] == 'PARTIAL' else "❌"
        print(f"   {i}. {status_icon} {test_detail['test']}: {test_detail['status']}")
        if test_detail['details']:
            print(f"      {test_detail['details']}")
    
    if summary['success_rate'] >= 80:
        print(f"\n🎉 SCRIPT 6 (RESPONSE GENERATOR) VALIDATION: ✅ PASSED")
        print("Ready to proceed to Script 7 (KGRAGEvaluator)")
    else:
        print(f"\n❌ SCRIPT 6 (RESPONSE GENERATOR) VALIDATION: FAILED") 
        print("Please fix issues before proceeding to Script 7")
    
    print(f"{'='*70}")