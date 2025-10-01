"""
Comprehensive testing script for the Baseline RAG system.

This script provides end-to-end testing with diverse queries covering all
categories and functionality verification for the complete RAG pipeline.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.baseline_rag import BaselineRAG
from src.retrieval.response_tracker import ResponseTracker


class RAGTester:
    """
    Comprehensive testing suite for the Baseline RAG system.
    """
    
    def __init__(self, test_mode: bool = True):
        """
        Initialize the RAG tester.
        
        Args:
            test_mode: Whether to run in test mode with smaller datasets
        """
        self.test_mode = test_mode
        
        # Initialize RAG system with test configuration
        tracker = ResponseTracker(
            log_dir=Path("data/test_logs" if test_mode else "data/logs"),
            log_filename="rag_test_queries.jsonl" if test_mode else "rag_queries.jsonl"
        )
        
        self.rag = BaselineRAG(
            response_tracker=tracker,
            model_name="gpt-3.5-turbo",  # Faster model for demo
            max_sources=5,
            temperature=0.1
        )
        
        print(f"🧪 RAG Tester initialized:")
        print(f"   Test mode: {test_mode}")
        print(f"   Log file: {tracker.log_file}")
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite with diverse queries.
        
        Returns:
            Test results and performance metrics
        """
        print(f"\n🚀 COMPREHENSIVE RAG TEST SUITE")
        print(f"=" * 60)
        
        # Define comprehensive test questions
        test_questions = self._get_test_questions()
        
        print(f"📋 Testing {len(test_questions)} diverse queries...")
        
        results = []
        category_results = {}
        
        # Process each test question
        for i, test_case in enumerate(test_questions, 1):
            question = test_case['question']
            category = test_case['category']
            expected_topics = test_case.get('expected_topics', [])
            
            print(f"\n--- Test {i}/{len(test_questions)}: {category.upper()} ---")
            print(f"Question: {question}")
            
            # Answer the question
            result = self.rag.answer_question(question)
            
            # Analyze result
            analysis = self._analyze_result(result, expected_topics)
            
            # Store results
            test_result = {
                **result,
                'test_metadata': {
                    'test_number': i,
                    'expected_category': category,
                    'expected_topics': expected_topics,
                    'analysis': analysis
                }
            }
            
            results.append(test_result)
            
            # Track by category
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(test_result)
            
            # Print brief result
            print(f"📝 Full Answer:")
            print(f"{result['answer']}")
            print(f"\n📊 Metrics:")
            print(f"   Answer length: {len(result['answer'])} chars")
            print(f"   Sources: {len(result['sources'])}")
            print(f"   Time: {result['response_time']:.2f}s")
            print(f"   Tokens: {result['token_usage']['total_tokens']}")
            
            # Brief pause between questions
            time.sleep(0.5)
        
        # Calculate comprehensive statistics
        test_stats = self._calculate_test_statistics(results, category_results)
        
        # Print results summary
        self._print_test_summary(test_stats)
        
        return {
            'results': results,
            'category_results': category_results,
            'statistics': test_stats,
            'test_mode': self.test_mode
        }
    
    def _get_test_questions(self) -> List[Dict[str, Any]]:
        """Get comprehensive set of test questions covering all categories."""
        return [
            # Technology queries
            {
                'question': 'What major technology companies and their developments are mentioned in the news?',
                'category': 'tech',
                'expected_topics': ['companies', 'developments', 'innovation', 'products']
            },
            {
                'question': 'Are there any artificial intelligence or machine learning breakthroughs discussed?',
                'category': 'tech', 
                'expected_topics': ['AI', 'machine learning', 'algorithms', 'automation']
            },
            
            # Sports queries
            {
                'question': 'What sporting events, competitions, or athletic achievements are covered?',
                'category': 'sport',
                'expected_topics': ['competitions', 'championships', 'athletes', 'teams']
            },
            {
                'question': 'Are there any football or tennis matches and results mentioned?',
                'category': 'sport',
                'expected_topics': ['football', 'tennis', 'matches', 'results', 'scores']
            },
            
            # Business queries  
            {
                'question': 'What business developments, mergers, or economic news are reported?',
                'category': 'business',
                'expected_topics': ['companies', 'mergers', 'acquisitions', 'markets', 'economy']
            },
            {
                'question': 'Are there any financial markets, stock performance, or investment news discussed?',
                'category': 'business',
                'expected_topics': ['markets', 'stocks', 'investments', 'financial', 'earnings']
            },
            
            # Entertainment queries
            {
                'question': 'What entertainment news, celebrity updates, or cultural events are covered?',
                'category': 'entertainment',
                'expected_topics': ['celebrities', 'movies', 'music', 'shows', 'culture']
            },
            {
                'question': 'Are there any movie releases, music albums, or award ceremonies mentioned?',
                'category': 'entertainment',
                'expected_topics': ['movies', 'music', 'albums', 'awards', 'releases']
            },
            
            # Politics queries
            {
                'question': 'What political developments, policy changes, or government news are discussed?',
                'category': 'politics', 
                'expected_topics': ['government', 'policy', 'legislation', 'politics', 'officials']
            },
            {
                'question': 'Are there any international relations, diplomatic meetings, or global affairs covered?',
                'category': 'politics',
                'expected_topics': ['international', 'diplomatic', 'global', 'relations', 'countries']
            }
        ]
    
    def _analyze_result(self, result: Dict[str, Any], expected_topics: List[str]) -> Dict[str, Any]:
        """Analyze a single test result for quality and relevance."""
        answer = result['answer'].lower()
        
        # Check for expected topics in the answer
        topics_found = []
        for topic in expected_topics:
            if topic.lower() in answer:
                topics_found.append(topic)
        
        topic_coverage = len(topics_found) / len(expected_topics) if expected_topics else 0
        
        # Check for source citations
        has_citations = '[source' in answer.lower() or 'source 1' in answer.lower()
        
        # Basic quality checks
        analysis = {
            'answer_length': len(result['answer']),
            'sources_count': len(result['sources']),
            'topics_found': topics_found,
            'topic_coverage': topic_coverage,
            'has_citations': has_citations,
            'response_time_acceptable': result['response_time'] < 10.0,  # Under 10 seconds
            'token_usage_reasonable': result['token_usage']['total_tokens'] < 2000,
            'has_error': bool(result.get('error')),
            'quality_score': self._calculate_quality_score(
                len(result['answer']),
                len(result['sources']),
                topic_coverage,
                has_citations,
                result['response_time'],
                bool(result.get('error'))
            )
        }
        
        return analysis
    
    def _calculate_quality_score(self, 
                               answer_length: int,
                               sources_count: int, 
                               topic_coverage: float,
                               has_citations: bool,
                               response_time: float,
                               has_error: bool) -> float:
        """Calculate a quality score for the answer (0-100)."""
        if has_error:
            return 0.0
        
        score = 0.0
        
        # Answer length (0-20 points)
        if answer_length > 500:
            score += 20
        elif answer_length > 200:
            score += 15
        elif answer_length > 100:
            score += 10
        elif answer_length > 50:
            score += 5
        
        # Sources count (0-20 points)
        if sources_count >= 3:
            score += 20
        elif sources_count >= 2:
            score += 15
        elif sources_count >= 1:
            score += 10
        
        # Topic coverage (0-30 points)
        score += topic_coverage * 30
        
        # Citations (0-15 points)
        if has_citations:
            score += 15
        
        # Response time (0-15 points)
        if response_time < 3.0:
            score += 15
        elif response_time < 5.0:
            score += 10
        elif response_time < 10.0:
            score += 5
        
        return min(100.0, score)
    
    def _calculate_test_statistics(self, 
                                 results: List[Dict[str, Any]], 
                                 category_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate comprehensive test statistics."""
        total_tests = len(results)
        
        # Overall statistics
        successful_tests = [r for r in results if not r.get('error')]
        success_rate = len(successful_tests) / total_tests * 100 if total_tests > 0 else 0
        
        # Performance metrics
        response_times = [r['response_time'] for r in successful_tests]
        token_counts = [r['token_usage']['total_tokens'] for r in successful_tests]
        source_counts = [len(r['sources']) for r in successful_tests]
        quality_scores = [r['test_metadata']['analysis']['quality_score'] for r in successful_tests]
        
        # Category breakdown
        category_stats = {}
        for category, cat_results in category_results.items():
            cat_successful = [r for r in cat_results if not r.get('error')]
            if cat_successful:
                category_stats[category] = {
                    'total_tests': len(cat_results),
                    'successful_tests': len(cat_successful),
                    'success_rate': len(cat_successful) / len(cat_results) * 100,
                    'avg_response_time': sum(r['response_time'] for r in cat_successful) / len(cat_successful),
                    'avg_quality_score': sum(r['test_metadata']['analysis']['quality_score'] for r in cat_successful) / len(cat_successful),
                    'avg_sources': sum(len(r['sources']) for r in cat_successful) / len(cat_successful)
                }
        
        return {
            'overall': {
                'total_tests': total_tests,
                'successful_tests': len(successful_tests),
                'success_rate': success_rate,
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
                'avg_sources': sum(source_counts) / len(source_counts) if source_counts else 0,
                'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0
            },
            'category_breakdown': category_stats
        }
    
    def _print_test_summary(self, stats: Dict[str, Any]) -> None:
        """Print comprehensive test summary."""
        overall = stats['overall']
        
        print(f"\n📊 COMPREHENSIVE TEST RESULTS")
        print(f"=" * 60)
        
        # Overall performance
        print(f"🎯 Overall Performance:")
        print(f"   Tests completed: {overall['successful_tests']}/{overall['total_tests']}")
        print(f"   Success rate: {overall['success_rate']:.1f}%")
        print(f"   Average quality score: {overall['avg_quality_score']:.1f}/100")
        print(f"   Average response time: {overall['avg_response_time']:.2f}s")
        print(f"   Average sources per answer: {overall['avg_sources']:.1f}")
        print(f"   Average tokens per query: {overall['avg_tokens']:.0f}")
        
        # Performance ranges
        print(f"\n⏱️  Response Time Range:")
        print(f"   Fastest: {overall['min_response_time']:.2f}s")
        print(f"   Slowest: {overall['max_response_time']:.2f}s")
        
        # Category breakdown
        print(f"\n📂 Category Performance:")
        for category, cat_stats in stats['category_breakdown'].items():
            print(f"   {category.upper()}:")
            print(f"     Success: {cat_stats['successful_tests']}/{cat_stats['total_tests']} ({cat_stats['success_rate']:.1f}%)")
            print(f"     Avg quality: {cat_stats['avg_quality_score']:.1f}/100")
            print(f"     Avg time: {cat_stats['avg_response_time']:.2f}s")
            print(f"     Avg sources: {cat_stats['avg_sources']:.1f}")
    
    def test_category_filtering(self) -> Dict[str, Any]:
        """Test category-specific filtering functionality."""
        print(f"\n🔍 TESTING CATEGORY FILTERING")
        print(f"=" * 40)
        
        categories = ['tech', 'sport', 'business', 'entertainment', 'politics']
        test_question = "What important developments or news are mentioned?"
        
        results = {}
        
        for category in categories:
            print(f"\n📂 Testing category: {category}")
            
            result = self.rag.answer_question(
                question=test_question,
                category_filter=category
            )
            
            # Analyze category consistency
            source_categories = []
            for source in result['sources']:
                source_cat = source.get('metadata', {}).get('category', 'unknown')
                source_categories.append(source_cat)
            
            category_consistency = (
                sum(1 for cat in source_categories if cat == category) / 
                len(source_categories) if source_categories else 0
            ) * 100
            
            results[category] = {
                'sources_found': len(result['sources']),
                'source_categories': source_categories,
                'category_consistency': category_consistency,
                'response_time': result['response_time'],
                'answer_length': len(result['answer'])
            }
            
            print(f"   Sources found: {len(result['sources'])}")
            print(f"   Category consistency: {category_consistency:.1f}%")
            print(f"   Response time: {result['response_time']:.2f}s")
        
        return results
    
    def save_test_report(self, test_results: Dict[str, Any]) -> Path:
        """Save comprehensive test report to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = Path("data/test_reports") / f"rag_test_report_{timestamp}.json"
        
        # Create directory if needed
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report data
        report_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'test_mode': self.test_mode,
            'summary': test_results['statistics'],
            'detailed_results': test_results['results'],
            'rag_config': self.rag.get_statistics()
        }
        
        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Test report saved: {report_file}")
        return report_file


def main():
    """Main testing function."""
    print("🚀 BASELINE RAG COMPREHENSIVE TESTING")
    print("=" * 60)
    
    # Initialize tester
    tester = RAGTester(test_mode=True)
    
    # Run comprehensive test suite
    test_results = tester.run_comprehensive_test()
    
    # Test category filtering
    print("\n" + "="*60)
    category_filter_results = tester.test_category_filtering()
    
    # Print final system statistics
    print("\n" + "="*60)
    tester.rag.print_statistics()
    
    # Save test report
    print("\n" + "="*60)
    report_file = tester.save_test_report(test_results)
    
    print(f"\n✅ COMPREHENSIVE TESTING COMPLETE!")
    print(f"📊 Overall success rate: {test_results['statistics']['overall']['success_rate']:.1f}%")
    print(f"📄 Detailed report: {report_file}")


if __name__ == "__main__":
    main()