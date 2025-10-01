"""
Response tracker for logging RAG queries and responses.

Tracks all queries, responses, and performance metrics for analysis and debugging.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class ResponseTracker:
    """
    Tracks and logs RAG system queries and responses with performance metrics.
    """
    
    def __init__(self, log_dir: Path = None, log_filename: str = "baseline_queries.jsonl"):
        """
        Initialize the ResponseTracker.
        
        Args:
            log_dir: Directory to save log files
            log_filename: Name of the log file
        """
        self.log_dir = log_dir or Path("data/logs")
        self.log_file = self.log_dir / log_filename
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.session_stats = {
            'queries_processed': 0,
            'total_response_time': 0.0,
            'total_tokens_used': 0,
            'total_sources_retrieved': 0,
            'session_start': datetime.now().isoformat(),
            'errors': 0
        }
        
        print(f"📊 ResponseTracker initialized:")
        print(f"   Log file: {self.log_file}")
        print(f"   Log directory: {self.log_dir}")
    
    def log_query(self, 
                  query: str,
                  answer: str,
                  sources: List[Dict[str, Any]],
                  response_time: float,
                  token_usage: Dict[str, int],
                  retrieval_method: str = "baseline_vector",
                  error: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a query and its response to the JSONL file.
        
        Args:
            query: Original user query
            answer: Generated answer
            sources: List of source chunks used
            response_time: Time taken to generate response
            token_usage: Token counts (prompt_tokens, completion_tokens, total_tokens)
            retrieval_method: Method used for retrieval
            error: Error message if any
            metadata: Additional metadata
        """
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'sources': self._format_sources_for_log(sources),
            'response_time_seconds': response_time,
            'token_usage': token_usage,
            'retrieval_method': retrieval_method,
            'sources_count': len(sources),
            'answer_length': len(answer),
            'error': error,
            'metadata': metadata or {}
        }
        
        # Append to JSONL file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"❌ Failed to write to log file: {e}")
        
        # Update session statistics
        self._update_session_stats(log_entry)
        
        # Print summary
        self._print_query_summary(log_entry)
    
    def _format_sources_for_log(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for logging (remove full content to save space)."""
        formatted_sources = []
        
        for i, source in enumerate(sources):
            metadata = source.get('metadata', {})
            document = source.get('document', '')
            
            formatted_source = {
                'source_number': i + 1,
                'article_id': metadata.get('article_id', 'unknown'),
                'chunk_id': metadata.get('chunk_id', 'unknown'),
                'category': metadata.get('category', 'unknown'),
                'title': metadata.get('title', 'Unknown Title'),
                'position': metadata.get('position', 0),
                'content_preview': document[:100] + '...' if len(document) > 100 else document,
                'content_length': len(document),
                'distance': source.get('distance', None)  # Similarity score if available
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources
    
    def _update_session_stats(self, log_entry: Dict[str, Any]) -> None:
        """Update session statistics with the latest entry."""
        self.session_stats['queries_processed'] += 1
        self.session_stats['total_response_time'] += log_entry['response_time_seconds']
        
        token_usage = log_entry.get('token_usage', {})
        self.session_stats['total_tokens_used'] += token_usage.get('total_tokens', 0)
        self.session_stats['total_sources_retrieved'] += log_entry['sources_count']
        
        if log_entry.get('error'):
            self.session_stats['errors'] += 1
    
    def _print_query_summary(self, log_entry: Dict[str, Any]) -> None:
        """Print a summary of the logged query."""
        query_num = self.session_stats['queries_processed']
        
        print(f"\n📝 Query {query_num} Logged:")
        print(f"   Query: '{log_entry['query'][:60]}{'...' if len(log_entry['query']) > 60 else ''}'")
        print(f"   Sources: {log_entry['sources_count']}")
        print(f"   Response time: {log_entry['response_time_seconds']:.2f}s")
        print(f"   Tokens: {log_entry['token_usage'].get('total_tokens', 0):,}")
        
        if log_entry.get('error'):
            print(f"   ❌ Error: {log_entry['error']}")
        else:
            print(f"   ✅ Success")
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get current session statistics."""
        if self.session_stats['queries_processed'] > 0:
            avg_response_time = self.session_stats['total_response_time'] / self.session_stats['queries_processed']
            avg_sources = self.session_stats['total_sources_retrieved'] / self.session_stats['queries_processed']
            avg_tokens = self.session_stats['total_tokens_used'] / self.session_stats['queries_processed']
        else:
            avg_response_time = avg_sources = avg_tokens = 0
        
        return {
            **self.session_stats,
            'average_response_time': avg_response_time,
            'average_sources_per_query': avg_sources,
            'average_tokens_per_query': avg_tokens,
            'success_rate': (self.session_stats['queries_processed'] - self.session_stats['errors']) / max(1, self.session_stats['queries_processed']) * 100
        }
    
    def print_session_summary(self) -> None:
        """Print a summary of the current session."""
        stats = self.get_session_statistics()
        
        print(f"\n📊 SESSION SUMMARY")
        print(f"=" * 50)
        print(f"📝 Queries processed: {stats['queries_processed']}")
        print(f"⚡ Average response time: {stats['average_response_time']:.2f}s")
        print(f"📚 Average sources per query: {stats['average_sources_per_query']:.1f}")
        print(f"🎯 Average tokens per query: {stats['average_tokens_per_query']:.0f}")
        print(f"✅ Success rate: {stats['success_rate']:.1f}%")
        print(f"⏱️  Total response time: {stats['total_response_time']:.2f}s")
        print(f"🔢 Total tokens used: {stats['total_tokens_used']:,}")
        
        if stats['errors'] > 0:
            print(f"❌ Errors encountered: {stats['errors']}")
    
    def load_all_queries(self) -> List[Dict[str, Any]]:
        """Load all logged queries from the JSONL file."""
        queries = []
        
        if not self.log_file.exists():
            return queries
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        queries.append(json.loads(line))
        except Exception as e:
            print(f"❌ Failed to load queries from log: {e}")
        
        return queries
    
    def analyze_historical_performance(self) -> Dict[str, Any]:
        """Analyze performance across all logged queries."""
        queries = self.load_all_queries()
        
        if not queries:
            return {'message': 'No historical queries found'}
        
        # Calculate statistics
        response_times = [q['response_time_seconds'] for q in queries]
        source_counts = [q['sources_count'] for q in queries]
        token_counts = [q['token_usage'].get('total_tokens', 0) for q in queries]
        error_count = len([q for q in queries if q.get('error')])
        
        analysis = {
            'total_queries': len(queries),
            'date_range': {
                'earliest': queries[0]['timestamp'] if queries else None,
                'latest': queries[-1]['timestamp'] if queries else None
            },
            'response_time': {
                'average': sum(response_times) / len(response_times),
                'min': min(response_times),
                'max': max(response_times)
            },
            'sources_retrieved': {
                'average': sum(source_counts) / len(source_counts),
                'min': min(source_counts),
                'max': max(source_counts)
            },
            'token_usage': {
                'total': sum(token_counts),
                'average': sum(token_counts) / len(token_counts),
                'min': min(token_counts) if token_counts else 0,
                'max': max(token_counts) if token_counts else 0
            },
            'error_rate': error_count / len(queries) * 100,
            'success_rate': (len(queries) - error_count) / len(queries) * 100
        }
        
        return analysis
    
    def clear_logs(self) -> None:
        """Clear the log file (use with caution)."""
        if self.log_file.exists():
            self.log_file.unlink()
            print(f"🗑️  Cleared log file: {self.log_file}")
        
        # Reset session stats
        self.session_stats = {
            'queries_processed': 0,
            'total_response_time': 0.0,
            'total_tokens_used': 0,
            'total_sources_retrieved': 0,
            'session_start': datetime.now().isoformat(),
            'errors': 0
        }


# Test mode functionality
def test_response_tracker() -> None:
    """Test the ResponseTracker with sample data."""
    print("🧪 TESTING RESPONSE TRACKER")
    print("=" * 50)
    
    # Initialize tracker with test directory
    test_log_dir = Path("data/test_logs")
    tracker = ResponseTracker(log_dir=test_log_dir, log_filename="test_queries.jsonl")
    
    # Clear any existing test logs
    tracker.clear_logs()
    
    # Test sample queries
    test_queries = [
        {
            'query': 'What companies are mentioned in technology news?',
            'answer': 'Based on the provided context, several technology companies are mentioned including Microsoft, Google, and Apple. [Source 1] discusses Microsoft\'s latest developments, while [Source 2] covers Google\'s AI initiatives.',
            'sources': [
                {
                    'document': 'Microsoft announced new AI features in their latest software update...',
                    'metadata': {
                        'article_id': 'tech_001',
                        'chunk_id': 'tech_001_chunk_0',
                        'category': 'tech',
                        'title': 'Microsoft AI Update',
                        'position': 0
                    },
                    'distance': 0.85
                },
                {
                    'document': 'Google released their quarterly earnings showing growth in cloud services...',
                    'metadata': {
                        'article_id': 'tech_002', 
                        'chunk_id': 'tech_002_chunk_1',
                        'category': 'tech',
                        'title': 'Google Q4 Earnings',
                        'position': 1
                    },
                    'distance': 0.78
                }
            ],
            'response_time': 2.3,
            'token_usage': {'prompt_tokens': 450, 'completion_tokens': 120, 'total_tokens': 570}
        },
        {
            'query': 'What sporting events are discussed?',
            'answer': 'The context mentions several sporting events including the championship tennis match and the football league results. [Source 1] provides details about the tennis tournament.',
            'sources': [
                {
                    'document': 'The championship tennis match concluded with an exciting final set...',
                    'metadata': {
                        'article_id': 'sport_001',
                        'chunk_id': 'sport_001_chunk_0', 
                        'category': 'sport',
                        'title': 'Tennis Championship Final',
                        'position': 0
                    },
                    'distance': 0.92
                }
            ],
            'response_time': 1.8,
            'token_usage': {'prompt_tokens': 320, 'completion_tokens': 85, 'total_tokens': 405}
        }
    ]
    
    # Log test queries
    for i, test_data in enumerate(test_queries):
        print(f"\n📝 Logging test query {i+1}...")
        
        tracker.log_query(
            query=test_data['query'],
            answer=test_data['answer'],
            sources=test_data['sources'],
            response_time=test_data['response_time'],
            token_usage=test_data['token_usage'],
            retrieval_method='baseline_vector',
            metadata={'test_query': True, 'query_id': i+1}
        )
    
    # Print session summary
    tracker.print_session_summary()
    
    # Test historical analysis
    print(f"\n🔍 Historical Analysis:")
    analysis = tracker.analyze_historical_performance()
    print(f"   Total queries analyzed: {analysis['total_queries']}")
    print(f"   Average response time: {analysis['response_time']['average']:.2f}s")
    print(f"   Average sources: {analysis['sources_retrieved']['average']:.1f}")
    print(f"   Success rate: {analysis['success_rate']:.1f}%")
    
    # Test loading queries
    loaded_queries = tracker.load_all_queries()
    print(f"\n📂 Loaded {len(loaded_queries)} queries from log file")
    
    print(f"\n✅ ResponseTracker testing complete!")
    print(f"   Test log file: {tracker.log_file}")


if __name__ == "__main__":
    test_response_tracker()