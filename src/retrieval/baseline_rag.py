"""
Baseline RAG implementation combining vector retrieval with language model generation.

This module provides the main RAG pipeline that:
1. Embeds user queries using OpenAI embeddings
2. Retrieves relevant context from ChromaDB vector store  
3. Formats context using prompt templates
4. Generates answers using OpenAI Chat API
5. Tracks all queries and responses
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv(project_root / ".env")

from src.storage.vector_store import VectorStore
from src.storage.embedder import Embedder
from src.retrieval.prompts import BASELINE_SYSTEM_PROMPT, BASELINE_USER_TEMPLATE
from src.retrieval.context_formatter import ContextFormatter
from src.retrieval.response_tracker import ResponseTracker

# OpenAI API for chat completions
import openai
from openai import OpenAI


class BaselineRAG:
    """
    Baseline RAG implementation using vector similarity search and GPT generation.
    
    Provides a complete question-answering pipeline that retrieves relevant context
    and generates informed answers with proper source citations.
    """
    
    def __init__(self, 
                 vector_store: VectorStore = None,
                 embedder: Embedder = None,
                 context_formatter: ContextFormatter = None,
                 response_tracker: ResponseTracker = None,
                 model_name: str = "gpt-4-turbo-preview",
                 max_sources: int = 5,
                 temperature: float = 0.1):
        """
        Initialize the Baseline RAG system.
        
        Args:
            vector_store: Vector store for similarity search
            embedder: Embedder for query embeddings
            context_formatter: Context formatter for prompt preparation
            response_tracker: Response tracker for logging
            model_name: OpenAI model name for generation
            max_sources: Maximum number of sources to retrieve
            temperature: Generation temperature (0.0 = deterministic)
        """
        # Initialize components (use defaults if not provided)
        self.vector_store = vector_store or VectorStore(
            db_path=Path("data/chroma_db"),
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.embedder = embedder or Embedder(api_key=os.getenv('OPENAI_API_KEY'))
        self.context_formatter = context_formatter or ContextFormatter()
        self.response_tracker = response_tracker or ResponseTracker()
        
        # Generation settings
        self.model_name = model_name
        self.max_sources = max_sources
        self.temperature = temperature
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        print(f"🤖 BaselineRAG initialized:")
        print(f"   Model: {self.model_name}")
        print(f"   Max sources: {self.max_sources}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Vector store collection: {self.vector_store.collection.name}")
    
    def answer_question(self, 
                       question: str, 
                       category_filter: Optional[str] = None,
                       custom_max_sources: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User question to answer
            category_filter: Optional category to filter sources (e.g., 'tech', 'sport')
            custom_max_sources: Optional override for max sources
            
        Returns:
            Dictionary containing answer, sources, timing, and metadata
        """
        start_time = time.time()
        
        try:
            print(f"\n🔍 Processing question: '{question}'")
            
            # Step 1: Retrieve relevant sources using text query
            print("📚 Searching for relevant sources...")
            max_sources = custom_max_sources or self.max_sources
            
            # Prepare filter if category specified
            filter_metadata = None
            if category_filter:
                filter_metadata = {"category": category_filter}
                print(f"   Filtering by category: {category_filter}")
            
            search_results = self.vector_store.similarity_search(
                query=question,
                top_k=max_sources,
                filter_metadata=filter_metadata
            )
            
            # Convert search results to expected format
            sources = []
            distances = search_results.get('distances', [])
            
            for i in range(len(search_results['documents'])):
                distance = distances[i] if i < len(distances) else None
                
                # Convert distance to similarity score 
                # ChromaDB uses L2 (Euclidean) distance, convert to similarity
                score = 0.0
                if distance is not None:
                    # For L2 distance, convert to similarity: 1 / (1 + distance)
                    score = 1.0 / (1.0 + distance)
                
                source = {
                    'document': search_results['documents'][i],
                    'metadata': search_results['metadatas'][i],
                    'distance': distance,
                    'score': score  # Add score directly
                }
                sources.append(source)
                
                # Debug output with more precision
                print(f"   Source {i+1}: distance={distance:.6f}, score={score:.6f}")
            
            print(f"   Retrieved {len(sources)} sources")
            
            # Step 2: Format context for prompt
            print("📝 Formatting context...")
            formatted_context, formatting_stats = self.context_formatter.format_chunks(sources, query=question)
            
            # Step 3: Prepare prompt
            user_prompt = BASELINE_USER_TEMPLATE.format(
                context=formatted_context,
                query=question
            )
            
            # Step 4: Generate answer using OpenAI
            print("🧠 Generating answer...")
            response = self._generate_answer(user_prompt)
            
            # Step 5: Extract response data
            answer = response.choices[0].message.content
            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            # Step 6: Calculate timing
            end_time = time.time()
            response_time = end_time - start_time
            
            # Step 7: Prepare result
            result = {
                'question': question,
                'answer': answer,
                'sources': sources,
                'response_time': response_time,
                'token_usage': token_usage,
                'metadata': {
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'sources_requested': max_sources,
                    'sources_retrieved': len(sources),
                    'category_filter': category_filter,
                    'context_length': len(formatted_context)
                }
            }
            
            # Step 8: Log the interaction
            self.response_tracker.log_query(
                query=question,
                answer=answer,
                sources=sources,
                response_time=response_time,
                token_usage=token_usage,
                retrieval_method='baseline_vector',
                metadata=result['metadata']
            )
            
            print(f"✅ Answer generated in {response_time:.2f}s using {token_usage['total_tokens']} tokens")
            
            return result
            
        except Exception as e:
            error_message = str(e)
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"❌ Error generating answer: {error_message}")
            
            # Log the error
            self.response_tracker.log_query(
                query=question,
                answer="",
                sources=[],
                response_time=response_time,
                token_usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                retrieval_method='baseline_vector',
                error=error_message,
                metadata={'category_filter': category_filter}
            )
            
            # Return error result
            return {
                'question': question,
                'answer': f"I apologize, but I encountered an error while processing your question: {error_message}",
                'sources': [],
                'response_time': response_time,
                'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'error': error_message,
                'metadata': {'category_filter': category_filter}
            }
    
    def _generate_answer(self, user_prompt: str) -> Any:
        """Generate answer using OpenAI Chat API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000,  # Reasonable limit for answers
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            return response
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            raise
    
    def batch_answer_questions(self, 
                              questions: List[str],
                              category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions to answer
            category_filter: Optional category filter to apply to all questions
            
        Returns:
            List of answer results
        """
        print(f"\n📋 Processing {len(questions)} questions in batch")
        if category_filter:
            print(f"   Category filter: {category_filter}")
        
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            
            result = self.answer_question(
                question=question,
                category_filter=category_filter
            )
            
            results.append(result)
            
            # Brief pause between questions to avoid rate limits
            if i < len(questions):
                time.sleep(0.5)
        
        print(f"\n✅ Batch processing complete: {len(results)} answers generated")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            'response_tracker': self.response_tracker.get_session_statistics(),
            'vector_store': {
                'collection_name': self.vector_store.collection.name,
                'total_documents': self.vector_store.collection.count()
            },
            'embedder': {
                'cache_size': len(self.embedder.cache),
                'model': self.embedder.model
            },
            'rag_config': {
                'model': self.model_name,
                'max_sources': self.max_sources,
                'temperature': self.temperature
            }
        }
    
    def print_statistics(self) -> None:
        """Print comprehensive system statistics."""
        stats = self.get_statistics()
        
        print(f"\n📊 BASELINE RAG STATISTICS")
        print(f"=" * 50)
        
        # RAG configuration
        rag_config = stats['rag_config']
        print(f"🤖 RAG Configuration:")
        print(f"   Model: {rag_config['model']}")
        print(f"   Max sources: {rag_config['max_sources']}")
        print(f"   Temperature: {rag_config['temperature']}")
        
        # Vector store info
        vs_stats = stats['vector_store']
        print(f"\n📚 Vector Store:")
        print(f"   Collection: {vs_stats['collection_name']}")
        print(f"   Total documents: {vs_stats['total_documents']:,}")
        
        # Embedder info
        embed_stats = stats['embedder']
        print(f"\n🔢 Embedder:")
        print(f"   Model: {embed_stats['model']}")
        print(f"   Cache size: {embed_stats['cache_size']:,}")
        
        # Response tracker session stats
        print(f"\n📝 Session Performance:")
        self.response_tracker.print_session_summary()


# Test mode functionality
def test_baseline_rag() -> None:
    """Test the Baseline RAG system with sample queries."""
    print("🧪 TESTING BASELINE RAG")
    print("=" * 50)
    
    # Initialize RAG system
    print("🚀 Initializing Baseline RAG...")
    rag = BaselineRAG(
        model_name="gpt-4-turbo-preview",
        max_sources=3,  # Use fewer sources for testing
        temperature=0.1
    )
    
    # Test queries covering different categories
    test_questions = [
        "What technology companies are mentioned in the articles?",
        "What sporting events or achievements are discussed?", 
        "Are there any political developments or policy changes mentioned?"
    ]
    
    print(f"\n🔍 Testing with {len(test_questions)} sample questions...")
    
    # Answer each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} TEST {i}/{len(test_questions)} {'='*20}")
        
        result = rag.answer_question(question)
        
        # Print results
        print(f"\n📝 Question: {result['question']}")
        print(f"🎯 Answer:\n{result['answer']}")
        print(f"\n📚 Sources: {len(result['sources'])}")
        print(f"⏱️  Time: {result['response_time']:.2f}s")
        print(f"🔢 Tokens: {result['token_usage']['total_tokens']}")
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
        
        # Add separator for readability
        print(f"\n{'-'*80}")
    
    # Print final statistics
    rag.print_statistics()
    
    print(f"\n✅ Baseline RAG testing complete!")


def test_category_filtering() -> None:
    """Test category-specific filtering functionality."""
    print("🧪 TESTING CATEGORY FILTERING")
    print("=" * 50)
    
    rag = BaselineRAG(max_sources=2, temperature=0.0)
    
    # Test category-specific queries
    test_cases = [
        ("What technology developments are mentioned?", "tech"),
        ("What sports news is covered?", "sport"),
        ("What entertainment news is discussed?", "entertainment"),
        ("What business news is reported?", "business"),
        ("What political events are covered?", "politics")
    ]
    
    for question, category in test_cases:
        print(f"\n🔍 Testing category filter: '{category}'")
        
        result = rag.answer_question(
            question=question,
            category_filter=category
        )
        
        print(f"   Question: {question}")
        print(f"   Sources found: {len(result['sources'])}")
        print(f"   Response time: {result['response_time']:.2f}s")
        
        # Check if sources match the category
        if result['sources']:
            categories = [s.get('metadata', {}).get('category', 'unknown') for s in result['sources']]
            print(f"   Source categories: {set(categories)}")
    
    print(f"\n✅ Category filtering testing complete!")


if __name__ == "__main__":
    # Run basic tests
    test_baseline_rag()
    
    # Test category filtering
    print("\n" + "="*60)
    test_category_filtering()