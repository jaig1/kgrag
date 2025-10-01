#!/usr/bin/env python3
"""
Phase 6 - Script 5: KG-Enhanced RAG

This module provides the main KG-Enhanced RAG system that combines:
1. Graph-enhanced retrieval via HybridRetriever
2. Knowledge graph-aware prompt engineering
3. Enhanced answer generation with entity reasoning
4. Comprehensive result tracking and explanation
"""

import sys
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Import required components
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.context_formatter import ContextFormatter
from retrieval.response_tracker import ResponseTracker
from storage.vector_store import VectorStore
from storage.embedder import Embedder

# OpenAI API for enhanced generation
import openai
from openai import OpenAI


class KGEnhancedRAG:
    """
    Knowledge Graph-Enhanced RAG system that combines graph-enhanced retrieval
    with knowledge graph-aware prompt engineering for superior answer generation.
    """
    
    def __init__(self, 
                 response_tracker: Optional[ResponseTracker] = None,
                 model_name: str = "gpt-3.5-turbo",  # Faster model for demo
                 max_sources: int = 6,  # Reduced for faster processing
                 temperature: float = 0.1):
        """
        Initialize the KG-Enhanced RAG system.
        
        Args:
            response_tracker: Optional response tracker for logging
            model_name: OpenAI model for answer generation
            max_sources: Maximum number of sources to retrieve
            temperature: Temperature for answer generation
        """
        print("🚀 Initializing KG-Enhanced RAG...")
        
        # Configuration
        self.model_name = model_name
        self.max_sources = max_sources
        self.temperature = temperature
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize core components
        print("🔄 Initializing HybridRetriever...")
        self.hybrid_retriever = HybridRetriever()
        
        print("📝 Initializing ContextFormatter...")
        self.context_formatter = ContextFormatter()
        
        # Initialize response tracker
        if response_tracker is None:
            log_dir = Path("data/logs")
            log_dir.mkdir(exist_ok=True)
            self.response_tracker = ResponseTracker(
                log_dir=log_dir,
                log_filename="kg_enhanced_rag_queries.jsonl"
            )
        else:
            self.response_tracker = response_tracker
        
        # Statistics tracking
        self.stats = {
            'queries_processed': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'total_response_time': 0.0,
            'total_retrieval_time': 0.0,
            'total_generation_time': 0.0,
            'average_response_time': 0.0,
            'graph_enhanced_queries': 0,
            'fallback_queries': 0
        }
        
        print(f"✅ KG-Enhanced RAG initialized successfully")
        print(f"   Model: {self.model_name}")
        print(f"   Max sources: {self.max_sources}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Response tracker: {self.response_tracker.log_file}")
    
    def generate_answer(self, 
                       question: str, 
                       category_filter: Optional[str] = None,
                       include_explanation: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive answer using KG-enhanced retrieval and generation.
        
        Args:
            question: User's question
            category_filter: Optional category to filter sources
            include_explanation: Whether to include graph enhancement explanation
            
        Returns:
            Dict containing answer, sources, metadata, and explanations
        """
        print(f"\n🔍 KG-ENHANCED RAG QUERY: '{question}'")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Graph-enhanced retrieval
            print("📊 Step 1: Graph-Enhanced Retrieval...")
            retrieval_start = time.time()
            
            retrieval_result = self.hybrid_retriever.retrieve(query=question)
            
            retrieval_time = time.time() - retrieval_start
            print(f"✅ Retrieved {len(retrieval_result['results'])} sources in {retrieval_time:.2f}s")
            
            if not retrieval_result['success'] or not retrieval_result['results']:
                return self._handle_no_results(question, start_time, retrieval_time)
            
            # Step 2: Prepare graph-aware context
            print("🧠 Step 2: Preparing Graph-Aware Context...")
            context_data = self._prepare_graph_context(retrieval_result, category_filter)
            
            # Step 3: Generate enhanced prompts
            print("✨ Step 3: Creating Enhanced Prompts...")
            system_prompt = self._create_enhanced_system_prompt(retrieval_result['graph_data'])
            user_prompt = self._create_enhanced_user_prompt(question, context_data, retrieval_result['graph_data'])
            
            # Step 4: Generate answer
            print("🤖 Step 4: Generating KG-Enhanced Answer...")
            generation_start = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            generation_time = time.time() - generation_start
            
            # Step 5: Process response
            answer = response.choices[0].message.content
            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            # Step 6: Create comprehensive result
            total_time = time.time() - start_time
            
            result = self._create_result(
                question=question,
                answer=answer,
                retrieval_result=retrieval_result,
                context_data=context_data,
                response_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                token_usage=token_usage,
                include_explanation=include_explanation
            )
            
            # Step 7: Update statistics and log
            self._update_statistics(result, retrieval_result)
            self._log_interaction(result)
            
            print(f"✅ KG-Enhanced answer generated in {total_time:.2f}s using {token_usage['total_tokens']} tokens")
            
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            return self._handle_error(question, str(e), error_time)
    
    def _prepare_graph_context(self, retrieval_result: Dict[str, Any], category_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare graph-aware context from retrieval results.
        
        Args:
            retrieval_result: Results from hybrid retriever
            category_filter: Optional category filter
            
        Returns:
            Dict containing formatted context and metadata
        """
        sources = []
        
        # Filter and format sources
        for i, doc in enumerate(retrieval_result['results'], 1):
            # Apply category filter if specified
            if category_filter:
                doc_category = doc.get('metadata', {}).get('category', '').lower()
                if category_filter.lower() not in doc_category:
                    continue
            
            # Format source with enhanced metadata
            source = {
                'source_num': len(sources) + 1,
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'score': doc.get('score', 0.0),
                'article_id': doc.get('article_id', 'unknown'),
                'chunk_id': doc.get('chunk_id', 'unknown')
            }
            sources.append(source)
        
        # Format context using ContextFormatter
        formatted_context, formatting_stats = self.context_formatter.format_chunks(sources)
        
        return {
            'sources': sources,
            'formatted_context': formatted_context,
            'formatting_stats': formatting_stats,
            'total_sources': len(sources),
            'category_filter': category_filter
        }
    
    def _create_enhanced_system_prompt(self, graph_data: Dict[str, Any]) -> str:
        """
        Create a knowledge graph-aware system prompt.
        
        Args:
            graph_data: Graph enhancement data from retrieval
            
        Returns:
            Enhanced system prompt string
        """
        base_prompt = """You are a knowledge graph-enhanced AI assistant that provides comprehensive, well-sourced answers based on retrieved context and entity relationships.

CORE CAPABILITIES:
- Analyze entity relationships and connections from knowledge graphs
- Synthesize information across multiple related sources  
- Provide clear source attribution using [Source N] notation
- Explain reasoning using entity connections and graph insights
- Generate comprehensive answers that leverage both direct and indirect relationships

ANSWER STRUCTURE:
1. Direct answer to the question
2. Supporting evidence with source citations
3. Entity relationship insights (when relevant)
4. Comprehensive synthesis of available information

GUIDELINES:
- Always cite sources using [Source N] notation for all claims
- Only use information explicitly provided in the context
- When entity relationships provide additional insights, explain the connections
- If context is insufficient, clearly state limitations
- Maintain professional, informative tone
- Prioritize accuracy over completeness"""

        # Add graph-specific guidance if graph data is available
        graph_enhancement = graph_data.get('success', False) if graph_data else False
        
        if graph_enhancement:
            extracted_entities = graph_data.get('extracted_entities', [])
            related_entities = graph_data.get('related_entities', [])
            
            graph_guidance = f"\n\nKNOWLEDGE GRAPH CONTEXT:"
            
            if extracted_entities:
                graph_guidance += f"\n- Primary entities in query: {', '.join(extracted_entities[:5])}"
            
            if related_entities:
                graph_guidance += f"\n- Related entities discovered: {', '.join(related_entities[:8])}"
                graph_guidance += "\n- Consider how these related entities might provide additional context or connections"
            
            graph_guidance += "\n- Use entity relationships to provide richer, more comprehensive answers"
            
            return base_prompt + graph_guidance
        
        return base_prompt
    
    def _create_enhanced_user_prompt(self, question: str, context_data: Dict[str, Any], graph_data: Dict[str, Any]) -> str:
        """
        Create an enhanced user prompt with graph context.
        
        Args:
            question: User's question
            context_data: Formatted context data
            graph_data: Graph enhancement data
            
        Returns:
            Enhanced user prompt string
        """
        prompt_parts = []
        
        # Add context
        prompt_parts.append(f"CONTEXT:\n{context_data['formatted_context']}")
        
        # Add graph insights if available
        if graph_data.get('success', False):
            graph_section = "\nKNOWLEDGE GRAPH INSIGHTS:"
            
            # Extracted entities
            extracted_entities = graph_data.get('extracted_entities', [])
            if extracted_entities:
                graph_section += f"\n- Query entities: {', '.join(extracted_entities)}"
            
            # Related entities with context
            related_entities = graph_data.get('related_entities', [])
            entity_contexts = graph_data.get('entity_contexts', {})
            
            if related_entities:
                graph_section += f"\n- Related entities: {', '.join(related_entities[:6])}"
                
                # Add entity type information if available
                if entity_contexts:
                    entity_types = {}
                    for entity in related_entities[:6]:
                        if entity in entity_contexts:
                            entity_type = entity_contexts[entity].get('type', 'unknown')
                            if entity_type not in entity_types:
                                entity_types[entity_type] = []
                            entity_types[entity_type].append(entity_contexts[entity].get('name', entity))
                    
                    if entity_types:
                        graph_section += "\n- Entity relationships:"
                        for etype, names in entity_types.items():
                            graph_section += f"\n  * {etype.capitalize()}: {', '.join(names[:3])}"
            
            # Connected articles
            connected_articles = len(graph_data.get('connected_articles', []))
            if connected_articles > 0:
                graph_section += f"\n- Connected articles in knowledge graph: {connected_articles}"
            
            prompt_parts.append(graph_section)
        
        # Add question
        prompt_parts.append(f"\nQUESTION: {question}")
        
        # Add instructions
        instructions = """
INSTRUCTIONS:
Provide a comprehensive answer using the context above. If knowledge graph insights are available, use entity relationships to enrich your response. Always cite sources using [Source N] notation."""
        
        prompt_parts.append(instructions)
        
        return "\n".join(prompt_parts)
    
    def _create_result(self, question: str, answer: str, retrieval_result: Dict[str, Any], 
                      context_data: Dict[str, Any], response_time: float, retrieval_time: float,
                      generation_time: float, token_usage: Dict[str, Any], 
                      include_explanation: bool = True) -> Dict[str, Any]:
        """Create comprehensive result dictionary."""
        
        # Basic result structure
        result = {
            'question': question,
            'answer': answer,
            'sources': context_data['sources'],
            'response_time': response_time,
            'token_usage': token_usage,
            'metadata': {
                'model': self.model_name,
                'temperature': self.temperature,
                'sources_requested': self.max_sources,
                'sources_retrieved': len(context_data['sources']),
                'category_filter': context_data.get('category_filter'),
                'context_length': len(context_data['formatted_context']),
                'retrieval_method': 'kg_enhanced_hybrid',
                'retrieval_time': retrieval_time,
                'generation_time': generation_time
            }
        }
        
        # Add graph enhancement details
        graph_data = retrieval_result.get('graph_data', {})
        graph_enhancement = {
            'enhancement_applied': graph_data.get('success', False),
            'entities_extracted': len(graph_data.get('extracted_entities', [])),
            'entities_related': len(graph_data.get('related_entities', [])),
            'articles_connected': len(graph_data.get('connected_articles', [])),
            'query_enhanced': len(retrieval_result.get('enhanced_query', '')) > len(question)
        }
        
        result['graph_enhancement'] = graph_enhancement
        
        # Store retrieval result and enhanced query for transparency
        result['retrieval_result'] = retrieval_result
        result['enhanced_query'] = retrieval_result.get('enhanced_query', question)
        
        # Add explanation if requested
        if include_explanation:
            result['explanation'] = self._generate_explanation(retrieval_result, graph_enhancement)
        
        return result
    
    def _generate_explanation(self, retrieval_result: Dict[str, Any], graph_enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation of the KG-enhanced process."""
        
        explanation = {
            'process_overview': "KG-Enhanced RAG combines knowledge graph insights with vector retrieval for comprehensive answers.",
            'steps': []
        }
        
        # Step 1: Query Analysis
        graph_data = retrieval_result.get('graph_data', {})
        extracted_entities = graph_data.get('extracted_entities', [])
        
        explanation['steps'].append({
            'step': 1,
            'name': "Query Analysis & Entity Extraction",
            'description': f"Extracted {len(extracted_entities)} entities from query",
            'details': {
                'entities_found': extracted_entities,
                'success': len(extracted_entities) > 0
            }
        })
        
        # Step 2: Graph Traversal
        related_entities = graph_data.get('related_entities', [])
        connected_articles = graph_data.get('connected_articles', [])
        
        explanation['steps'].append({
            'step': 2,
            'name': "Knowledge Graph Traversal",
            'description': f"Discovered {len(related_entities)} related entities across {len(connected_articles)} articles",
            'details': {
                'related_entities': related_entities[:10],  # Top 10
                'connected_articles': len(connected_articles),
                'success': len(related_entities) > 0
            }
        })
        
        # Step 3: Enhanced Retrieval
        enhanced_query = retrieval_result.get('enhanced_query', '')
        original_query = retrieval_result.get('query', '')
        
        explanation['steps'].append({
            'step': 3,
            'name': "Graph-Informed Retrieval",
            'description': f"Enhanced query with graph insights for improved retrieval",
            'details': {
                'query_enhanced': len(enhanced_query) > len(original_query),
                'enhancement_method': "Entity and relationship context added to search query",
                'results_count': len(retrieval_result.get('results', []))
            }
        })
        
        # Step 4: Enhanced Generation
        explanation['steps'].append({
            'step': 4,
            'name': "Knowledge Graph-Aware Answer Generation",
            'description': "Generated answer using graph-aware prompts and entity relationship context",
            'details': {
                'prompt_enhancement': "System and user prompts enhanced with graph insights",
                'entity_awareness': graph_enhancement['enhancement_applied'],
                'relationship_context': len(related_entities) > 0
            }
        })
        
        return explanation
    
    def _handle_no_results(self, question: str, start_time: float, retrieval_time: float) -> Dict[str, Any]:
        """Handle case when no results are retrieved."""
        
        total_time = time.time() - start_time
        
        result = {
            'question': question,
            'answer': f"I don't have sufficient relevant information in my knowledge base to answer your question about '{question}'. The available sources don't contain specific details related to your inquiry.",
            'sources': [],
            'response_time': total_time,
            'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'metadata': {
                'model': self.model_name,
                'sources_retrieved': 0,
                'retrieval_method': 'kg_enhanced_hybrid',
                'retrieval_time': retrieval_time,
                'generation_time': 0.0,
                'error': 'No relevant sources found'
            },
            'graph_enhancement': {
                'enhancement_applied': False,
                'entities_extracted': 0,
                'entities_related': 0,
                'articles_connected': 0,
                'query_enhanced': False
            }
        }
        
        self.stats['failed_responses'] += 1
        return result
    
    def _handle_error(self, question: str, error_message: str, error_time: float) -> Dict[str, Any]:
        """Handle errors during processing."""
        
        result = {
            'question': question,
            'answer': f"I encountered an error while processing your question. Please try again or rephrase your query.",
            'sources': [],
            'response_time': error_time,
            'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'metadata': {
                'model': self.model_name,
                'sources_retrieved': 0,
                'retrieval_method': 'kg_enhanced_hybrid',
                'error': error_message
            },
            'graph_enhancement': {
                'enhancement_applied': False,
                'entities_extracted': 0,
                'entities_related': 0,
                'articles_connected': 0,
                'query_enhanced': False
            },
            'error': error_message
        }
        
        self.stats['failed_responses'] += 1
        print(f"❌ Error processing query: {error_message}")
        return result
    
    def _update_statistics(self, result: Dict[str, Any], retrieval_result: Dict[str, Any]) -> None:
        """Update internal statistics."""
        
        self.stats['queries_processed'] += 1
        
        if result.get('error'):
            self.stats['failed_responses'] += 1
        else:
            self.stats['successful_responses'] += 1
        
        # Time statistics
        self.stats['total_response_time'] += result['response_time']
        self.stats['total_retrieval_time'] += result['metadata'].get('retrieval_time', 0.0)
        self.stats['total_generation_time'] += result['metadata'].get('generation_time', 0.0)
        
        # Calculate averages
        if self.stats['queries_processed'] > 0:
            self.stats['average_response_time'] = (
                self.stats['total_response_time'] / self.stats['queries_processed']
            )
        
        # Graph enhancement statistics
        if result.get('graph_enhancement', {}).get('enhancement_applied', False):
            self.stats['graph_enhanced_queries'] += 1
        else:
            self.stats['fallback_queries'] += 1
    
    def _log_interaction(self, result: Dict[str, Any]) -> None:
        """Log the interaction using ResponseTracker."""
        
        self.response_tracker.log_query(
            query=result['question'],
            answer=result['answer'],
            sources=result['sources'],
            response_time=result['response_time'],
            token_usage=result['token_usage'],
            retrieval_method='kg_enhanced_hybrid',
            metadata=result['metadata']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        
        base_stats = {
            'kg_enhanced_rag': self.stats.copy(),
            'hybrid_retriever': self.hybrid_retriever.stats.copy(),
            'response_tracker': self.response_tracker.get_session_statistics()
        }
        
        # Calculate success rate
        if self.stats['queries_processed'] > 0:
            base_stats['kg_enhanced_rag']['success_rate'] = (
                self.stats['successful_responses'] / self.stats['queries_processed']
            ) * 100
            
            base_stats['kg_enhanced_rag']['graph_enhancement_rate'] = (
                self.stats['graph_enhanced_queries'] / self.stats['queries_processed']
            ) * 100
        else:
            base_stats['kg_enhanced_rag']['success_rate'] = 0.0
            base_stats['kg_enhanced_rag']['graph_enhancement_rate'] = 0.0
        
        return base_stats
    
    def print_statistics(self) -> None:
        """Print comprehensive system statistics."""
        
        stats = self.get_statistics()
        kg_stats = stats['kg_enhanced_rag']
        
        print(f"\n📊 KG-ENHANCED RAG STATISTICS")
        print("=" * 60)
        
        # Query processing stats
        print(f"🔍 Query Processing:")
        print(f"   Total queries: {kg_stats['queries_processed']}")
        print(f"   Successful responses: {kg_stats['successful_responses']}")
        print(f"   Failed responses: {kg_stats['failed_responses']}")
        print(f"   Success rate: {kg_stats['success_rate']:.1f}%")
        
        # Performance stats
        print(f"\n⏱️  Performance:")
        print(f"   Average response time: {kg_stats['average_response_time']:.2f}s")
        print(f"   Total retrieval time: {kg_stats['total_retrieval_time']:.2f}s")
        print(f"   Total generation time: {kg_stats['total_generation_time']:.2f}s")
        
        # Graph enhancement stats
        print(f"\n🕸️  Graph Enhancement:")
        print(f"   Graph-enhanced queries: {kg_stats['graph_enhanced_queries']}")
        print(f"   Fallback queries: {kg_stats['fallback_queries']}")
        print(f"   Enhancement rate: {kg_stats['graph_enhancement_rate']:.1f}%")
        
        # Component stats
        retriever_stats = stats['hybrid_retriever']
        print(f"\n🔄 HybridRetriever:")
        print(f"   Graph searches: {retriever_stats['graph_searches']}")
        print(f"   Vector searches: {retriever_stats['vector_searches']}")
        print(f"   Average retrieval time: {retriever_stats['average_retrieval_time']:.2f}s")


# Test function for single KG-Enhanced RAG query
def test_single_kg_enhanced_query() -> Dict[str, Any]:
    """Test KG-Enhanced RAG with a single comprehensive query."""
    
    print("🧪 TESTING KG-ENHANCED RAG")
    print("=" * 60)
    
    # Initialize KG-Enhanced RAG
    kg_rag = KGEnhancedRAG()
    
    # Test with a query that should benefit from graph enhancement
    test_query = "What major technology companies are mentioned and what are their key developments?"
    
    print(f"\n📋 Testing with query: '{test_query}'")
    
    # Generate answer
    result = kg_rag.generate_answer(
        question=test_query,
        include_explanation=True
    )
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Success: {not result.get('error', False)}")
    print(f"   Response time: {result['response_time']:.2f}s")
    print(f"   Sources used: {len(result['sources'])}")
    print(f"   Tokens used: {result['token_usage']['total_tokens']}")
    
    # Show graph enhancement
    graph_enhancement = result.get('graph_enhancement', {})
    print(f"\n🕸️  GRAPH ENHANCEMENT:")
    print(f"   Applied: {graph_enhancement.get('enhancement_applied', False)}")
    print(f"   Entities extracted: {graph_enhancement.get('entities_extracted', 0)}")
    print(f"   Related entities: {graph_enhancement.get('entities_related', 0)}")
    print(f"   Connected articles: {graph_enhancement.get('articles_connected', 0)}")
    
    # Show answer preview
    answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
    print(f"\n📝 ANSWER PREVIEW:")
    print(f"   {answer_preview}")
    
    # Show explanation if available
    if 'explanation' in result:
        explanation = result['explanation']
        print(f"\n🔍 PROCESS EXPLANATION:")
        print(f"   Overview: {explanation['process_overview']}")
        print(f"   Steps completed: {len(explanation['steps'])}")
        
        for step in explanation['steps']:
            print(f"   {step['step']}. {step['name']}: {step['description']}")
    
    # Print statistics
    kg_rag.print_statistics()
    
    print(f"\n✅ KG-Enhanced RAG test complete!")
    
    return result


if __name__ == "__main__":
    # Test with single query
    result = test_single_kg_enhanced_query()